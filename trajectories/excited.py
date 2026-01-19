from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from factory import instantiate

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig
from .fourier import FourierTrajectory, FourierTrajectoryConfig
from .spline import SplineTrajectoryConfig
from .window import WindowTrajectory, WindowTrajectoryConfig
from .windowed_fourier import WindowedFourierTrajectoryConfig


@dataclass(kw_only=True)
class ExcitedTrajectoryConfig(WindowedFourierTrajectoryConfig):
    # Use Any instead of Union to avoid OmegaConf limitation with Union of containers

    # Optional window trajectory config (will be auto-created if not provided)
    guide_config: SplineTrajectoryConfig | None = None

    # MuJoCo model paths for kinematics_func construction (optional)
    manipulator: str | None = None
    object: str | None = None
    ee_body_name: str = "link6"  # End-effector body name

    target_class: str = "ExcitedTrajectory"


class ExcitedTrajectory(BaseTrajectory):
    def __init__(self, cfg: ExcitedTrajectoryConfig, *args, **kwargs):
        """
        Excited Trajectory: A base trajectory enriched with an optimized Fourier series excitation component.

        The Fourier component is weighted by a window function to ensure zero position, velocity,
        and acceleration at the boundaries (start and end), preserving the boundary conditions
        of the main trajectory.

        Args:
            cfg: Configuration object.
            kinematics_func (callable): f(q, dq, ddq) -> regressor_matrix. (passed via kwargs)
        """
        super().__init__(cfg, *args, **kwargs)
        # Instantiate main_trajectory using factory.instantiate
        # Handle the case where main_trajectory is a dict (from YAML/Hydra) instead of a config object
        main_traj_cfg = cfg.main_trajectory
        if isinstance(main_traj_cfg, dict):
            # Convert dict to appropriate config class based on target_class
            target_class_name = main_traj_cfg.get("target_class", "SplineTrajectory")
            if "Spline" in target_class_name:
                from .spline import SplineTrajectoryConfig
                main_traj_cfg = SplineTrajectoryConfig(**main_traj_cfg)
            else:
                main_traj_cfg = FourierTrajectoryConfig(**main_traj_cfg)

        self.main_trajectory = instantiate(main_traj_cfg, *args, **kwargs)

        # Assuming main_trajectory exposes num_joints via its data or attribute
        # We need to generate it once to get basic info if not available
        # But let's assume valid access.
        if hasattr(self.main_trajectory, "num_joints"):
            self.num_joints = self.main_trajectory.num_joints
        else:
            # Fallback: trigger generation to find out
            pos, _, _ = self.main_trajectory.generate()
            self.num_joints = pos.shape[1]

        self.num_harmonics = cfg.num_harmonics
        self.base_freq = cfg.base_freq
        self.kinematics_func = kwargs.get("kinematics_func", None)

        self._is_optimized = False

        # Initialize Fourier coeffs (a, b). No q0 (offset) as we overlay on main trajectory.
        self.a = np.random.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))
        self.b = np.random.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))

        # Pre-calculation of main trajectory is lazy or done in generate
        self._main_cache = None

        # Setup Window Trajectory
        if cfg.window_trajectory is None:
            # Create default window config matching this trajectory
            win_cfg = WindowTrajectoryConfig(duration=self.duration, fps=self.fps, num_joints=self.num_joints)
        else:
            win_cfg = cfg.window_trajectory
            # Ensure it matches essential params
            win_cfg.duration = self.duration
            win_cfg.fps = self.fps
            win_cfg.num_joints = self.num_joints

        self.window_trajectory = WindowTrajectory(win_cfg, *args, **kwargs)

    def _apply_window_trajectory(self, q_raw: np.ndarray, dq_raw: np.ndarray, ddq_raw: np.ndarray):
        """
        Apply window trajectory to raw excitation.
        """
        # Get window values
        s, ds, dds = self.window_trajectory.get_value()

        # Apply product rule
        # q_exc = s * q
        # dq_exc = s'q + sq'
        # ddq_exc = s''q + 2s'q' + sq''

        q_exc = s * q_raw
        dq_exc = ds * q_raw + s * dq_raw
        ddq_exc = dds * q_raw + 2 * ds * dq_raw + s * ddq_raw

        return q_exc, dq_exc, ddq_exc

    def _optimize(self, max_iter=50):
        print("Starting ExcitedTrajectory Optimization...")

        # Pre-calculate main trajectory if strictly needed (though generate handles caching usually)
        if self._main_cache is None:
            self._main_cache = self.main_trajectory.generate()

        q_main, dq_main, ddq_main = self._main_cache

        # Optimization variables: a and b
        x0 = np.concatenate([self.a.flatten(), self.b.flatten()])

        # Callback logging
        it_count = 0
        current_cond = float("inf")

        def objective(x):
            nonlocal current_cond
            split = self.num_joints * self.num_harmonics
            a_flat = x[:split]
            b_flat = x[split:]

            a = a_flat.reshape(self.num_joints, self.num_harmonics)
            b = b_flat.reshape(self.num_joints, self.num_harmonics)

            # Pure Fourier (offset 0)
            coeffs = {"a": a, "b": b, "q0": np.zeros(self.num_joints)}
            f_cfg = FourierTrajectoryConfig(
                duration=self.duration,
                fps=self.fps,
                num_joints=self.num_joints,
                num_harmonics=self.num_harmonics,
                base_freq=self.base_freq,
                coefficients=coeffs,
            )
            f_traj = FourierTrajectory(f_cfg)

            # 1. Fourier Raw
            q_raw, dq_raw, ddq_raw = f_traj.get_value()

            # 2. Apply Window
            q_exc, dq_exc, ddq_exc = self._apply_window_trajectory(q_raw, dq_raw, ddq_raw)

            # 3. Superposition
            q_total = q_main + q_exc
            dq_total = dq_main + dq_exc
            ddq_total = ddq_main + ddq_exc

            # 4. Calculate Condition Number
            Y = None
            # Loop (can be slow, but robust)
            for i in range(len(self.time_array)):
                A_k = self.kinematics_func(q_total[i], dq_total[i], ddq_total[i])
                if Y is None:
                    n_params = A_k.shape[1]
                    Y = np.zeros((n_params, n_params))
                Y += A_k.T @ A_k

            eigvals = np.linalg.eigvalsh(Y)
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)

            if min_eig < 1e-9:
                cond = 1e9
            else:
                cond = max_eig / min_eig

            current_cond = cond
            return cond

        def callback(xk):
            nonlocal it_count
            it_count += 1
            print(f"Iteration {it_count}: Condition Number = {current_cond:.4f}")

        # Bounds: Keep excitation reasonable. +/- 0.5 rad seems okay for "enriched" trajectory
        bounds = [(-0.5, 0.5)] * len(x0)

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={"maxiter": max_iter, "disp": True},
        )

        x_opt = res.x
        split = self.num_joints * self.num_harmonics
        self.a = x_opt[:split].reshape(self.num_joints, self.num_harmonics)
        self.b = x_opt[split:].reshape(self.num_joints, self.num_harmonics)

        self._is_optimized = True
        print(f"Optimization Finished. Final Condition Number: {res.fun:.4f}")

    def _generate(
        self, *args, **kwargs
    ):
        """
        Generate trajectory. Runs optimization if needed.
        """
        max_iter = kwargs.get("max_iter", 50)
        
        # Always need main trajectory cache
        if self._main_cache is None:
            self._main_cache = self.main_trajectory.generate()

        q_main, dq_main, ddq_main = self._main_cache

        if not self._is_optimized:
            if self.kinematics_func:
                self._optimize(max_iter=max_iter)
            else:
                print("Warning: Generating ExcitedTrajectory without optimization (no kinematics_func).")

        # Generate final
        coeffs = {"a": self.a, "b": self.b, "q0": np.zeros(self.num_joints)}
        f_cfg = FourierTrajectoryConfig(
            duration=self.duration,
            fps=self.fps,
            num_joints=self.num_joints,
            num_harmonics=self.num_harmonics,
            base_freq=self.base_freq,
            coefficients=coeffs,
        )
        f_traj = FourierTrajectory(f_cfg)

        q_raw, dq_raw, ddq_raw = f_traj.get_value()
        q_exc, dq_exc, ddq_exc = self._apply_window_trajectory(q_raw, dq_raw, ddq_raw)

        q_total = q_main + q_exc
        dq_total = dq_main + dq_exc
        ddq_total = ddq_main + ddq_exc

        return q_total, dq_total, ddq_total
