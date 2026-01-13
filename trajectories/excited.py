from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import MISSING
from scipy.optimize import minimize

from factory import instantiate

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig
from .fourier import FourierTrajectory, FourierTrajectoryConfig
from .spline import QuinticSplineTrajectoryConfig


@dataclass
class ExcitedTrajectoryConfig(BaseTrajectoryConfig):
    # Use Any instead of Union to avoid OmegaConf limitation with Union of containers
    main_trajectory: Any = field(default_factory=lambda: MISSING)
    num_harmonics: int = MISSING
    base_freq: float = MISSING

    # MuJoCo model paths for kinematics_func construction (optional)
    manipulator: str | None = field(default_factory=lambda: MISSING)
    object: str | None = field(default_factory=lambda: MISSING)
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
        # Handle the case where main_trajectory is a dict (from YAML) instead of a config object
        main_traj_cfg = cfg.main_trajectory
        if isinstance(main_traj_cfg, dict):
            # Convert dict to appropriate config class based on target_class
            from omegaconf import OmegaConf

            target_class_name = main_traj_cfg.get("target_class", "QuinticSplineTrajectory")
            if "Spline" in target_class_name:
                main_traj_cfg = OmegaConf.to_object(OmegaConf.merge(QuinticSplineTrajectoryConfig(), main_traj_cfg))
            else:
                main_traj_cfg = OmegaConf.to_object(OmegaConf.merge(FourierTrajectoryConfig(), main_traj_cfg))

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

    def _apply_window(self, q_raw: np.ndarray, dq_raw: np.ndarray, ddq_raw: np.ndarray):
        """
        Apply a polynomial window function s(r) = 256 * r^4 * (1-r)^4 to enforce zero boundaries.
        r = t / T (normalized time)

        Args:
            q_raw, dq_raw, ddq_raw: Arrays of shape (N, dof) from pure Fourier

        Returns:
            q_exc, dq_exc, ddq_exc: Windowed excitation arrays
        """
        T = self.duration
        # Avoid division by zero
        if T == 0:
            return np.zeros_like(q_raw), np.zeros_like(dq_raw), np.zeros_like(ddq_raw)

        # r: (N,)
        r = self.time_array / T

        # Window s(r)
        # s = 256 * r^4 * (1-r)^4
        # Let u = r * (1-r) = r - r^2
        # s = 256 * u^4
        u = r * (1.0 - r)
        s = 256.0 * (u**4)

        # Derivatives with respect to time t
        # dr/dt = 1/T
        # du/dr = 1 - 2r
        # du/dt = (1 - 2r) / T

        # ds/dt = ds/du * du/dt = 4 * 256 * u^3 * (1-2r)/T
        #       = 1024 * u^3 * (1-2r) / T

        dr_dt = 1.0 / T
        du_dr = 1.0 - 2.0 * r

        ds_du = 4.0 * 256.0 * (u**3)
        ds_dt = ds_du * du_dr * dr_dt

        # d2s/dt2
        # d(ds_dt)/dt = d/dt [ 1024/T * u^3 * (1-2r) ]
        #             = 1024/T * [ (d(u^3)/dt)*(1-2r) + u^3 * d(1-2r)/dt ]
        # d(u^3)/dt = 3u^2 * du/dt
        #           = 3u^2 * du_dr * dr_dt
        # d(1-2r)/dt = -2 * dr_dt = -2/T

        # So:
        # dds_dt2 = 1024/T * [ (3u^2 * du_dr * dr_dt) * (1-2r) + u^3 * (-2/T) ]
        #         = 1024/T * [ 3u^2 * (1-2r)^2 / T - 2u^3 / T ]
        #         = 1024/T^2 * [ 3u^2*(1-2r)^2 - 2u^3 ]

        dds_dt2 = (1024.0 / (T**2)) * (3.0 * (u**2) * (du_dr**2) - 2.0 * (u**3))

        # Reshape for broadcasting along joints (N, 1) if necessary, but numpy usually handles (N,) * (N, J)
        s = s[:, np.newaxis]
        ds_dt = ds_dt[:, np.newaxis]
        dds_dt2 = dds_dt2[:, np.newaxis]

        # Apply chain rule for product
        # q_exc = s * q
        # dq_exc = s'q + sq'
        # ddq_exc = s''q + 2s'q' + sq''

        q_exc = s * q_raw
        dq_exc = ds_dt * q_raw + s * dq_raw
        ddq_exc = dds_dt2 * q_raw + 2 * ds_dt * dq_raw + s * ddq_raw

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
            q_exc, dq_exc, ddq_exc = self._apply_window(q_raw, dq_raw, ddq_raw)

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

    def generate(
        self, show_plot: bool = False, plot_path: str | None = None, json_path: str | None = None, max_iter: int = 50
    ):
        """
        Generate trajectory. Runs optimization if needed.
        """
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
        q_exc, dq_exc, ddq_exc = self._apply_window(q_raw, dq_raw, ddq_raw)

        q_total = q_main + q_exc
        dq_total = dq_main + dq_exc
        ddq_total = ddq_main + ddq_exc

        self.plot(q_total, dq_total, ddq_total, show=show_plot, plot_path=plot_path)

        if json_path:
            self.write_to_json(q_total, dq_total, ddq_total, json_path)

        return q_total, dq_total, ddq_total
