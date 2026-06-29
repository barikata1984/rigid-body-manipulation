import time
from dataclasses import dataclass, field

import numpy as np
from omegaconf import MISSING, OmegaConf
from scipy.optimize import minimize

from factory import instantiate

from .base_trajectory import BaseTrajectory
from .fourier import FourierTrajectory, FourierTrajectoryConfig
from .spline import SplineTrajectoryConfig
from .window import WindowTrajectory, WindowTrajectoryConfig
from .windowed_fourier import WindowedFourierTrajectoryConfig


@dataclass
class ExcitedTrajectoryConfig(WindowedFourierTrajectoryConfig):
    # Use Any instead of Union to avoid OmegaConf limitation with Union of containers

    # Main trajectory config (e.g., spline) to be enriched with excitation
    main_trajectory: dict | None = field(default_factory=lambda: MISSING)

    # MuJoCo model paths for kinematics_func construction (optional)
    manipulator: str | None = field(default_factory=lambda: MISSING)
    object: str | None = field(default_factory=lambda: MISSING)
    ee_body_name: str = "link6"  # End-effector body name
    max_iter: int = 50

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
        if main_traj_cfg is None:
            raise ValueError("ExcitedTrajectoryConfig.main_trajectory must not be None")
        if isinstance(main_traj_cfg, dict):
            target_class_name = main_traj_cfg.get("target_class", "SplineTrajectory")
            if "Spline" in target_class_name:
                main_traj_cfg = OmegaConf.to_object(OmegaConf.merge(SplineTrajectoryConfig(), main_traj_cfg))
            else:
                main_traj_cfg = OmegaConf.to_object(OmegaConf.merge(FourierTrajectoryConfig(), main_traj_cfg))

        self.main_trajectory = instantiate(main_traj_cfg, *args, **kwargs)

        self._main_cache = None
        if hasattr(self.main_trajectory, "num_joints"):
            self.num_joints = self.main_trajectory.num_joints
        else:
            self._main_cache = self.main_trajectory.generate()
            self.num_joints = self._main_cache[0].shape[1]

        self.num_harmonics = cfg.num_harmonics
        self.base_freq = cfg.base_freq
        self.max_iter = cfg.max_iter
        self.kinematics_func = kwargs.get("kinematics_func", None)

        self._is_optimized = False

        self.a = np.random.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))
        self.b = np.random.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))

        # Setup Window Trajectory
        win_cfg = WindowTrajectoryConfig(duration=self.duration, fps=self.fps, num_joints=self.num_joints)
        self.window_trajectory = WindowTrajectory(win_cfg, *args, **kwargs)

    def _apply_window_trajectory(self, q_raw: np.ndarray, dq_raw: np.ndarray, ddq_raw: np.ndarray):
        return self.window_trajectory.apply(q_raw, dq_raw, ddq_raw)

    def _optimize(self):
        max_iter = self.max_iter
        print("Starting ExcitedTrajectory Optimization...")

        if self._main_cache is None:
            self._main_cache = self.main_trajectory.generate()

        q_main, dq_main, ddq_main = self._main_cache

        x0 = np.concatenate([self.a.flatten(), self.b.flatten()])

        it_count = 0
        current_cond = float("inf")

        _f_cfg = FourierTrajectoryConfig(
            duration=self.duration,
            fps=self.fps,
            num_joints=self.num_joints,
            num_harmonics=self.num_harmonics,
            base_freq=self.base_freq,
            coefficients={
                "a": np.zeros((self.num_joints, self.num_harmonics)),
                "b": np.zeros((self.num_joints, self.num_harmonics)),
                "q0": np.zeros(self.num_joints),
            },
        )
        _f_traj = FourierTrajectory(_f_cfg)

        def objective(x):
            nonlocal current_cond
            split = self.num_joints * self.num_harmonics
            a_flat = x[:split]
            b_flat = x[split:]

            a = a_flat.reshape(self.num_joints, self.num_harmonics)
            b = b_flat.reshape(self.num_joints, self.num_harmonics)

            _f_traj.a = a
            _f_traj.b = b

            q_raw, dq_raw, ddq_raw = _f_traj.get_value()
            q_exc, dq_exc, ddq_exc = self._apply_window_trajectory(q_raw, dq_raw, ddq_raw)

            q_total = q_main + q_exc
            dq_total = dq_main + dq_exc
            ddq_total = ddq_main + ddq_exc

            cond = BaseTrajectory.compute_condition_number(
                self.time_array, self.kinematics_func, q_total, dq_total, ddq_total
            )
            current_cond = cond
            return cond

        start_time = time.time()

        def callback(xk):
            nonlocal it_count
            it_count += 1
            elapsed = time.time() - start_time
            eta = elapsed / it_count * (max_iter - it_count) if it_count < max_iter else 0
            print(
                f"Iteration {it_count}/{max_iter}: Cond = {current_cond:.4f} | {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining"
            )

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

    def _generate(self, *args, **kwargs):
        """
        Generate trajectory. Runs optimization if needed.
        """
        # Always need main trajectory cache
        if self._main_cache is None:
            self._main_cache = self.main_trajectory.generate()

        q_main, dq_main, ddq_main = self._main_cache

        if not self._is_optimized:
            if self.kinematics_func:
                self._optimize()
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
