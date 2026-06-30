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

    # Per-joint position limits for the total trajectory (penalty-enforced)
    q_min: list[float] | None = None
    q_max: list[float] | None = None

    # Fourier coefficient bounds: per-joint list, or omit for default 0.5 all joints
    coeff_bounds: list[float] | None = None

    # Velocity/acceleration limits for analytical bound derivation (Stage 3)
    dq_max: list[float] | None = None
    ddq_max: list[float] | None = None

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
            config_registry = {
                "SplineTrajectory": SplineTrajectoryConfig,
                "FourierTrajectory": FourierTrajectoryConfig,
            }
            config_cls = config_registry.get(target_class_name)
            if config_cls is None:
                raise ValueError(f"Unknown main_trajectory target_class: {target_class_name}")
            main_traj_cfg = OmegaConf.to_object(OmegaConf.merge(config_cls(), main_traj_cfg))

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
        self.q_min = np.array(cfg.q_min, dtype=np.float64) if cfg.q_min is not None else None
        self.q_max = np.array(cfg.q_max, dtype=np.float64) if cfg.q_max is not None else None

        # Compute coeff_bounds: manual, analytical, or both (take tighter)
        if cfg.coeff_bounds is None:
            manual_bounds = np.full(self.num_joints, 0.5)
        else:
            manual_bounds = np.array([float(b) for b in cfg.coeff_bounds])

        dq_max = np.array(cfg.dq_max) if cfg.dq_max is not None else None
        ddq_max = np.array(cfg.ddq_max) if cfg.ddq_max is not None else None

        if dq_max is not None or ddq_max is not None:
            analytical_bounds = self.compute_fourier_bounds(
                self.num_joints, cfg.num_harmonics, cfg.base_freq, cfg.duration, dq_max, ddq_max
            )
            self.coeff_bounds = np.minimum(manual_bounds, analytical_bounds).tolist()
            print(f"Analytical bounds: {analytical_bounds}")
            print(f"Final coeff_bounds (min of manual & analytical): {self.coeff_bounds}")
        else:
            self.coeff_bounds = manual_bounds.tolist()
        self.kinematics_func = kwargs.get("kinematics_func", None)

        self._is_optimized = False

        self.a = np.random.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))
        self.b = np.random.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))

        # Setup Window Trajectory
        win_cfg = WindowTrajectoryConfig(duration=self.duration, fps=self.fps, num_joints=self.num_joints)
        self.window_trajectory = WindowTrajectory(win_cfg, *args, **kwargs)

    def _apply_window_trajectory(self, q_raw: np.ndarray, dq_raw: np.ndarray, ddq_raw: np.ndarray):
        return self.window_trajectory.apply(q_raw, dq_raw, ddq_raw)

    @staticmethod
    def compute_fourier_bounds(
        num_joints: int,
        num_harmonics: int,
        base_freq: float,
        duration: float,
        dq_max: np.ndarray | None = None,
        ddq_max: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute per-coefficient box bounds from velocity/acceleration limits.

        Uses the triangle inequality on the windowed (256s^4(1-s)^4) Fourier trajectory.
        Returns an array of shape (num_joints,) with the tightest bound per joint.
        """
        if dq_max is None and ddq_max is None:
            raise ValueError("At least one of dq_max or ddq_max must be provided")

        s = np.linspace(0, 1, 10_000)
        harmonics = np.arange(1, num_harmonics + 1, dtype=np.float64)
        omega = 2.0 * np.pi * base_freq * harmonics

        # w(s) = 256 s^4 (1-s)^4
        # w'(s) = 1024 s^3 (1-s)^3 (1-2s)
        # w''(s) = 1024 [3 s^2 (1-s)^2 (1-2s)^2 - 2 s^3 (1-s)^3]
        dw_ds = 1024.0 * s**3 * (1.0 - s) ** 3 * (1.0 - 2.0 * s)
        d2w_ds2 = 1024.0 * (3.0 * s**2 * (1.0 - s) ** 2 * (1.0 - 2.0 * s) ** 2 - 2.0 * s**3 * (1.0 - s) ** 3)

        upper = np.full(num_joints, np.inf, dtype=np.float64)

        if dq_max is not None:
            dw_dt_max = float(np.max(np.abs(dw_ds))) / duration
            alpha_vel = dw_dt_max + omega
            for j in range(num_joints):
                for k in range(num_harmonics):
                    bound = float(dq_max[j]) / (2.0 * num_harmonics * alpha_vel[k])
                    upper[j] = min(upper[j], bound)

        if ddq_max is not None:
            dw_dt_max = float(np.max(np.abs(dw_ds))) / duration
            d2w_dt2_max = float(np.max(np.abs(d2w_ds2))) / (duration**2)
            alpha_acc = d2w_dt2_max + 2.0 * dw_dt_max * omega + omega**2
            for j in range(num_joints):
                for k in range(num_harmonics):
                    bound = float(ddq_max[j]) / (2.0 * num_harmonics * alpha_acc[k])
                    upper[j] = min(upper[j], bound)

        return upper

    def _build_trajectory(self, x, q_main, dq_main, ddq_main, _f_traj):
        split = self.num_joints * self.num_harmonics
        _f_traj.a = x[:split].reshape(self.num_joints, self.num_harmonics)
        _f_traj.b = x[split:].reshape(self.num_joints, self.num_harmonics)

        q_raw, dq_raw, ddq_raw = _f_traj.get_value()
        q_exc, dq_exc, ddq_exc = self._apply_window_trajectory(q_raw, dq_raw, ddq_raw)

        return q_main + q_exc, dq_main + dq_exc, ddq_main + ddq_exc

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

        _cache_key = None
        _cache_val = None

        def _get_trajectory(x):
            nonlocal _cache_key, _cache_val
            key = x.tobytes()
            if key != _cache_key:
                _cache_key = key
                _cache_val = self._build_trajectory(x, q_main, dq_main, ddq_main, _f_traj)
            return _cache_val

        penalty_weight = 1e5
        use_q_limits = self.q_min is not None or self.q_max is not None

        def objective(x):
            nonlocal current_cond
            q_total, dq_total, ddq_total = _get_trajectory(x)
            cond = BaseTrajectory.compute_condition_number(
                self.time_array, self.kinematics_func, q_total, dq_total, ddq_total
            )
            current_cond = cond

            if use_q_limits:
                if self.q_min is not None:
                    lo_viol = np.maximum(0.0, self.q_min - q_total)
                    cond += penalty_weight * np.sum(lo_viol**2)
                if self.q_max is not None:
                    hi_viol = np.maximum(0.0, q_total - self.q_max)
                    cond += penalty_weight * np.sum(hi_viol**2)

            return cond

        if use_q_limits:
            print(f"Joint position limits (penalty, weight={penalty_weight}):")
            if self.q_min is not None:
                print(f"  q_min = {self.q_min}")
            if self.q_max is not None:
                print(f"  q_max = {self.q_max}")

        start_time = time.time()

        def callback(xk):
            nonlocal it_count
            it_count += 1
            elapsed = time.time() - start_time
            eta = elapsed / it_count * (max_iter - it_count) if it_count < max_iter else 0
            print(
                f"Iteration {it_count}/{max_iter}: Cond = {current_cond:.4f} | "
                f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining"
            )

        bounds = []
        for j in range(self.num_joints):
            B = self.coeff_bounds[j]
            bounds.extend([(-B, B)] * self.num_harmonics)
        bounds = bounds * 2  # a coefficients + b coefficients

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
        print(f"Optimization Finished. Final Condition Number: {current_cond:.4f}")

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
