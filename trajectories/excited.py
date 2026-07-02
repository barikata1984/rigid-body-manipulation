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

    # Per-joint singularity exclusion: require |q_j(t) - center_j| >= margin_j.
    # None (or margin == 0) disables the constraint for that joint. Unlike q_min/q_max
    # (an inclusion range), this excludes a neighborhood around center_j.
    singularity_center: list[float] | None = None
    singularity_margin: list[float] | None = None

    # Fourier coefficient bounds: per-joint list, or omit for default 0.5 all joints
    coeff_bounds: list[float] | None = None

    # Velocity/acceleration limits for analytical bound derivation (Stage 3)
    dq_max: list[float] | None = None
    ddq_max: list[float] | None = None

    # Objective function: "condition_number" or "d_optimal"
    objective_type: str = "condition_number"
    # Optimizer method: "L-BFGS-B" or "SLSQP"
    optimizer_method: str = "L-BFGS-B"

    # Multi-start optimization (Stage 5)
    n_restarts: int = 1
    seed: int = 42
    early_stop_patience: int = 5

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

        if cfg.objective_type not in ("condition_number", "d_optimal"):
            raise ValueError(
                f"Invalid objective_type: {cfg.objective_type!r}. Expected 'condition_number' or 'd_optimal'."
            )
        if cfg.optimizer_method not in ("L-BFGS-B", "SLSQP"):
            raise ValueError(f"Invalid optimizer_method: {cfg.optimizer_method!r}. Expected 'L-BFGS-B' or 'SLSQP'.")

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

        for _name in (
            "coeff_bounds",
            "q_min",
            "q_max",
            "dq_max",
            "ddq_max",
            "singularity_center",
            "singularity_margin",
        ):
            _val = getattr(cfg, _name)
            if _val is not None and len(_val) != self.num_joints:
                raise ValueError(f"{_name} has {len(_val)} elements but expected {self.num_joints} (num_joints).")

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
            for j in range(self.num_joints):
                if analytical_bounds[j] < 0.5 * manual_bounds[j]:
                    print(
                        f"Warning: analytical bound for joint {j} ({analytical_bounds[j]:.4f}) "
                        f"is much tighter than manual coeff_bounds ({manual_bounds[j]:.4f}); "
                        "analytical bound will be used."
                    )
            print(f"Final coeff_bounds (min of manual & analytical): {self.coeff_bounds}")
        else:
            self.coeff_bounds = manual_bounds.tolist()
        self.kinematics_func = kwargs.get("kinematics_func", None)
        self.objective_type = cfg.objective_type
        self.optimizer_method = cfg.optimizer_method
        self.n_restarts = cfg.n_restarts
        self.seed = cfg.seed
        self.early_stop_patience = cfg.early_stop_patience

        # Singularity exclusion: |q_j - center_j| >= margin_j for active joints (margin > 0).
        if cfg.singularity_margin is not None:
            center_src = cfg.singularity_center if cfg.singularity_center is not None else [0.0] * self.num_joints
            self.singularity_center = np.array([0.0 if c is None else float(c) for c in center_src], dtype=np.float64)
            self.singularity_margin = np.array(
                [0.0 if m is None else float(m) for m in cfg.singularity_margin], dtype=np.float64
            )
            self.singularity_active = bool(np.any(self.singularity_margin > 0.0))
        else:
            self.singularity_center = None
            self.singularity_margin = None
            self.singularity_active = False

        self._is_optimized = False

        init_rng = np.random.default_rng(cfg.seed)
        self.a = init_rng.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))
        self.b = init_rng.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))

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

    def _generate_random_x0(self, rng: np.random.Generator) -> np.ndarray:
        nj = self.num_joints
        nh = self.num_harmonics
        x = np.zeros(2 * nj * nh, dtype=np.float64)
        for k in range(nh):
            scale = 0.3 / (k + 1)
            for j in range(nj):
                bound = self.coeff_bounds[j]
                s = min(scale, bound)
                x[j * nh + k] = rng.uniform(-s, s)
                x[nj * nh + j * nh + k] = rng.uniform(-s, s)
        return x

    def _build_q_constraints(self, get_trajectory_fn):
        """Build SLSQP inequality constraints for joint position limits.

        Reuses the caller's cached trajectory function (get_trajectory_fn(x) -> (q, dq, ddq))
        to avoid rebuilding a separate FourierTrajectory and recomputing _build_trajectory.
        """
        constraints = []

        def _get_q_total(x):
            return get_trajectory_fn(x)[0]

        if self.q_min is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: np.min(_get_q_total(x) - self.q_min),
                }
            )
        if self.q_max is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: np.min(self.q_max - _get_q_total(x)),
                }
            )

        return constraints

    def _run_single_optimization(self, x0, q_main, dq_main, ddq_main, bounds, restart_label=""):
        max_iter = self.max_iter
        use_q_limits = self.q_min is not None or self.q_max is not None
        use_d_optimal = self.objective_type == "d_optimal"
        use_slsqp = self.optimizer_method == "SLSQP"
        penalty_weight = 1e5

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

        current_cond = float("inf")
        current_obj = float("inf")
        it_count = 0

        def objective(x):
            nonlocal current_cond, current_obj
            q_total, dq_total, ddq_total = _get_trajectory(x)
            obj_val, cond = BaseTrajectory.compute_objective_with_cond(
                self.time_array,
                self.kinematics_func,
                q_total,
                dq_total,
                ddq_total,
                objective_type=self.objective_type,
            )
            current_cond = cond
            current_obj = obj_val

            # L-BFGS-B: use penalty for q_limits (no constraint support)
            if not use_slsqp and use_q_limits:
                if self.q_min is not None:
                    lo_viol = np.maximum(0.0, self.q_min - q_total)
                    obj_val += penalty_weight * np.sum(lo_viol**2)
                if self.q_max is not None:
                    hi_viol = np.maximum(0.0, q_total - self.q_max)
                    obj_val += penalty_weight * np.sum(hi_viol**2)

            # Singularity exclusion is non-convex (|q - center| >= margin is a disjunction),
            # so it cannot be a single smooth SLSQP inequality; enforce it as a penalty for
            # both optimizers.
            if self.singularity_active:
                dev = np.abs(q_total - self.singularity_center)
                sing_viol = np.maximum(0.0, self.singularity_margin - dev)
                obj_val += penalty_weight * np.sum(sing_viol**2)

            return obj_val

        start_time = time.time()

        def callback(xk):
            nonlocal it_count
            it_count += 1
            elapsed = time.time() - start_time
            eta = elapsed / it_count * (max_iter - it_count) if it_count < max_iter else 0
            if use_d_optimal:
                print(
                    f"  {restart_label}iter {it_count}/{max_iter}: D-opt = {current_obj:.4f}, "
                    f"Cond = {current_cond:.4f} | {elapsed:.1f}s, ~{eta:.0f}s left"
                )
            else:
                print(
                    f"  {restart_label}iter {it_count}/{max_iter}: "
                    f"Cond = {current_cond:.4f} | {elapsed:.1f}s, ~{eta:.0f}s left"
                )

        minimize_kwargs = {
            "fun": objective,
            "x0": x0,
            "method": self.optimizer_method,
            "bounds": bounds,
            "callback": callback,
            "options": {"maxiter": max_iter},
        }

        # SLSQP: use proper inequality constraints instead of penalty
        if use_slsqp and use_q_limits:
            minimize_kwargs["constraints"] = self._build_q_constraints(_get_trajectory)

        res = minimize(**minimize_kwargs)

        wall_time = time.time() - start_time
        return res.x, current_cond, current_obj, wall_time

    def _optimize(self):
        print("Starting ExcitedTrajectory Optimization...")
        print(f"Objective: {self.objective_type}, method: {self.optimizer_method}, restarts: {self.n_restarts}")

        if self._main_cache is None:
            self._main_cache = self.main_trajectory.generate()

        q_main, dq_main, ddq_main = self._main_cache

        use_q_limits = self.q_min is not None or self.q_max is not None
        use_d_optimal = self.objective_type == "d_optimal"

        if use_q_limits:
            print("Joint position limits (penalty):")
            if self.q_min is not None:
                print(f"  q_min = {self.q_min}")
            if self.q_max is not None:
                print(f"  q_max = {self.q_max}")

        bounds = []
        for j in range(self.num_joints):
            B = self.coeff_bounds[j]
            bounds.extend([(-B, B)] * self.num_harmonics)
        bounds = bounds * 2

        rng = np.random.default_rng(self.seed)

        best_x = None
        best_cond = float("inf")
        best_idx = 0
        no_improve_count = 0
        t0 = time.time()

        for i in range(self.n_restarts):
            if i == 0:
                x0 = np.concatenate([self.a.flatten(), self.b.flatten()])
            else:
                x0 = self._generate_random_x0(rng)

            label = f"[{i + 1}/{self.n_restarts}] " if self.n_restarts > 1 else ""
            x_opt, cond, fun, wall = self._run_single_optimization(
                x0, q_main, dq_main, ddq_main, bounds, restart_label=label
            )

            # Always accept the first restart (even if cond is NaN) so best_x is never None;
            # afterwards, only update on a non-NaN improvement (or to replace a NaN best_cond).
            improved = best_x is None or (not np.isnan(cond) and (np.isnan(best_cond) or cond < best_cond))
            marker = " *" if improved else ""
            if use_d_optimal:
                print(
                    f"  restart {i + 1}/{self.n_restarts}: Cond = {cond:.4f}, D-opt = {fun:.4f} ({wall:.1f}s){marker}"
                )
            else:
                print(f"  restart {i + 1}/{self.n_restarts}: Cond = {cond:.4f} ({wall:.1f}s){marker}")

            if improved:
                best_cond = cond
                best_x = x_opt.copy()
                best_idx = i
                no_improve_count = 0
            else:
                no_improve_count += 1

            if self.n_restarts > 1 and no_improve_count >= self.early_stop_patience:
                print(
                    f"  Early stop: no improvement for {self.early_stop_patience} restarts "
                    f"(best Cond = {best_cond:.4f})"
                )
                break

        split = self.num_joints * self.num_harmonics
        self.a = best_x[:split].reshape(self.num_joints, self.num_harmonics)
        self.b = best_x[split:].reshape(self.num_joints, self.num_harmonics)

        self._is_optimized = True
        total_time = time.time() - t0
        print(f"Optimization Finished. Best Cond = {best_cond:.4f} (restart {best_idx + 1}, {total_time:.1f}s total)")

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
