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

    # Per-joint physical scale used to nondimensionalize the Fourier coefficients
    # seen by the optimizer. This is a numerical preconditioner, not a bound:
    # physical_coeff[j, k] = decision_variable_scale[j] * normalized_coeff[j, k].
    # None preserves the legacy raw-coordinate optimization (unit scale).
    decision_variable_scale: list[float] | None = None

    # Velocity/acceleration limits for analytical bound derivation (Stage 3)
    dq_max: list[float] | None = None
    ddq_max: list[float] | None = None

    # If True (default), tighten the Fourier coefficient box bounds with the
    # analytical triangle-inequality bounds from dq_max/ddq_max. If False, keep
    # the manual coeff_bounds as box bounds and instead enforce dq_max/ddq_max as
    # direct SLSQP inequality constraints on the sampled trajectory (requires
    # optimizer_method="SLSQP"); this avoids the worst-case conservatism of the
    # analytical bounds.
    use_analytical_bounds: bool = True

    # Column-equilibrate the observation matrix before computing the condition
    # number / D-optimal objective. Normalizes each regressor column to unit L2
    # norm so the objective is invariant to per-column unit choices; without it
    # the condition number is not a well-posed design criterion.
    column_scale: bool = True

    # Objective function: "condition_number" or "d_optimal"
    objective_type: str = "condition_number"
    # Optimizer method: "L-BFGS-B" or "SLSQP"
    optimizer_method: str = "L-BFGS-B"
    # Optional relative step for SciPy's two-point finite differences. None
    # preserves the optimizer default; a positive value enables jac="2-point".
    finite_diff_rel_step: float | None = None

    # Optional end-effector linear-speed limit. The corresponding callable is
    # supplied as ``ee_speed_func(q, dq) -> (n_samples,)`` when constructing the
    # trajectory because its implementation depends on the manipulator model.
    ee_speed_max: float | None = None

    # Reduce sampled kinematic constraints to the currently tightest sample for
    # each joint and constraint side. The fixed-size working set is refreshed at
    # accepted SLSQP iterates, avoiding a constraint vector whose dimension
    # changes during optimization.
    active_constraint_working_set: bool = False
    active_constraint_update_interval: int = 10

    # Multi-start optimization (Stage 5)
    n_restarts: int = 1
    seed: int = 42
    early_stop_patience: int = 5

    # Stop the optimizer as soon as the condition number reaches this target.
    # None (default) keeps the legacy behavior (run until max_iter / convergence).
    target_condition_number: float | None = None

    target_class: str = "ExcitedTrajectory"


class _TargetReached(Exception):
    """Signals that the target condition number was reached inside a scipy callback.

    scipy's ``minimize`` callback cannot request a stop via its return value, so we
    raise this from the callback and catch it around the ``minimize`` call, carrying
    the accepted iterate ``x`` and its condition number.
    """

    def __init__(self, x: np.ndarray, cond: float):
        self.x = x
        self.cond = cond
        super().__init__(f"Target condition number reached: {cond:.4f}")


class ExcitedTrajectory(BaseTrajectory):
    def __init__(self, cfg: ExcitedTrajectoryConfig, *args, **kwargs):  # noqa: C901
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
            "decision_variable_scale",
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

        self.dq_max = np.array(cfg.dq_max, dtype=np.float64) if cfg.dq_max is not None else None
        self.ddq_max = np.array(cfg.ddq_max, dtype=np.float64) if cfg.ddq_max is not None else None
        self.use_analytical_bounds = cfg.use_analytical_bounds

        # coeff_bounds and the analytical (dq_max/ddq_max-derived) bound are two
        # independent upper bounds on the same variable only if both are physically
        # grounded. When the analytical bound is active it is the sole grounded
        # constraint, so a manual coeff_bounds cannot also be set here (it would
        # either be redundant or silently override the physical bound with an
        # arbitrary number).
        if (self.dq_max is not None or self.ddq_max is not None) and self.use_analytical_bounds:
            if cfg.coeff_bounds is not None:
                raise ValueError(
                    "coeff_bounds cannot be combined with dq_max/ddq_max + use_analytical_bounds=True: "
                    "the analytical triangle-inequality bound is the sole grounded constraint in this "
                    "mode. Either drop coeff_bounds, or set use_analytical_bounds=False to use "
                    "coeff_bounds as the box bound with dq_max/ddq_max enforced as direct constraints."
                )
            analytical_bounds = self.compute_fourier_bounds(
                self.num_joints, cfg.num_harmonics, cfg.base_freq, cfg.duration, self.dq_max, self.ddq_max
            )
            self.coeff_bounds = analytical_bounds.tolist()
            print(f"Analytical coeff_bounds: {self.coeff_bounds}")
        elif cfg.coeff_bounds is None:
            self.coeff_bounds = np.full(self.num_joints, 0.5).tolist()
        else:
            self.coeff_bounds = [float(b) for b in cfg.coeff_bounds]

        coeff_bounds_array = np.asarray(self.coeff_bounds, dtype=np.float64)
        if np.any(np.isnan(coeff_bounds_array)) or np.any(coeff_bounds_array <= 0.0):
            raise ValueError("coeff_bounds values must be positive or +inf")

        if cfg.decision_variable_scale is None:
            self.decision_variable_scale = np.ones(self.num_joints, dtype=np.float64)
        else:
            self.decision_variable_scale = np.asarray(cfg.decision_variable_scale, dtype=np.float64)
            if not np.all(np.isfinite(self.decision_variable_scale)) or np.any(self.decision_variable_scale <= 0.0):
                raise ValueError("decision_variable_scale values must be finite and strictly positive")

        if self.q_min is not None:
            if np.any(np.isnan(self.q_min)) or np.any(np.isposinf(self.q_min)):
                raise ValueError("q_min values must be finite or -inf")
        if self.q_max is not None:
            if np.any(np.isnan(self.q_max)) or np.any(np.isneginf(self.q_max)):
                raise ValueError("q_max values must be finite or +inf")
        if self.q_min is not None and self.q_max is not None:
            finite_pair = np.isfinite(self.q_min) & np.isfinite(self.q_max)
            if np.any(self.q_min[finite_pair] >= self.q_max[finite_pair]):
                raise ValueError("q_min must be strictly less than q_max for every finite bound pair")
        for _name, _limit in (("dq_max", self.dq_max), ("ddq_max", self.ddq_max)):
            if _limit is not None:
                if np.any(np.isnan(_limit)) or np.any(np.isneginf(_limit)):
                    raise ValueError(f"{_name} values must be positive or +inf")
                if np.any(_limit[np.isfinite(_limit)] <= 0.0):
                    raise ValueError(f"{_name} finite values must be strictly positive")
        self.kinematics_func = kwargs.get("kinematics_func", None)
        self.column_scale = cfg.column_scale
        self.objective_type = cfg.objective_type
        self.optimizer_method = cfg.optimizer_method
        self.finite_diff_rel_step = cfg.finite_diff_rel_step
        if self.finite_diff_rel_step is not None:
            if not np.isfinite(self.finite_diff_rel_step) or self.finite_diff_rel_step <= 0.0:
                raise ValueError("finite_diff_rel_step must be finite and strictly positive")
        self.ee_speed_max = cfg.ee_speed_max
        if self.ee_speed_max is not None and (not np.isfinite(self.ee_speed_max) or self.ee_speed_max <= 0.0):
            raise ValueError("ee_speed_max must be finite and strictly positive")
        self.ee_speed_func = kwargs.get("ee_speed_func")
        if self.ee_speed_max is not None and not callable(self.ee_speed_func):
            raise ValueError("ee_speed_func must be callable when ee_speed_max is configured")
        self.active_constraint_working_set = cfg.active_constraint_working_set
        self.active_constraint_update_interval = cfg.active_constraint_update_interval
        if self.active_constraint_update_interval <= 0:
            raise ValueError("active_constraint_update_interval must be positive")
        self.n_restarts = cfg.n_restarts
        self.seed = cfg.seed
        self.early_stop_patience = cfg.early_stop_patience
        self.target_condition_number = cfg.target_condition_number
        self.restart_callback = kwargs.get("restart_callback")
        if self.restart_callback is not None and not callable(self.restart_callback):
            raise ValueError("restart_callback must be callable or None")
        self.restart_state = kwargs.get("restart_state")
        if self.restart_state is not None and not isinstance(self.restart_state, dict):
            raise ValueError("restart_state must be a dict or None")

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
        # Final condition number from the last optimization; None until optimized.
        self.final_condition_number: float | None = None
        # Per-restart solver diagnostics.  Kept separate from trajectory metadata so
        # callers can inspect failed restarts without treating them as successes.
        self.optimization_results: list[dict] = []
        self.best_restart_index: int | None = None
        self.best_solver_success: bool | None = None

        init_rng = np.random.default_rng(cfg.seed)
        initial_a_normalized = init_rng.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))
        initial_b_normalized = init_rng.uniform(-0.01, 0.01, (self.num_joints, self.num_harmonics))
        self.a = initial_a_normalized * self.decision_variable_scale[:, None]
        self.b = initial_b_normalized * self.decision_variable_scale[:, None]

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

    def _decision_scale_vector(self) -> np.ndarray:
        """Return scales matching flattened ``a`` followed by flattened ``b``."""
        one_block = np.repeat(self.decision_variable_scale, self.num_harmonics)
        return np.concatenate([one_block, one_block])

    def _to_physical_coefficients(self, normalized_x: np.ndarray) -> np.ndarray:
        """Map dimensionless optimizer coordinates to physical Fourier coefficients."""
        return np.asarray(normalized_x, dtype=np.float64) * self._decision_scale_vector()

    def _q_residual_scale(self) -> np.ndarray:
        """Return per-joint scales for dimensionless position-limit residuals."""
        scale = self.decision_variable_scale.copy()
        if self.q_min is not None and self.q_max is not None:
            finite_pair = np.isfinite(self.q_min) & np.isfinite(self.q_max)
            scale[finite_pair] = 0.5 * (self.q_max[finite_pair] - self.q_min[finite_pair])
        return scale

    def _generate_random_x0(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a dimensionless multistart point within normalized bounds."""
        nj = self.num_joints
        nh = self.num_harmonics
        x = np.zeros(2 * nj * nh, dtype=np.float64)
        for k in range(nh):
            scale = 0.3 / (k + 1)
            for j in range(nj):
                bound = self.coeff_bounds[j] / self.decision_variable_scale[j]
                s = min(scale, bound)
                x[j * nh + k] = rng.uniform(-s, s)
                x[nj * nh + j * nh + k] = rng.uniform(-s, s)
        return x

    def _active_constraint_indices(self, trajectory):
        """Return the tightest sampled point for every constraint side."""
        q, dq, ddq = trajectory
        indices: dict[str, np.ndarray] = {}
        if self.q_min is not None:
            mask = np.isfinite(self.q_min)
            indices["q_min"] = np.argmin(q[:, mask] - self.q_min[mask], axis=0)
        if self.q_max is not None:
            mask = np.isfinite(self.q_max)
            indices["q_max"] = np.argmin(self.q_max[mask] - q[:, mask], axis=0)
        if not self.use_analytical_bounds:
            if self.dq_max is not None:
                mask = np.isfinite(self.dq_max)
                indices["dq_pos"] = np.argmax(dq[:, mask], axis=0)
                indices["dq_neg"] = np.argmin(dq[:, mask], axis=0)
            if self.ddq_max is not None:
                mask = np.isfinite(self.ddq_max)
                indices["ddq_pos"] = np.argmax(ddq[:, mask], axis=0)
                indices["ddq_neg"] = np.argmin(ddq[:, mask], axis=0)
        if self.ee_speed_max is not None:
            speed = np.asarray(self.ee_speed_func(q, dq), dtype=np.float64)
            if speed.shape != (q.shape[0],):
                raise ValueError(f"ee_speed_func must return shape ({q.shape[0]},), got {speed.shape}")
            indices["ee_speed"] = np.array([int(np.argmax(speed))], dtype=np.int64)
        return indices

    @staticmethod
    def _select_active_residual(residual: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Select one time index per residual column from a sampled matrix."""
        if residual.ndim == 1:
            return residual[indices]
        return residual[indices, np.arange(residual.shape[1])]

    def _build_q_constraints(self, get_trajectory_fn, active_indices=None):
        """Build SLSQP inequality constraints for joint kinematic limits.

        Always adds position limits (q_min/q_max). When use_analytical_bounds is
        False, also adds symmetric velocity/acceleration limits (|dq| <= dq_max,
        |ddq| <= ddq_max) as direct constraints on the sampled trajectory.

        All residuals are dimensionless and returned per sample and constrained
        joint. This prevents meters, radians, m/s, and rad/s from entering the
        SLSQP constraint Jacobian at unrelated numerical scales.

        Reuses the caller's cached trajectory function (get_trajectory_fn(x) -> (q, dq, ddq))
        to avoid rebuilding a separate FourierTrajectory and recomputing _build_trajectory.
        """
        constraints = []

        def sampled(residual, name):
            residual = np.asarray(residual)
            if active_indices is None:
                return residual.ravel()
            return self._select_active_residual(residual, active_indices[name]).ravel()

        q_scale = self._q_residual_scale()

        if self.q_min is not None and np.any(np.isfinite(self.q_min)):
            mask = np.isfinite(self.q_min)
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, mask=mask, scale=q_scale: sampled(
                        (get_trajectory_fn(x)[0][:, mask] - self.q_min[mask]) / scale[mask], "q_min"
                    ),
                }
            )
        if self.q_max is not None and np.any(np.isfinite(self.q_max)):
            mask = np.isfinite(self.q_max)
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, mask=mask, scale=q_scale: sampled(
                        (self.q_max[mask] - get_trajectory_fn(x)[0][:, mask]) / scale[mask], "q_max"
                    ),
                }
            )

        if not self.use_analytical_bounds:
            if self.dq_max is not None and np.any(np.isfinite(self.dq_max)):
                mask = np.isfinite(self.dq_max)
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, mask=mask: sampled(
                            (self.dq_max[mask] - get_trajectory_fn(x)[1][:, mask]) / self.dq_max[mask], "dq_pos"
                        ),
                    }
                )
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, mask=mask: sampled(
                            (self.dq_max[mask] + get_trajectory_fn(x)[1][:, mask]) / self.dq_max[mask], "dq_neg"
                        ),
                    }
                )
            if self.ddq_max is not None and np.any(np.isfinite(self.ddq_max)):
                mask = np.isfinite(self.ddq_max)
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, mask=mask: sampled(
                            (self.ddq_max[mask] - get_trajectory_fn(x)[2][:, mask]) / self.ddq_max[mask], "ddq_pos"
                        ),
                    }
                )
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, mask=mask: sampled(
                            (self.ddq_max[mask] + get_trajectory_fn(x)[2][:, mask]) / self.ddq_max[mask], "ddq_neg"
                        ),
                    }
                )

        if self.ee_speed_max is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: sampled(
                        1.0
                        - np.asarray(self.ee_speed_func(get_trajectory_fn(x)[0], get_trajectory_fn(x)[1]))
                        / self.ee_speed_max,
                        "ee_speed",
                    ),
                }
            )

        return constraints

    def _run_single_optimization(self, x0, q_main, dq_main, ddq_main, bounds, restart_label=""):  # noqa: C901
        max_iter = self.max_iter
        use_q_limits = self.q_min is not None or self.q_max is not None
        use_direct_kin = (not self.use_analytical_bounds) and (self.dq_max is not None or self.ddq_max is not None)
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
        decision_scale_vector = self._decision_scale_vector()

        def _get_trajectory(x):
            nonlocal _cache_key, _cache_val
            key = x.tobytes()
            if key != _cache_key:
                _cache_key = key
                physical_x = x * decision_scale_vector
                _cache_val = self._build_trajectory(physical_x, q_main, dq_main, ddq_main, _f_traj)
            return _cache_val

        current_cond = float("inf")
        current_obj = float("inf")
        it_count = 0
        feasible_tolerance = 1e-6
        best_feasible_x = None
        best_feasible_obj = float("inf")
        best_feasible_cond = float("inf")

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
                column_scale=self.column_scale,
            )
            current_cond = cond
            current_obj = obj_val

            # L-BFGS-B: use penalty for q_limits (no constraint support)
            if not use_slsqp and use_q_limits:
                q_scale = self._q_residual_scale()
                if self.q_min is not None:
                    lo_viol = np.maximum(0.0, self.q_min - q_total) / q_scale
                    obj_val += penalty_weight * np.sum(lo_viol**2)
                if self.q_max is not None:
                    hi_viol = np.maximum(0.0, q_total - self.q_max) / q_scale
                    obj_val += penalty_weight * np.sum(hi_viol**2)

            # Singularity exclusion is non-convex (|q - center| >= margin is a disjunction),
            # so it cannot be a single smooth SLSQP inequality; enforce it as a penalty for
            # both optimizers.
            if self.singularity_active:
                dev = np.abs(q_total - self.singularity_center)
                sing_viol = np.maximum(0.0, self.singularity_margin - dev) / self.decision_variable_scale
                obj_val += penalty_weight * np.sum(sing_viol**2)

            return obj_val

        start_time = time.time()
        target_cond = self.target_condition_number

        def retain_feasible(x, obj, cond):
            nonlocal best_feasible_x, best_feasible_obj, best_feasible_cond
            trajectory = _get_trajectory(x)
            violation = self._constraint_violation_ratio(trajectory)
            if violation <= feasible_tolerance and obj < best_feasible_obj:
                best_feasible_x = np.array(x, dtype=np.float64, copy=True)
                best_feasible_obj = float(obj)
                best_feasible_cond = float(cond)
            return violation

        initial_obj = objective(x0)
        retain_feasible(x0, initial_obj, current_cond)

        def callback(xk):
            nonlocal it_count
            it_count += 1
            if (
                active_indices is not None
                and it_count % self.active_constraint_update_interval == 0
            ):
                active_indices.update(self._active_constraint_indices(_get_trajectory(xk)))
            obj_xk = objective(xk)
            cond_xk = current_cond
            retain_feasible(xk, obj_xk, cond_xk)
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

            if target_cond is not None:
                # current_cond may reflect a line-search trial point, so evaluate the
                # true condition number at the accepted iterate xk before stopping.
                if not np.isnan(cond_xk) and cond_xk <= target_cond:
                    raise _TargetReached(np.array(xk, dtype=np.float64), cond_xk)

        minimize_kwargs = {
            "fun": objective,
            "x0": x0,
            "method": self.optimizer_method,
            "bounds": bounds,
            "callback": callback,
            "options": {"maxiter": max_iter},
        }
        if self.finite_diff_rel_step is not None:
            minimize_kwargs["jac"] = "2-point"
            minimize_kwargs["options"]["finite_diff_rel_step"] = self.finite_diff_rel_step

        # SLSQP: use proper inequality constraints instead of penalty
        active_indices = None
        if self.active_constraint_working_set:
            active_indices = self._active_constraint_indices(_get_trajectory(x0))
        if use_slsqp and (use_q_limits or use_direct_kin or self.ee_speed_max is not None):
            minimize_kwargs["constraints"] = self._build_q_constraints(_get_trajectory, active_indices)

        target_reached = False
        try:
            res = minimize(**minimize_kwargs)
            solver_x = np.asarray(res.x, dtype=np.float64)
        except _TargetReached as e:
            # The callback exception is an intentional termination path.  It is
            # represented explicitly rather than fabricating a scipy OptimizeResult.
            solver_x = e.x
            current_cond = e.cond
            target_reached = True
            res = None
            print(f"  {restart_label}target condition number reached (Cond = {e.cond:.4f}); stopping optimizer early.")

        solver_trajectory = _get_trajectory(solver_x)
        solver_obj, solver_cond = BaseTrajectory.compute_objective_with_cond(
            self.time_array,
            self.kinematics_func,
            *solver_trajectory,
            objective_type=self.objective_type,
            column_scale=self.column_scale,
        )
        solver_violation = retain_feasible(solver_x, solver_obj, solver_cond)
        returned_feasible_incumbent = best_feasible_x is not None and not np.array_equal(best_feasible_x, solver_x)
        x_result = best_feasible_x.copy() if best_feasible_x is not None else solver_x

        # Re-evaluate at the returned iterate.  Values cached by the objective can
        # correspond to a line-search trial point, not res.x.
        q_result, dq_result, ddq_result = _get_trajectory(x_result)
        final_obj, final_cond = BaseTrajectory.compute_objective_with_cond(
            self.time_array,
            self.kinematics_func,
            q_result,
            dq_result,
            ddq_result,
            objective_type=self.objective_type,
            column_scale=self.column_scale,
        )

        # For direct dq/ddq constraints, SLSQP may return an infeasible iterate
        # (line-search / subproblem failure) whose low condition number is an
        # artifact of over-large amplitudes. Report the peak limit violation so the
        # restart loop can prefer feasible solutions.
        kin_violation = self._kin_violation((q_result, dq_result, ddq_result)) if use_direct_kin else 0.0
        constraint_violation = self._constraint_violation_ratio((q_result, dq_result, ddq_result))
        native_violations = self._native_constraint_violations((q_result, dq_result, ddq_result))

        wall_time = time.time() - start_time
        solver_result = {
            "success": bool(res.success) if res is not None else False,
            "status": int(res.status) if res is not None else None,
            "message": str(res.message) if res is not None else "target condition number reached",
            "nit": int(getattr(res, "nit", 0)) if res is not None else int(it_count),
            "nfev": int(getattr(res, "nfev", 0)) if res is not None else None,
            "njev": int(getattr(res, "njev", 0)) if res is not None else None,
            "target_reached": target_reached,
            "objective_value": float(final_obj),
            "condition_number": float(final_cond),
            "solver_fun": float(res.fun) if res is not None and np.isfinite(res.fun) else None,
            "solver_returned_constraint_violation_ratio": float(solver_violation),
            "returned_feasible_incumbent": bool(returned_feasible_incumbent),
            "best_feasible_objective_value": (
                float(best_feasible_obj) if best_feasible_x is not None else None
            ),
            "best_feasible_condition_number": (
                float(best_feasible_cond) if best_feasible_x is not None else None
            ),
            "wall_time_s": float(wall_time),
            "kinematic_violation_ratio": float(kin_violation),
            "constraint_violation_ratio": float(constraint_violation),
            "native_constraint_violations": native_violations,
        }
        return x_result, final_cond, final_obj, wall_time, constraint_violation, solver_result

    def _kin_violation(self, trajectory) -> float:
        """Peak dq/ddq limit overshoot for a trajectory (<=0 means feasible)."""
        _, dq_r, ddq_r = trajectory
        violation = 0.0
        if self.dq_max is not None:
            violation = max(violation, float(np.max(np.abs(dq_r) / self.dq_max)) - 1.0)
        if self.ddq_max is not None:
            violation = max(violation, float(np.max(np.abs(ddq_r) / self.ddq_max)) - 1.0)
        return violation

    def _constraint_violation_ratio(self, trajectory) -> float:
        """Largest configured q/dq/ddq violation as a dimensionless ratio."""
        q, dq, ddq = trajectory
        violation = 0.0
        if self.q_min is not None:
            mask = np.isfinite(self.q_min)
            if np.any(mask):
                violation = max(
                    violation,
                    float(np.max((self.q_min[mask] - q[:, mask]) / self._q_residual_scale()[mask])),
                )
        if self.q_max is not None:
            mask = np.isfinite(self.q_max)
            if np.any(mask):
                violation = max(
                    violation,
                    float(np.max((q[:, mask] - self.q_max[mask]) / self._q_residual_scale()[mask])),
                )
        if self.dq_max is not None:
            mask = np.isfinite(self.dq_max)
            if np.any(mask):
                violation = max(violation, float(np.max(np.abs(dq[:, mask]) / self.dq_max[mask] - 1.0)))
        if self.ddq_max is not None:
            mask = np.isfinite(self.ddq_max)
            if np.any(mask):
                violation = max(violation, float(np.max(np.abs(ddq[:, mask]) / self.ddq_max[mask] - 1.0)))
        if self.ee_speed_max is not None:
            speed = np.asarray(self.ee_speed_func(q, dq), dtype=np.float64)
            violation = max(violation, float(np.max(speed / self.ee_speed_max - 1.0)))
        return max(0.0, violation)

    def _native_constraint_violations(self, trajectory) -> dict[str, list[float]]:
        """Return per-joint native-unit violations without mixing m and rad."""
        q, dq, ddq = trajectory
        q_violation = np.zeros(self.num_joints, dtype=np.float64)
        dq_violation = np.zeros(self.num_joints, dtype=np.float64)
        ddq_violation = np.zeros(self.num_joints, dtype=np.float64)
        if self.q_min is not None:
            mask = np.isfinite(self.q_min)
            if np.any(mask):
                q_violation[mask] = np.maximum(q_violation[mask], np.max(self.q_min[mask] - q[:, mask], axis=0))
        if self.q_max is not None:
            mask = np.isfinite(self.q_max)
            if np.any(mask):
                q_violation[mask] = np.maximum(q_violation[mask], np.max(q[:, mask] - self.q_max[mask], axis=0))
        if self.dq_max is not None:
            mask = np.isfinite(self.dq_max)
            if np.any(mask):
                dq_violation[mask] = np.maximum(
                    dq_violation[mask], np.max(np.abs(dq[:, mask]) - self.dq_max[mask], axis=0)
                )
        if self.ddq_max is not None:
            mask = np.isfinite(self.ddq_max)
            if np.any(mask):
                ddq_violation[mask] = np.maximum(
                    ddq_violation[mask], np.max(np.abs(ddq[:, mask]) - self.ddq_max[mask], axis=0)
                )
        return {
            "q": np.maximum(0.0, q_violation).tolist(),
            "dq": np.maximum(0.0, dq_violation).tolist(),
            "ddq": np.maximum(0.0, ddq_violation).tolist(),
        }

    @staticmethod
    def _is_better_candidate(
        cond,
        viol,
        best_cond,
        best_violation,
        feas_tol,
        objective_value=None,
        best_objective_value=None,
    ):
        """Return whether a restart should replace the current candidate.

        Candidates are ordered first by feasibility, then by the configured
        objective.  Solver termination status is diagnostic metadata: a feasible
        incumbent stopped by the iteration limit can be better than a converged
        but poor local optimum.  The objective value is not always ``cond`` (for
        example, D-optimal optimization).
        """
        if np.isnan(cond):
            return False
        objective_value = cond if objective_value is None else objective_value
        best_objective_value = best_cond if best_objective_value is None else best_objective_value
        if np.isnan(objective_value):
            return False

        feasible = viol <= feas_tol
        best_feasible = best_violation <= feas_tol
        if feasible != best_feasible:
            return feasible
        if feasible:
            return np.isnan(best_objective_value) or objective_value < best_objective_value
        return viol < best_violation - 1e-12 or (
            abs(viol - best_violation) <= 1e-12
            and (np.isnan(best_objective_value) or objective_value < best_objective_value)
        )

    def _optimize(self):  # noqa: C901
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
            B = self.coeff_bounds[j] / self.decision_variable_scale[j]
            bounds.extend([(-B, B)] * self.num_harmonics)
        bounds = bounds * 2
        decision_scale_vector = self._decision_scale_vector()

        rng = np.random.default_rng(self.seed)

        feas_tol = 1e-3
        t0 = time.time()
        prior_wall_time = 0.0
        completed_restarts = 0
        if self.restart_state is None:
            best_x = None
            best_cond = float("inf")
            best_violation = float("inf")
            best_objective_value = float("inf")
            best_idx = 0
            best_solver_success = False
            no_improve_count = 0
            self.optimization_results = []
        else:
            state = self.restart_state
            if state.get("objective_type") != self.objective_type:
                raise ValueError(
                    "restart checkpoint objective_type does not match configuration: "
                    f"{state.get('objective_type')!r} != {self.objective_type!r}"
                )
            if int(state.get("seed", -1)) != self.seed:
                raise ValueError(
                    f"restart checkpoint seed does not match configuration: {state.get('seed')!r} != {self.seed}"
                )
            completed_restarts = int(state.get("completed_restarts", -1))
            checkpoint_results = state.get("restarts")
            if not isinstance(checkpoint_results, list) or completed_restarts != len(checkpoint_results):
                raise ValueError("restart checkpoint completed_restarts does not match its restart record count")
            if not 0 < completed_restarts <= self.n_restarts:
                raise ValueError(
                    f"restart checkpoint completed_restarts must be in [1, {self.n_restarts}], "
                    f"got {completed_restarts}"
                )
            best_idx = int(state.get("selected_restart_index", 0)) - 1
            if not 0 <= best_idx < completed_restarts:
                raise ValueError("restart checkpoint selected_restart_index is out of range")
            self.optimization_results = [dict(result) for result in checkpoint_results]
            coefficient_count = 2 * self.num_joints * self.num_harmonics
            best_coefficients = self.optimization_results[best_idx].get("final_normalized_coefficients")
            if not isinstance(best_coefficients, list) or len(best_coefficients) != coefficient_count:
                raise ValueError(
                    "restart checkpoint selected candidate has an invalid final_normalized_coefficients length"
                )
            best_x = np.asarray(best_coefficients, dtype=np.float64)
            if not np.all(np.isfinite(best_x)):
                raise ValueError("restart checkpoint selected candidate coefficients must be finite")
            best_cond = float(state["best_condition_number"])
            best_violation = float(state["best_constraint_violation_ratio"])
            best_objective_value = float(state["best_objective_value"])
            best_solver_success = bool(state.get("selected_solver_success", False))
            no_improve_count = int(state.get("no_improve_count", 0))
            prior_wall_time = float(
                state.get(
                    "optimization_wall_time_s",
                    sum(float(result.get("wall_time_s", 0.0)) for result in self.optimization_results),
                )
            )
            print(
                f"Resuming after {completed_restarts}/{self.n_restarts} completed restarts; "
                f"current best is restart {best_idx + 1}."
            )

        for i in range(self.n_restarts):
            if i == 0:
                physical_x0 = np.concatenate([self.a.flatten(), self.b.flatten()])
                x0 = physical_x0 / decision_scale_vector
            else:
                x0 = self._generate_random_x0(rng)
            if i < completed_restarts:
                continue

            label = f"[{i + 1}/{self.n_restarts}] " if self.n_restarts > 1 else ""
            x_opt, cond, fun, wall, viol, solver_result = self._run_single_optimization(
                x0, q_main, dq_main, ddq_main, bounds, restart_label=label
            )

            solver_result["restart_index"] = i + 1
            solver_result["initial_normalized_coefficients"] = x0.tolist()
            solver_result["initial_coefficients"] = self._to_physical_coefficients(x0).tolist()
            solver_result["final_normalized_coefficients"] = x_opt.tolist()
            solver_result["final_coefficients"] = self._to_physical_coefficients(x_opt).tolist()
            self.optimization_results.append(solver_result)

            # Solver success is retained for diagnostics, but candidate quality is
            # determined by constraint feasibility and the configured objective.
            # In particular, an iteration-limited feasible incumbent may replace a
            # converged but much worse local optimum.
            solver_success = solver_result["success"]

            metric_improved = best_x is None or self._is_better_candidate(
                cond,
                viol,
                best_cond,
                best_violation,
                feas_tol,
                objective_value=fun,
                best_objective_value=best_objective_value,
            )
            improved = best_x is None or metric_improved

            marker = " *" if improved else ""
            viol_str = f", viol = {viol:.3f}" if viol > feas_tol else ""
            if use_d_optimal:
                print(
                    f"  restart {i + 1}/{self.n_restarts}: Cond = {cond:.4f}, D-opt = {fun:.4f}"
                    f"{viol_str} ({wall:.1f}s){marker}"
                )
            else:
                print(f"  restart {i + 1}/{self.n_restarts}: Cond = {cond:.4f}{viol_str} ({wall:.1f}s){marker}")

            if improved:
                best_cond = cond
                best_violation = viol
                best_objective_value = fun
                best_x = x_opt.copy()
                best_idx = i
                best_solver_success = solver_success
                no_improve_count = 0
            else:
                no_improve_count += 1

            if self.restart_callback is not None:
                self.restart_callback(
                    {
                        "objective_type": self.objective_type,
                        "seed": self.seed,
                        "completed_restarts": i + 1,
                        "selected_restart_index": best_idx + 1,
                        "selected_solver_success": best_solver_success,
                        "best_objective_value": float(best_objective_value),
                        "best_condition_number": float(best_cond),
                        "best_constraint_violation_ratio": float(best_violation),
                        "no_improve_count": no_improve_count,
                        "optimization_wall_time_s": prior_wall_time + time.time() - t0,
                        "restarts": self.optimization_results.copy(),
                    }
                )

            if (
                self.target_condition_number is not None
                and best_x is not None
                and not np.isnan(best_cond)
                and best_cond <= self.target_condition_number
            ):
                print(
                    f"  Target condition number {self.target_condition_number} reached "
                    f"(best Cond = {best_cond:.4f}); stopping restarts."
                )
                break

            if self.n_restarts > 1 and no_improve_count >= self.early_stop_patience:
                print(
                    f"  Early stop: no improvement for {self.early_stop_patience} restarts "
                    f"(best Cond = {best_cond:.4f})"
                )
                break

        split = self.num_joints * self.num_harmonics
        best_physical_x = self._to_physical_coefficients(best_x)
        self.a = best_physical_x[:split].reshape(self.num_joints, self.num_harmonics)
        self.b = best_physical_x[split:].reshape(self.num_joints, self.num_harmonics)

        self._is_optimized = True
        self.best_restart_index = best_idx + 1
        self.best_solver_success = best_solver_success
        self.final_condition_number = float(best_cond) if not np.isnan(best_cond) else None
        total_time = prior_wall_time + time.time() - t0
        if best_solver_success:
            print(f"Optimization Finished. Best Cond = {best_cond:.4f} (restart {best_idx + 1}, {total_time:.1f}s total)")
        else:
            print(
                f"Selected incumbent did not report solver convergence; Best Cond = {best_cond:.4f} "
                f"(restart {best_idx + 1}, {total_time:.1f}s total)"
            )
        if best_violation > feas_tol:
            print(
                f"Warning: best solution violates q/dq/ddq limits by {100.0 * best_violation:.2f}% "
                "of its configured scale (no feasible restart found); the reported condition number "
                "is not achievable within the kinematic envelope."
            )

        if self.optimization_results and not any(r["success"] for r in self.optimization_results):
            print("Warning: no restart reported solver success; selected iterate is a diagnostic fallback only.")

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

    def _trajectory_metadata(self) -> dict:
        meta: dict = {
            "objective_type": self.objective_type,
            "column_scale": self.column_scale,
            "decision_variable_scale": self.decision_variable_scale.tolist(),
            "finite_diff_rel_step": self.finite_diff_rel_step,
            "ee_speed_max": self.ee_speed_max,
            "active_constraint_working_set": self.active_constraint_working_set,
            "active_constraint_update_interval": self.active_constraint_update_interval,
        }
        if self.final_condition_number is not None:
            meta["condition_number"] = self.final_condition_number
        return meta
