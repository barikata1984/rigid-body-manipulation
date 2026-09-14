"""Run the primary condition-number vs D-optimal excitation comparison.

The two objective variants are deliberately executed in separate processes.  Each
process builds its own MuJoCo model and writes only to its own run directory, so no
catalog or log file is shared between concurrent runs.

Usage::

    python experiments/compare_excitation_objectives.py

The output directory can be changed with ``--output-root``.  It contains one JSON
summary per objective and a combined ``summary.json`` manifest.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Avoid two optimizer processes each creating a full BLAS thread pool.  The
# expensive part here is the Python/MuJoCo regressor evaluation, not a large BLAS
# matrix multiplication, so one native thread per worker is the predictable choice.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_kinematics_func():
    """Build the same hammer regressor closure as ``trajectories.generate``."""
    import mujoco
    from dm_control import mjcf
    from mujoco._structs import MjData, MjModel

    from dynamics import make_kinematics_func, setup_robot_dynamics_parameters
    from simulators.setup import spawn_target_object

    manipulator_dir = REPO_ROOT / "xml_models" / "manipulators" / "sequential"
    target_dir = REPO_ROOT / "xml_models" / "targets" / "hammer"
    target_object, assets, _ = spawn_target_object(
        target_dir / "object.xml", target_dir / "object_cad_gt.csv", compare_cad_mujoco=False
    )
    manipulator = mjcf.from_path(str(manipulator_dir / "manipulator.xml"))
    attachment_site = manipulator.find("site", "attachment")
    if attachment_site is None:
        raise RuntimeError("sequential manipulator has no attachment site")
    attachment_site.attach(target_object)

    model = MjModel.from_xml_string(manipulator.to_xml_string(filename_with_hash=False), assets=assets)
    data = MjData(model)
    mujoco.mj_forward(model, data)
    params = setup_robot_dynamics_parameters(model, data, ee_body_name="link6")
    pose_x_sen = params.poses.get_x_("site", "target/ft_sensor")
    return make_kinematics_func(params, pose_x_sen)


def _config_dict(
    objective_type: str,
    *,
    max_iter: int = 20,
    n_restarts: int = 3,
    seed: int = 42,
    constraint_safety_factor: float = 1.0,
) -> dict[str, Any]:
    raw_condition_diagnostic = objective_type == "condition_number_raw"
    configured_objective = "condition_number" if raw_condition_diagnostic else objective_type
    pi = 3.141592653589793
    return {
        "duration": 10.0,
        "fps": 60.0,
        "num_joints": 6,
        "num_harmonics": 5,
        "base_freq": 0.1,
        "manipulator": "xml_models/manipulators/sequential",
        "object": "xml_models/targets/hammer",
        "ee_body_name": "link6",
        "max_iter": max_iter,
        "q_min": [-0.5 * constraint_safety_factor] * 3 + [-pi * constraint_safety_factor] * 3,
        "q_max": [0.5 * constraint_safety_factor] * 3 + [pi * constraint_safety_factor] * 3,
        "dq_max": [1.5 * constraint_safety_factor] * 3 + [pi * constraint_safety_factor] * 3,
        "ddq_max": [3.0 * constraint_safety_factor] * 3 + [2.0 * pi * constraint_safety_factor] * 3,
        # Explicit +/-inf means that a,b are optimization variables without a
        # finite coefficient box.  This avoids changing the legacy default of
        # ExcitedTrajectoryConfig (which remains 0.5 for old configurations).
        "coeff_bounds": [float("inf")] * 6,
        # Nondimensionalize optimizer coordinates by the per-axis half-range.
        # This is a numerical preconditioner, not a Fourier coefficient bound.
        "decision_variable_scale": [0.5, 0.5, 0.5, pi, pi, pi],
        "use_analytical_bounds": False,
        # The established condition-number formulation needs column equilibration
        # because its value otherwise changes with parameter units.  Raw D-opt is
        # intentionally left unscaled so it rewards absolute information rather
        # than only the correlation structure of unit-norm columns.
        "column_scale": objective_type == "condition_number",
        "objective_type": configured_objective,
        "optimizer_method": "SLSQP",
        "finite_diff_rel_step": 1e-6,
        "n_restarts": n_restarts,
        "seed": seed,
        "early_stop_patience": 1_000_000,
        "target_condition_number": None,
        "main_trajectory": {
            "target_class": "SplineTrajectory",
            "module_name": "trajectories",
            "type": "quintic",
            "duration": 10.0,
            "fps": 60.0,
            "start_pos": [0.0] * 6,
            "end_pos": [0.0] * 6,
        },
    }


def _dense_trajectory(coefficients: dict[str, Any], duration: float, fps: float, base_freq: float = 0.1):
    """Evaluate the optimized windowed Fourier trajectory on a dense grid."""
    import numpy as np

    from trajectories.fourier import FourierTrajectory, FourierTrajectoryConfig
    from trajectories.window import WindowTrajectory, WindowTrajectoryConfig

    num_joints = len(coefficients["a"])
    num_harmonics = len(coefficients["a"][0])
    f_cfg = FourierTrajectoryConfig(
        duration=duration,
        fps=fps,
        num_joints=num_joints,
        num_harmonics=num_harmonics,
        base_freq=base_freq,
        coefficients=coefficients,
    )
    fourier = FourierTrajectory(f_cfg)
    # BaseTrajectory uses int(T*fps) samples and includes both endpoints.  Use
    # the same endpoint convention with an explicit dense 500-Hz-equivalent grid.
    fourier.time_array = np.linspace(0.0, duration, int(duration * fps) + 1)
    window = WindowTrajectory(
        WindowTrajectoryConfig(duration=duration, fps=fps, num_joints=num_joints)
    )
    window.time_array = fourier.time_array
    return window.apply(*fourier.get_value())


def _constraint_metrics(q, dq, ddq, config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    q_min = np.asarray(config["q_min"], dtype=float)
    q_max = np.asarray(config["q_max"], dtype=float)
    dq_max = np.asarray(config["dq_max"], dtype=float)
    ddq_max = np.asarray(config["ddq_max"], dtype=float)
    margins = {
        "q_min": q - q_min,
        "q_max": q_max - q,
        "dq": dq_max - np.abs(dq),
        "ddq": ddq_max - np.abs(ddq),
    }
    return {
        "max_abs": {
            "q": np.max(np.abs(q), axis=0).tolist(),
            "dq": np.max(np.abs(dq), axis=0).tolist(),
            "ddq": np.max(np.abs(ddq), axis=0).tolist(),
        },
        "min_margin": {name: float(np.min(value)) for name, value in margins.items()},
        "min_margin_by_joint": {name: np.min(value, axis=0).tolist() for name, value in margins.items()},
        "max_violation": {
            name: float(max(0.0, -np.min(value))) for name, value in margins.items()
        },
        "max_violation_native": float(max(0.0, max(-np.min(value) for value in margins.values()))),
        "sample_count": int(q.shape[0]),
        "fps_equivalent": 500.0,
    }


def _stack_regressor(time_array, q, dq, ddq, kinematics_func):
    import numpy as np

    return np.vstack([kinematics_func(q_i, dq_i, ddq_i) for q_i, dq_i, ddq_i in zip(q, dq, ddq, strict=True)])


def _information_metrics(regressor):
    import numpy as np

    regressor = np.asarray(regressor)
    info = regressor.T @ regressor
    singular_values = np.linalg.svd(regressor, compute_uv=False)
    eigvals = np.linalg.eigvalsh(info)
    diagonal = np.clip(np.diag(info), 1e-30, None)
    scale = 1.0 / np.sqrt(diagonal)
    scaled_regressor = regressor * scale
    # The scaled information matrix is the Gram matrix of scaled_regressor;
    # its condition number is therefore approximately the square of the
    # scaled-regressor condition number (up to numerical roundoff).
    scaled_info = scaled_regressor.T @ scaled_regressor
    scaled_eigvals = np.linalg.eigvalsh(scaled_info)
    return {
        "shape": [int(x) for x in regressor.shape],
        "raw_cond_F": float(np.linalg.cond(regressor)),
        "raw_cond_information": float(np.linalg.cond(info)),
        "sigma_min_F": float(singular_values[-1]),
        "lambda_min_information": float(eigvals[0]),
        "column_scaled_cond_F": float(np.linalg.cond(scaled_regressor)),
        "column_scaled_cond_information": float(np.linalg.cond(scaled_info)),
        "column_scaled_d_optimal": float(-np.sum(np.log(np.maximum(scaled_eigvals, 1e-30)))),
        "information_eigenvalues": eigvals.tolist(),
        "F_singular_values": singular_values.tolist(),
    }


def _json_safe(value):
    """Convert NumPy/dataclass values while retaining Infinity explicitly."""
    import math

    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {float("inf"): "Infinity", float("-inf"): "-Infinity"}.get(value, "NaN")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    """Replace a JSON checkpoint only after the new payload is fully written."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _run_one(
    objective_type: str,
    output_root: str,
    max_iter: int,
    n_restarts: int,
    seed: int,
    constraint_safety_factor: float,
    resume: bool,
) -> dict[str, Any]:
    import numpy as np

    from trajectories.excited import ExcitedTrajectory, ExcitedTrajectoryConfig
    from trajectories.spline import SplineTrajectoryConfig

    output_dir = Path(output_root) / objective_type
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "optimize.log"
    orig_config = _config_dict(
        objective_type,
        max_iter=max_iter,
        n_restarts=n_restarts,
        seed=seed,
        constraint_safety_factor=constraint_safety_factor,
    )
    config_path = output_dir / "config.json"
    checkpoint_path = output_dir / "restart_checkpoint.json"
    serialized_config = _json_safe(orig_config)
    restart_state = None
    if resume:
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requires existing config.json and restart_checkpoint.json")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != serialized_config:
            raise ValueError("resume configuration does not match the existing config.json")
        restart_state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    else:
        config_path.write_text(json.dumps(serialized_config, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with log_path.open("a" if resume else "w", buffering=1, encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            if resume:
                print("--- resuming from restart_checkpoint.json ---")
            print(f"objective_type={objective_type}")
            print(f"pid={os.getpid()} host={platform.node()}")
            print("compute_fourier_bounds_called=False (use_analytical_bounds=False)")
            start = time.perf_counter()
            kinematics_func = _build_kinematics_func()
            main_cfg = SplineTrajectoryConfig(**orig_config["main_trajectory"])
            cfg_values = {k: v for k, v in orig_config.items() if k != "main_trajectory"}
            cfg_values["main_trajectory"] = main_cfg
            cfg = ExcitedTrajectoryConfig(**cfg_values)
            trajectory = ExcitedTrajectory(
                cfg,
                kinematics_func=kinematics_func,
                restart_callback=lambda payload: _write_json_atomic(checkpoint_path, payload),
                restart_state=restart_state,
            )
            q, dq, ddq = trajectory.generate(
                show_plot=False,
                plot_path=None,
                json_path=str(output_dir / "trajectory.json"),
                metadata={
                    "experiment": "condition_number_vs_d_optimal_primary",
                    "compute_fourier_bounds_called": False,
                    "configuration": _json_safe(orig_config),
                },
            )
            generation_wall = time.perf_counter() - start
            print(f"generation_wall_time_s={generation_wall:.6f}")

            dense_q, dense_dq, dense_ddq = _dense_trajectory(
                {"a": trajectory.a, "b": trajectory.b, "q0": np.zeros(trajectory.num_joints)},
                duration=cfg.duration,
                fps=500.0,
                base_freq=cfg.base_freq,
            )
            dense_constraints = _constraint_metrics(dense_q, dense_dq, dense_ddq, orig_config)

            # Information metrics use exactly the optimizer's clean planned grid,
            # not the dense constraint grid, so the objective and the report share
            # the same definition.
            regressor = _stack_regressor(trajectory.time_array, q, dq, ddq, kinematics_func)
            information = _information_metrics(regressor)
            result = {
                "experiment_variant": objective_type,
                "objective_type": orig_config["objective_type"],
                "configuration": _json_safe(orig_config),
                "compute_fourier_bounds_called": False,
                "generation_wall_time_s": float(generation_wall),
                "solver": {
                    "restarts": _json_safe(trajectory.optimization_results),
                    "selected_restart_index": trajectory.best_restart_index,
                    "selected_solver_success": trajectory.best_solver_success,
                    "final_condition_number": trajectory.final_condition_number,
                },
                "dense_constraints": dense_constraints,
                "information": information,
                "coefficient_statistics": {
                    "a_max_abs": float(np.max(np.abs(trajectory.a))),
                    "b_max_abs": float(np.max(np.abs(trajectory.b))),
                    "a": trajectory.a.tolist(),
                    "b": trajectory.b.tolist(),
                },
                "trajectory_json": str(output_dir / "trajectory.json"),
                "log": str(log_path),
                "restart_checkpoint": str(checkpoint_path),
            }
            result = _json_safe(result)
            (output_dir / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, indent=2, allow_nan=True))
            return result


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "results" / "excitation_objective_ab_20260827_seed42"),
        help="directory for separate objective outputs and combined manifest",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--constraint-safety-factor", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="continue after completed restarts in the output checkpoint")
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("condition_number", "condition_number_raw", "d_optimal"),
        default=("condition_number", "d_optimal"),
    )
    args = parser.parse_args()
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if args.n_restarts <= 0:
        parser.error("--n-restarts must be positive")
    if not 0.0 < args.constraint_safety_factor <= 1.0:
        parser.error("--constraint-safety-factor must be in (0, 1]")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    objectives = tuple(args.objectives)
    started = time.perf_counter()
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    context = mp.get_context("spawn")
    max_workers = min(max(1, args.workers), len(objectives))
    print(f"cpu_count={os.cpu_count()} workers={max_workers} output_root={output_root}")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                _run_one,
                objective,
                str(output_root),
                args.max_iter,
                args.n_restarts,
                args.seed,
                args.constraint_safety_factor,
                args.resume,
            ): objective
            for objective in objectives
        }
        for future in as_completed(futures):
            objective = futures[future]
            try:
                results[objective] = future.result()
                print(f"completed objective={objective}")
            except Exception as exc:  # keep the other objective's result/report
                errors[objective] = f"{type(exc).__name__}: {exc}"
                print(f"failed objective={objective}: {errors[objective]}")

    manifest = {
        "experiment": "condition_number_vs_d_optimal_primary",
        "git_revision": _git_revision(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "workers": max_workers,
        "started_at_epoch_s": started,
        "wall_time_s": time.perf_counter() - started,
        "results": results,
        "errors": errors,
    }
    (output_root / "summary.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_safe(manifest), indent=2, allow_nan=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
