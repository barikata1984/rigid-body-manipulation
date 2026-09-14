"""Run 24 independent D-optimal excitation restarts on the 300-frame setup."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.compare_excitation_objectives import (  # noqa: E402
    _build_kinematics_func,
    _constraint_metrics,
    _dense_trajectory,
    _information_metrics,
    _json_safe,
    _stack_regressor,
)


def _configuration(
    seed: int,
    max_iter: int = 100,
    *,
    active_constraint_working_set: bool = True,
) -> dict[str, Any]:
    pi = 3.141592653589793
    return {
        "duration": 5.0,
        "fps": 60.0,
        "num_joints": 6,
        "num_harmonics": 5,
        "base_freq": 0.2,
        "manipulator": "xml_models/manipulators/sequential",
        "object": "xml_models/targets/hammer",
        "ee_body_name": "link6",
        "max_iter": max_iter,
        "q_min": [-0.5] * 3 + [-pi] * 3,
        "q_max": [0.5] * 3 + [pi] * 3,
        "dq_max": [0.5] * 3 + [pi] * 3,
        "ddq_max": [1.0] * 3 + [2.0 * pi] * 3,
        "ee_speed_max": 1.0,
        "optimizer_constraint_safety_factor": 0.99,
        "dense_constraint_tolerance_ratio": 1e-3,
        "coeff_bounds": [float("inf")] * 6,
        "decision_variable_scale": [0.5, 0.5, 0.5, pi, pi, pi],
        "use_analytical_bounds": False,
        "column_scale": False,
        "objective_type": "d_optimal",
        "optimizer_method": "SLSQP",
        "finite_diff_rel_step": 1e-6,
        "active_constraint_working_set": active_constraint_working_set,
        "active_constraint_update_interval": 10,
        "n_restarts": 1,
        "seed": seed,
        "early_stop_patience": 1_000_000,
        "target_condition_number": None,
        "main_trajectory": {
            "target_class": "SplineTrajectory",
            "module_name": "trajectories",
            "type": "quintic",
            "duration": 5.0,
            "fps": 60.0,
            "start_pos": [0.0] * 6,
            "end_pos": [0.0] * 6,
        },
    }


def _ee_speed(_q, dq):
    """Linear speed of the sequential manipulator's coincident attachment origin."""
    import numpy as np

    return np.linalg.norm(dq[:, :3], axis=1)


def _initial_point(trajectory, seed: int, restart_index: int):
    """Reproduce the initial point used by the corresponding serial multistart."""
    import numpy as np

    if restart_index == 0:
        physical = np.concatenate([trajectory.a.ravel(), trajectory.b.ravel()])
        return physical / trajectory._decision_scale_vector()
    rng = np.random.default_rng(seed)
    point = None
    for _ in range(restart_index):
        point = trajectory._generate_random_x0(rng)
    return point


def _dense_metrics(trajectory, config):
    import numpy as np

    dense_q, dense_dq, dense_ddq = _dense_trajectory(
        {"a": trajectory.a, "b": trajectory.b, "q0": np.zeros(trajectory.num_joints)},
        duration=config["duration"],
        fps=500.0,
        base_freq=config["base_freq"],
    )
    constraints = _constraint_metrics(dense_q, dense_dq, dense_ddq, config)
    ee_speed = _ee_speed(dense_q, dense_dq)
    constraints["ee_speed"] = {
        "max": float(np.max(ee_speed)),
        "limit": float(config["ee_speed_max"]),
        "margin": float(config["ee_speed_max"] - np.max(ee_speed)),
    }
    constraints["max_violation_native"] = max(
        constraints["max_violation_native"],
        max(0.0, constraints["ee_speed"]["max"] - constraints["ee_speed"]["limit"]),
    )
    q_min = np.asarray(config["q_min"], dtype=float)
    q_max = np.asarray(config["q_max"], dtype=float)
    q_scale = 0.5 * (q_max - q_min)
    dq_max = np.asarray(config["dq_max"], dtype=float)
    ddq_max = np.asarray(config["ddq_max"], dtype=float)
    constraints["max_violation_ratio"] = float(
        max(
            0.0,
            np.max((q_min - dense_q) / q_scale),
            np.max((dense_q - q_max) / q_scale),
            np.max(np.abs(dense_dq) / dq_max - 1.0),
            np.max(np.abs(dense_ddq) / ddq_max - 1.0),
            np.max(ee_speed / config["ee_speed_max"] - 1.0),
        )
    )
    constraints["sample_count"] = int(dense_q.shape[0])
    return dense_q, dense_dq, dense_ddq, constraints


def _run_restart(
    seed: int,
    restart_index: int,
    output_root: str,
    max_iter: int = 100,
    active_constraint_working_set: bool = True,
) -> dict[str, Any]:
    import numpy as np

    from trajectories.excited import ExcitedTrajectory, ExcitedTrajectoryConfig
    from trajectories.spline import SplineTrajectoryConfig

    run_dir = Path(output_root) / f"seed{seed}_restart{restart_index + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))

    config = _configuration(
        seed,
        max_iter=max_iter,
        active_constraint_working_set=active_constraint_working_set,
    )
    (run_dir / "config.json").write_text(
        json.dumps(_json_safe(config), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    log_path = run_dir / "optimize.log"
    with log_path.open("w", buffering=1, encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"pid={os.getpid()} host={platform.node()} seed={seed} restart={restart_index + 1}")
            print("compute_fourier_bounds_called=False (use_analytical_bounds=False)")
            started = time.perf_counter()
            kinematics_func = _build_kinematics_func()
            main_cfg = SplineTrajectoryConfig(**config["main_trajectory"])
            cfg_values = {
                key: value
                for key, value in config.items()
                if key
                not in {
                    "main_trajectory",
                    "optimizer_constraint_safety_factor",
                    "dense_constraint_tolerance_ratio",
                }
            }
            safety_factor = config["optimizer_constraint_safety_factor"]
            q_min = np.asarray(config["q_min"], dtype=float)
            q_max = np.asarray(config["q_max"], dtype=float)
            q_mid = 0.5 * (q_min + q_max)
            q_half_range = 0.5 * (q_max - q_min) * safety_factor
            cfg_values["q_min"] = (q_mid - q_half_range).tolist()
            cfg_values["q_max"] = (q_mid + q_half_range).tolist()
            cfg_values["dq_max"] = (np.asarray(config["dq_max"]) * safety_factor).tolist()
            cfg_values["ddq_max"] = (np.asarray(config["ddq_max"]) * safety_factor).tolist()
            cfg_values["main_trajectory"] = main_cfg
            trajectory = ExcitedTrajectory(
                ExcitedTrajectoryConfig(**cfg_values),
                kinematics_func=kinematics_func,
                ee_speed_func=_ee_speed,
            )
            q_main, dq_main, ddq_main = trajectory.main_trajectory.generate()
            bounds = [(-np.inf, np.inf)] * (2 * trajectory.num_joints * trajectory.num_harmonics)
            x0 = _initial_point(trajectory, seed, restart_index)
            x_opt, cond_info, objective, wall_time, violation, solver = trajectory._run_single_optimization(
                x0,
                q_main,
                dq_main,
                ddq_main,
                bounds,
                restart_label=f"[{restart_index + 1}/6] ",
            )

            split = trajectory.num_joints * trajectory.num_harmonics
            physical = trajectory._to_physical_coefficients(x_opt)
            trajectory.a = physical[:split].reshape(trajectory.num_joints, trajectory.num_harmonics)
            trajectory.b = physical[split:].reshape(trajectory.num_joints, trajectory.num_harmonics)
            trajectory._is_optimized = True
            q, dq, ddq = trajectory._generate()
            trajectory.write_to_json(
                q,
                dq,
                ddq,
                json_path=run_dir / "trajectory.json",
                metadata={"configuration": _json_safe(config), "seed": seed, "restart_index": restart_index + 1},
            )

            _, _, _, dense_constraints = _dense_metrics(trajectory, config)
            regressor = _stack_regressor(trajectory.time_array, q, dq, ddq, kinematics_func)
            information = _information_metrics(regressor)
            result = {
                "seed": seed,
                "restart_index": restart_index + 1,
                "configuration": _json_safe(config),
                "solver": _json_safe(solver),
                "objective_value": float(objective),
                "condition_number_information": float(cond_info),
                "optimizer_constraint_violation_ratio": float(violation),
                "dense_constraints": dense_constraints,
                "information": information,
                "wall_time_s": float(wall_time),
                "total_wall_time_s": float(time.perf_counter() - started),
                "coefficients": {"a": trajectory.a.tolist(), "b": trajectory.b.tolist()},
                "trajectory_json": str(run_dir / "trajectory.json"),
                "log": str(log_path),
            }
            result = _json_safe(result)
            result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "results" / "excitation_dopt_5s_300frames_20260901"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--smoke", action="store_true", help="run seed42/restart1 with max_iter=1")
    parser.add_argument("--single", action="store_true", help="run only seed42/restart1")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--fixed-constraints",
        action="store_true",
        help="evaluate every trajectory sample instead of using the active working set",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = (
        [(42, 0)]
        if args.smoke or args.single
        else [(seed, restart) for seed in range(42, 46) for restart in range(6)]
    )
    max_iter = 1 if args.smoke else args.max_iter

    started = time.perf_counter()
    results = []
    errors = []
    context = mp.get_context("spawn")
    workers = min(max(1, args.workers), len(tasks))
    print(f"workers={workers} tasks={len(tasks)} output_root={output_root}")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                _run_restart,
                seed,
                restart,
                str(output_root),
                max_iter,
                not args.fixed_constraints,
            ): (seed, restart)
            for seed, restart in tasks
        }
        for future in as_completed(futures):
            seed, restart = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"completed seed={seed} restart={restart + 1} "
                    f"status={result['solver']['status']} raw_cond_F={result['information']['raw_cond_F']:.6g}"
                )
            except Exception as exc:
                errors.append({"seed": seed, "restart_index": restart + 1, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed seed={seed} restart={restart + 1}: {errors[-1]['error']}")

    accepted = [
        result
        for result in results
        if result["solver"]["constraint_violation_ratio"] <= 1e-6
        and result["information"]["raw_cond_F"] < 10.0
        and result["dense_constraints"]["max_violation_ratio"]
        <= result["configuration"]["dense_constraint_tolerance_ratio"]
    ]
    accepted.sort(key=lambda result: result["objective_value"])
    summary = {
        "experiment": "d_optimal_5s_300frames",
        "workers": workers,
        "tasks": len(tasks),
        "wall_time_s": time.perf_counter() - started,
        "accepted_count": len(accepted),
        "selected": accepted[0] if accepted else None,
        "results": sorted(results, key=lambda result: (result["seed"], result["restart_index"])),
        "errors": errors,
    }
    (output_root / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_safe({key: value for key, value in summary.items() if key != "results"}), indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
