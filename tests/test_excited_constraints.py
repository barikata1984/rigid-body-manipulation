import importlib.util
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

_METRICS_PATH = Path(__file__).resolve().parents[1] / "experiments" / "compare_excitation_objectives.py"
_METRICS_SPEC = importlib.util.spec_from_file_location("compare_excitation_objectives_for_test", _METRICS_PATH)
if _METRICS_SPEC is None or _METRICS_SPEC.loader is None:
    raise ImportError(f"Cannot load comparison metrics from {_METRICS_PATH}")
_METRICS_MODULE = importlib.util.module_from_spec(_METRICS_SPEC)
_METRICS_SPEC.loader.exec_module(_METRICS_MODULE)
_information_metrics = _METRICS_MODULE._information_metrics
_write_json_atomic = _METRICS_MODULE._write_json_atomic

# Loading the comparison module first makes its repository-root bootstrap apply
# when this test is collected through the console-script pytest entry point.
import trajectories.excited as excited_module  # noqa: E402
from trajectories.excited import ExcitedTrajectory, ExcitedTrajectoryConfig  # noqa: E402


def _make_cfg(use_analytical_bounds: bool, **overrides) -> ExcitedTrajectoryConfig:
    cfg = ExcitedTrajectoryConfig(
        duration=2.0,
        fps=50.0,
        num_joints=2,
        num_harmonics=2,
        base_freq=0.3,
        coeff_bounds=[0.5, 0.5] if not use_analytical_bounds else None,
        dq_max=[1.0, 1.0],
        ddq_max=[5.0, 5.0],
        use_analytical_bounds=use_analytical_bounds,
        optimizer_method="SLSQP",
        main_trajectory={
            "target_class": "SplineTrajectory",
            "type": "quintic",
            "duration": 2.0,
            "fps": 50.0,
            "start_pos": [0.0, 0.0],
            "end_pos": [0.0, 0.0],
        },
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_analytical_bounds_replace_coeff_box():
    """With use_analytical_bounds=True the coeff box is the analytical bound (no manual box)."""
    traj = ExcitedTrajectory(_make_cfg(use_analytical_bounds=True))
    assert all(b < 0.5 for b in traj.coeff_bounds)


def test_coeff_bounds_forbidden_with_analytical_bounds():
    """coeff_bounds cannot be combined with an active analytical bound."""
    with pytest.raises(ValueError, match="coeff_bounds cannot be combined"):
        ExcitedTrajectory(_make_cfg(use_analytical_bounds=True, coeff_bounds=[0.5, 0.5]))


def test_direct_mode_keeps_manual_coeff_box():
    """With use_analytical_bounds=False the coeff box stays at the manual value."""
    traj = ExcitedTrajectory(_make_cfg(use_analytical_bounds=False))
    assert traj.coeff_bounds == [0.5, 0.5]


def test_direct_mode_accepts_unbounded_coefficients():
    """Explicit +/-inf bounds represent coefficient variables without a box."""
    traj = ExcitedTrajectory(_make_cfg(use_analytical_bounds=False, coeff_bounds=[np.inf, np.inf]))
    assert all(np.isinf(b) for b in traj.coeff_bounds)
    assert np.all(np.isfinite(traj._generate_random_x0(np.random.default_rng(42))))


def test_decision_variable_scale_maps_dimensionless_to_physical_coefficients():
    """Equal normalized coefficients represent equal fractions of each joint scale."""
    scales = np.array([0.5, np.pi])
    traj = ExcitedTrajectory(
        _make_cfg(
            use_analytical_bounds=False,
            coeff_bounds=[np.inf, np.inf],
            decision_variable_scale=scales.tolist(),
        )
    )
    normalized = np.full(8, 0.2)
    physical = traj._to_physical_coefficients(normalized)
    expected_one_block = np.repeat(0.2 * scales, 2)

    assert np.allclose(physical[:4], expected_one_block)
    assert np.allclose(physical[4:], expected_one_block)
    initial_physical = np.concatenate([traj.a.ravel(), traj.b.ravel()])
    initial_normalized = initial_physical / traj._decision_scale_vector()
    assert np.max(np.abs(initial_normalized)) <= 0.01


def test_decision_variable_scale_must_be_positive_and_finite():
    with pytest.raises(ValueError, match="finite and strictly positive"):
        ExcitedTrajectory(
            _make_cfg(use_analytical_bounds=False, decision_variable_scale=[0.5, 0.0])
        )


def test_finite_difference_relative_step_must_be_positive():
    with pytest.raises(ValueError, match="finite and strictly positive"):
        ExcitedTrajectory(_make_cfg(use_analytical_bounds=False, finite_diff_rel_step=0.0))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"q_min": [float("inf"), -1.0]}, "q_min values"),
        ({"q_max": [float("-inf"), 1.0]}, "q_max values"),
        ({"dq_max": [float("nan"), 1.0]}, "dq_max values"),
        ({"ddq_max": [float("-inf"), 1.0]}, "ddq_max values"),
    ],
)
def test_invalid_limit_values_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        ExcitedTrajectory(_make_cfg(use_analytical_bounds=False, **overrides))


def test_direct_mode_adds_dq_ddq_constraints():
    """Direct mode contributes velocity and acceleration inequality constraints."""
    traj = ExcitedTrajectory(_make_cfg(use_analytical_bounds=False))
    n_steps = 4
    dq_max = traj.dq_max
    ddq_max = traj.ddq_max

    def within(_x):
        q = np.zeros((n_steps, 2))
        dq = np.tile(0.5 * dq_max, (n_steps, 1))
        ddq = np.tile(0.5 * ddq_max, (n_steps, 1))
        return q, dq, ddq

    def violating(_x):
        q = np.zeros((n_steps, 2))
        dq = np.tile(2.0 * dq_max, (n_steps, 1))
        ddq = np.tile(2.0 * ddq_max, (n_steps, 1))
        return q, dq, ddq

    # q_min/q_max default to None here. Each symmetric dq/ddq limit is represented
    # by separate smooth positive-side and negative-side constraints.
    cons_within = traj._build_q_constraints(within)
    cons_viol = traj._build_q_constraints(violating)
    assert len(cons_within) == 4
    assert all(np.all(c["fun"](np.zeros(8)) >= 0) for c in cons_within)
    violating_residuals = [c["fun"](np.zeros(8)) for c in cons_viol]
    assert np.any(violating_residuals[0] < 0)
    assert np.any(violating_residuals[2] < 0)


def test_active_working_set_keeps_one_sample_per_joint_and_ee_limit():
    cfg = _make_cfg(
        use_analytical_bounds=False,
        q_min=[-0.5, -0.5],
        q_max=[0.5, 0.5],
        ee_speed_max=1.0,
        active_constraint_working_set=True,
    )
    traj = ExcitedTrajectory(cfg, ee_speed_func=lambda _q, dq: np.linalg.norm(dq, axis=1))
    q = np.array([[0.0, 0.0], [0.4, -0.1], [-0.45, 0.3], [0.1, 0.2]])
    dq = np.array([[0.0, 0.0], [0.8, 0.0], [-0.2, -0.9], [0.1, 0.2]])
    ddq = np.array([[0.0, 0.0], [4.0, -1.0], [-2.0, -4.5], [1.0, 1.0]])

    def sampled_trajectory(_x):
        return q, dq, ddq

    active = traj._active_constraint_indices((q, dq, ddq))
    constraints = traj._build_q_constraints(sampled_trajectory, active)
    residuals = [constraint["fun"](np.zeros(8)) for constraint in constraints]

    assert len(residuals) == 7
    assert [value.size for value in residuals] == [2, 2, 2, 2, 2, 2, 1]
    assert active["q_min"].tolist() == [2, 1]
    assert active["q_max"].tolist() == [1, 2]
    assert active["ee_speed"].tolist() == [2]


def test_ee_speed_is_included_in_candidate_feasibility():
    cfg = _make_cfg(use_analytical_bounds=False, ee_speed_max=1.0)
    traj = ExcitedTrajectory(cfg, ee_speed_func=lambda _q, dq: np.linalg.norm(dq, axis=1))
    q = np.zeros((2, 2))
    dq = np.array([[0.6, 0.8], [0.0, 1.2]])
    ddq = np.zeros_like(q)

    assert traj._constraint_violation_ratio((q, dq, ddq)) == pytest.approx(0.2)


def test_mixed_unit_constraint_residuals_are_dimensionless():
    """Meters and radians use comparable relative margins in SLSQP constraints."""
    q_min = np.array([-0.5, -np.pi])
    q_max = np.array([0.5, np.pi])
    dq_max = np.array([1.5, np.pi])
    ddq_max = np.array([3.0, 2.0 * np.pi])
    traj = ExcitedTrajectory(
        _make_cfg(
            use_analytical_bounds=False,
            q_min=q_min.tolist(),
            q_max=q_max.tolist(),
            dq_max=dq_max.tolist(),
            ddq_max=ddq_max.tolist(),
            decision_variable_scale=[0.5, np.pi],
        )
    )
    n_steps = 4

    def relative_half_range(_x):
        q = np.zeros((n_steps, 2))
        dq = np.tile(0.5 * dq_max, (n_steps, 1))
        ddq = np.tile(0.5 * ddq_max, (n_steps, 1))
        return q, dq, ddq

    constraints = traj._build_q_constraints(relative_half_range)
    residuals = [constraint["fun"](np.zeros(8)) for constraint in constraints]

    assert len(residuals) == 6
    assert np.allclose(residuals[0], 1.0)
    assert np.allclose(residuals[1], 1.0)
    assert np.allclose(residuals[2], 0.5)
    assert np.allclose(residuals[3], 1.5)
    assert np.allclose(residuals[4], 0.5)
    assert np.allclose(residuals[5], 1.5)


def test_restart_feasibility_includes_dimensionless_position_violation():
    q_min = [-0.5, -np.pi]
    q_max = [0.5, np.pi]
    traj = ExcitedTrajectory(
        _make_cfg(
            use_analytical_bounds=False,
            q_min=q_min,
            q_max=q_max,
            decision_variable_scale=[0.5, np.pi],
        )
    )
    q = np.array([[0.6, 0.0]])
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)

    assert traj._constraint_violation_ratio((q, dq, ddq)) == pytest.approx(0.2)
    native = traj._native_constraint_violations((q, dq, ddq))
    assert native["q"] == pytest.approx([0.1, 0.0])


def test_slsqp_uses_configured_relative_finite_difference(monkeypatch):
    cfg = _make_cfg(use_analytical_bounds=False, finite_diff_rel_step=1e-6, max_iter=1)
    traj = ExcitedTrajectory(cfg, kinematics_func=lambda _q, _dq, _ddq: np.eye(2))
    q_main, dq_main, ddq_main = traj.main_trajectory.generate()
    captured = {}

    def fake_minimize(**kwargs):
        captured.update(kwargs)
        x = np.asarray(kwargs["x0"])
        objective_value = kwargs["fun"](x)
        return SimpleNamespace(
            x=x,
            success=True,
            status=0,
            message="synthetic success",
            nit=1,
            nfev=1,
            njev=1,
            fun=objective_value,
        )

    monkeypatch.setattr(excited_module, "minimize", fake_minimize)
    x0 = np.concatenate([traj.a.ravel(), traj.b.ravel()]) / traj._decision_scale_vector()
    bounds = [(-0.5, 0.5)] * x0.size
    traj._run_single_optimization(x0, q_main, dq_main, ddq_main, bounds)

    assert captured["jac"] == "2-point"
    assert captured["options"]["finite_diff_rel_step"] == pytest.approx(1e-6)


def test_slsqp_infeasible_final_point_falls_back_to_feasible_incumbent(monkeypatch):
    cfg = _make_cfg(use_analytical_bounds=False, max_iter=1)
    traj = ExcitedTrajectory(cfg, kinematics_func=lambda _q, _dq, _ddq: np.eye(2))
    q_main, dq_main, ddq_main = traj.main_trajectory.generate()
    x0 = np.zeros(2 * traj.num_joints * traj.num_harmonics)
    infeasible_x = np.full_like(x0, 10.0)

    def fake_minimize(**kwargs):
        objective_value = kwargs["fun"](infeasible_x)
        return SimpleNamespace(
            x=infeasible_x,
            success=False,
            status=9,
            message="Iteration limit reached",
            nit=1,
            nfev=1,
            njev=1,
            fun=objective_value,
        )

    monkeypatch.setattr(excited_module, "minimize", fake_minimize)
    bounds = [(-np.inf, np.inf)] * x0.size
    x_result, _, _, _, violation, solver = traj._run_single_optimization(
        x0, q_main, dq_main, ddq_main, bounds
    )

    np.testing.assert_array_equal(x_result, x0)
    assert violation == pytest.approx(0.0)
    assert solver["solver_returned_constraint_violation_ratio"] > 0.0
    assert solver["returned_feasible_incumbent"] is True
    assert solver["best_feasible_objective_value"] is not None


def test_analytical_mode_omits_direct_constraints():
    """Analytical mode with no q limits yields no direct kinematic constraints."""
    traj = ExcitedTrajectory(_make_cfg(use_analytical_bounds=True))
    cons = traj._build_q_constraints(lambda _x: (np.zeros((4, 2)),) * 3)
    assert cons == []


def _run_synthetic_restarts(
    objective_type,
    records,
    restart_callback=None,
    restart_state=None,
    n_restarts=None,
):
    cfg = _make_cfg(
        use_analytical_bounds=False,
        objective_type=objective_type,
        n_restarts=len(records) if n_restarts is None else n_restarts,
        early_stop_patience=100,
        target_condition_number=None,
    )
    traj = ExcitedTrajectory(
        cfg,
        restart_callback=restart_callback,
        restart_state=restart_state,
    )
    pending = list(records)

    def fake_run(self, *args, **kwargs):
        return pending.pop(0)

    traj._run_single_optimization = MethodType(fake_run, traj)
    traj._optimize()
    return traj


def test_all_failed_later_better_restart_is_selected(capsys):
    """An all-failed run still chooses the later lower-condition fallback."""
    x1 = np.zeros(8)
    x2 = np.ones(8)
    records = [
        (x1, 10.0, 10.0, 0.0, 0.0, {"success": False}),
        (x2, 5.0, 5.0, 0.0, 0.0, {"success": False}),
    ]

    traj = _run_synthetic_restarts("condition_number", records)
    output = capsys.readouterr().out
    assert traj.best_restart_index == 2
    assert traj.best_solver_success is False
    assert np.allclose(np.concatenate([traj.a.ravel(), traj.b.ravel()]), x2)
    assert "restart 2/2: Cond = 5.0000 (0.0s) *" in output


def test_iteration_limited_feasible_incumbent_beats_converged_worse_restart(capsys):
    """Termination status must not outrank feasibility and objective quality."""
    x1 = np.zeros(8)
    x2 = np.ones(8)
    records = [
        (x1, 100.0, 100.0, 0.0, 0.0, {"success": True}),
        (x2, 5.0, 5.0, 0.0, 0.0, {"success": False}),
    ]

    traj = _run_synthetic_restarts("condition_number", records)
    output = capsys.readouterr().out
    assert traj.best_restart_index == 2
    assert traj.best_solver_success is False
    assert np.allclose(np.concatenate([traj.a.ravel(), traj.b.ravel()]), x2)
    assert "restart 2/2: Cond = 5.0000 (0.0s) *" in output


def test_restart_callback_preserves_completed_candidates(capsys):
    """Each completed restart exposes recoverable final coefficients and best state."""
    x1 = np.zeros(8)
    x2 = np.ones(8)
    records = [
        (x1, 10.0, 10.0, 0.0, 0.0, {"success": True}),
        (x2, 5.0, 5.0, 0.0, 0.0, {"success": False}),
    ]
    checkpoints = []

    _run_synthetic_restarts("condition_number", records, restart_callback=checkpoints.append)
    capsys.readouterr()

    assert len(checkpoints) == 2
    assert checkpoints[0]["completed_restarts"] == 1
    assert len(checkpoints[0]["restarts"]) == 1
    assert checkpoints[1]["completed_restarts"] == 2
    assert checkpoints[1]["selected_restart_index"] == 2
    assert checkpoints[1]["selected_solver_success"] is False
    assert checkpoints[1]["best_objective_value"] == pytest.approx(5.0)
    assert checkpoints[1]["restarts"][1]["final_normalized_coefficients"] == x2.tolist()
    assert checkpoints[1]["restarts"][1]["final_coefficients"] == x2.tolist()


def test_restart_checkpoint_is_atomically_replaced_with_strict_json(tmp_path):
    checkpoint = tmp_path / "restart_checkpoint.json"

    _write_json_atomic(checkpoint, {"completed_restarts": 1, "coefficients": np.ones(2), "metric": np.inf})
    first = json.loads(checkpoint.read_text(encoding="utf-8"))
    _write_json_atomic(checkpoint, {"completed_restarts": 2, "coefficients": np.zeros(2)})
    second = json.loads(checkpoint.read_text(encoding="utf-8"))

    assert first == {"completed_restarts": 1, "coefficients": [1.0, 1.0], "metric": "Infinity"}
    assert second == {"completed_restarts": 2, "coefficients": [0.0, 0.0]}
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_restart_checkpoint_resumes_only_remaining_restarts_with_same_initial_points(capsys):
    def records():
        return [
            (np.zeros(8), 10.0, 10.0, 0.0, 0.0, {"success": True}),
            (np.ones(8), 5.0, 5.0, 0.0, 0.0, {"success": False}),
            (np.full(8, 2.0), 7.0, 7.0, 0.0, 0.0, {"success": True}),
        ]

    full_checkpoints = []
    full = _run_synthetic_restarts("condition_number", records(), restart_callback=full_checkpoints.append)
    capsys.readouterr()

    resume_checkpoints = []
    resumed = _run_synthetic_restarts(
        "condition_number",
        records()[1:],
        restart_callback=resume_checkpoints.append,
        restart_state=full_checkpoints[0],
        n_restarts=3,
    )
    output = capsys.readouterr().out

    assert "Resuming after 1/3 completed restarts" in output
    assert resumed.best_restart_index == full.best_restart_index == 2
    assert len(resumed.optimization_results) == 3
    assert [r["initial_normalized_coefficients"] for r in resumed.optimization_results] == [
        r["initial_normalized_coefficients"] for r in full.optimization_results
    ]
    assert resume_checkpoints[-1]["completed_restarts"] == 3
    assert resume_checkpoints[-1]["no_improve_count"] == 1


def test_restart_checkpoint_rejects_seed_mismatch(capsys):
    checkpoints = []
    record = [(np.zeros(8), 10.0, 10.0, 0.0, 0.0, {"success": True})]
    _run_synthetic_restarts("condition_number", record, restart_callback=checkpoints.append)
    capsys.readouterr()
    bad_state = dict(checkpoints[0])
    bad_state["seed"] = 999

    with pytest.raises(ValueError, match="seed does not match"):
        _run_synthetic_restarts(
            "condition_number",
            [],
            restart_state=bad_state,
            n_restarts=2,
        )


def test_d_optimal_selects_objective_not_condition_number(capsys):
    """D-optimal restarts are ranked by D-opt value even if cond is worse."""
    x1 = np.zeros(8)
    x2 = np.ones(8)
    records = [
        (x1, 2.0, 1.0, 0.0, 0.0, {"success": True}),
        (x2, 10.0, 0.5, 0.0, 0.0, {"success": True}),
    ]

    traj = _run_synthetic_restarts("d_optimal", records)
    output = capsys.readouterr().out
    assert traj.best_restart_index == 2
    assert traj.best_solver_success is True
    assert np.allclose(np.concatenate([traj.a.ravel(), traj.b.ravel()]), x2)
    assert "restart 2/2: Cond = 10.0000, D-opt = 0.5000 (0.0s) *" in output


def test_information_metrics_distinguish_regressor_and_information_matrix():
    """Regressor and Gram-matrix metrics use explicit, unambiguous keys."""
    metrics = _information_metrics(np.diag([2.0, 3.0]))

    assert metrics["raw_cond_F"] == pytest.approx(1.5)
    assert metrics["raw_cond_information"] == pytest.approx(2.25)
    assert metrics["sigma_min_F"] == pytest.approx(2.0)
    assert metrics["lambda_min_information"] == pytest.approx(4.0)
    assert metrics["column_scaled_cond_F"] == pytest.approx(1.0)
    assert metrics["column_scaled_cond_information"] == pytest.approx(1.0)
    assert "raw_F_cond" not in metrics
    assert "sigma_min_Y" not in metrics
    assert "lambda_min_F" not in metrics
    assert "raw_F_eigenvalues" not in metrics
    assert "Y_singular_values" not in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
