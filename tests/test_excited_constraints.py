import numpy as np
import pytest

from trajectories.excited import ExcitedTrajectory, ExcitedTrajectoryConfig


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

    # q_min/q_max default to None here, so only dq/ddq constraints are present.
    # Each dq/ddq constraint returns the full per-sample, per-joint margin vector.
    cons_within = traj._build_q_constraints(within)
    cons_viol = traj._build_q_constraints(violating)
    assert len(cons_within) == 2
    assert all(np.all(c["fun"](np.zeros(8)) >= 0) for c in cons_within)
    assert all(np.all(c["fun"](np.zeros(8)) < 0) for c in cons_viol)


def test_analytical_mode_omits_direct_constraints():
    """Analytical mode with no q limits yields no direct kinematic constraints."""
    traj = ExcitedTrajectory(_make_cfg(use_analytical_bounds=True))
    cons = traj._build_q_constraints(lambda _x: (np.zeros((4, 2)),) * 3)
    assert cons == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
