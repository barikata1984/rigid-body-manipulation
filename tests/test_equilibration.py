import numpy as np
import pytest

from trajectories.base_trajectory import BaseTrajectory


def _kin_factory(F: np.ndarray):
    """Return a kinematics_func replaying rows of F, indexed by traj_q[i][0]."""

    def kin(qi, _dqi, _ddqi):
        return F[int(qi[0])].reshape(1, -1)

    return kin


def _run(F: np.ndarray, column_scale: bool) -> float:
    n = F.shape[0]
    q = np.arange(n, dtype=np.float64).reshape(n, 1)
    return BaseTrajectory.compute_condition_number(np.arange(n), _kin_factory(F), q, q, q, column_scale=column_scale)


def test_equilibration_is_invariant_to_column_scaling():
    """Rescaling regressor columns (unit changes) leaves the equilibrated cond fixed."""
    rng = np.random.default_rng(0)
    F = rng.standard_normal((30, 4))
    F_scaled = F * np.array([1.0, 1e3, 1e-2, 10.0])

    cond = _run(F, column_scale=True)
    cond_scaled = _run(F_scaled, column_scale=True)
    assert cond_scaled == pytest.approx(cond, rel=1e-6)


def test_raw_condition_number_depends_on_column_scaling():
    """Without equilibration the same unit change moves the condition number."""
    rng = np.random.default_rng(1)
    F = rng.standard_normal((30, 4))
    F_scaled = F * np.array([1.0, 1e3, 1e-2, 10.0])

    cond = _run(F, column_scale=False)
    cond_scaled = _run(F_scaled, column_scale=False)
    assert cond_scaled != pytest.approx(cond, rel=1e-2)


def test_equilibrate_unit_diagonal():
    """Column-equilibrated Y has a unit diagonal by construction."""
    rng = np.random.default_rng(2)
    F = rng.standard_normal((30, 4)) * np.array([1.0, 100.0, 0.1, 5.0])
    Y = F.T @ F
    Y_eq = BaseTrajectory._equilibrate(Y, column_scale=True)
    assert np.allclose(np.diag(Y_eq), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
