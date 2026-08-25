import numpy as np
import pytest
from transforms3d.euler import euler2mat

SE3 = pytest.importorskip("liegroups.numpy").SE3
SO3 = pytest.importorskip("liegroups.numpy").SO3
dynamics = pytest.importorskip("dynamics")

iparams_to_simat = dynamics.iparams_to_simat
simat_to_iparams = dynamics.simat_to_iparams
transfer_iparams = dynamics.transfer_iparams


def _iparams(mass, com, imat_com):
    """Pack [m, m*com, moments about the frame origin] from the inertia about the com."""
    com = np.asarray(com, dtype=float)
    imat = imat_com + mass * (com @ com * np.eye(3) - np.outer(com, com))
    return np.array([mass, *(mass * com), imat[0, 0], imat[1, 1], imat[2, 2], imat[0, 1], imat[1, 2], imat[2, 0]])


def test_simat_iparams_round_trip():
    rng = np.random.default_rng(0)
    rot = euler2mat(0.3, -0.7, 1.1, "sxyz")
    iparams = _iparams(2.5, rng.uniform(-0.3, 0.3, 3), rot @ np.diag([0.03, 0.05, 0.07]) @ rot.T)

    np.testing.assert_allclose(simat_to_iparams(iparams_to_simat(iparams)), iparams, atol=1e-12)


def test_transfer_iparams_matches_point_mass_relocation():
    # A point mass is fully described by its position, so transferring the frame must agree with
    # re-expressing that position in the target frame
    mass = 3.0
    pos_a = np.array([0.11, -0.24, 0.37])  # position w.r.t frame {a}
    iparams_a = _iparams(mass, pos_a, np.zeros((3, 3)))

    rot = euler2mat(0.4, 0.9, -0.2, "sxyz")
    trans = np.array([-0.15, 0.25, 0.05])
    pose_a_b = SE3(SO3(rot), trans)  # configuration of {b} w.r.t {a}

    pos_b = rot.T @ (pos_a - trans)
    expected = _iparams(mass, pos_b, np.zeros((3, 3)))

    np.testing.assert_allclose(transfer_iparams(pose_a_b.inv(), iparams_a), expected, atol=1e-12)


def test_transfer_iparams_preserves_kinetic_energy():
    rng = np.random.default_rng(1)
    rot_i = euler2mat(-0.6, 0.2, 0.8, "sxyz")
    iparams_a = _iparams(4.2, [0.2, -0.1, 0.3], rot_i @ np.diag([0.02, 0.06, 0.09]) @ rot_i.T)

    pose_a_b = SE3(SO3(euler2mat(1.2, -0.3, 0.5, "sxyz")), np.array([0.3, -0.4, 0.2]))
    iparams_b = transfer_iparams(pose_a_b.inv(), iparams_a)

    twist_b = rng.normal(size=6)
    twist_a = pose_a_b.adjoint() @ twist_b
    energy_a = twist_a @ iparams_to_simat(iparams_a) @ twist_a
    energy_b = twist_b @ iparams_to_simat(iparams_b) @ twist_b

    assert energy_a == pytest.approx(energy_b, rel=1e-12)


def test_transfer_iparams_flips_signs_under_180deg_z_rotation():
    # The ft sensor site is rotated by 180 deg about z w.r.t the object frame, which flips
    # mx, my, iyz, and izx while leaving the remaining parameters untouched
    rot_i = euler2mat(0.5, -0.4, 0.9, "sxyz")
    iparams = _iparams(1.7, [0.12, -0.31, 0.08], rot_i @ np.diag([0.04, 0.05, 0.06]) @ rot_i.T)

    pose = SE3(SO3(np.diag([-1.0, -1.0, 1.0])), np.zeros(3))  # self-inverse
    transferred = transfer_iparams(pose, iparams)

    signs = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1])
    np.testing.assert_allclose(transferred, signs * iparams, atol=1e-12)
