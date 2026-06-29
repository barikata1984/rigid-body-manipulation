from math import pi

import numpy as np
import pytest
from numpy import linalg as la

liegroups_numpy = pytest.importorskip("liegroups.numpy")
SO3 = liegroups_numpy.SO3
SE3 = liegroups_numpy.SE3


def test_adjoint_transpose_equals_inverse_adjoint():
    rot = SO3.from_rpy(10 / 180 * pi, 20 / 180 * pi, 40 / 180 * pi)
    trans = np.array([1, 2, 3])
    pose = SE3(rot, trans)

    Ad = pose.adjoint()
    Ad_T = Ad.T
    Ad_inv = pose.inv().adjoint()
    pinv_Ad = la.pinv(pose.adjoint())

    assert np.allclose(Ad_T, Ad_inv)
    assert np.allclose(Ad_inv, pinv_Ad)
