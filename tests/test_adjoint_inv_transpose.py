from math import pi

import numpy as np
import pytest
from numpy import linalg as la

liegroups_numpy = pytest.importorskip("liegroups.numpy")
SO3 = liegroups_numpy.SO3
SE3 = liegroups_numpy.SE3


def test_inverse_adjoint_equals_adjoint_of_inverse():
    rot = SO3.from_rpy(10 / 180 * pi, 20 / 180 * pi, 40 / 180 * pi)
    trans = np.array([1, 2, 3])
    pose = SE3(rot, trans)

    Ad = pose.adjoint()
    Ad_inv = pose.inv().adjoint()
    pinv_Ad = la.pinv(Ad)

    assert np.allclose(Ad @ Ad_inv, np.eye(6))
    assert np.allclose(Ad_inv, pinv_Ad)
