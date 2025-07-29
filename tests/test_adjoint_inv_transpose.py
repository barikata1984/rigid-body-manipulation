import unittest
from math import pi

import numpy as np
from liegroups.numpy import SE3, SO3


class TestAdjoint(unittest.TestCase):
    def test_adjoint_inverse_transpose_relationship(self):
        """
        Tests the relationship Ad(T^-1) == Ad(T)^T.
        """
        rot = SO3.from_rpy(10 / 180 * pi, 20 / 180 * pi, 40 / 180 * pi)
        trans = np.array([1, 2, 3])
        pose = SE3(rot, trans)

        # Adjoint of the inverse pose
        Ad_inv_pose = pose.inv().adjoint()

        # Inverse of the original adjoint
        Ad_inv_from_adj = np.linalg.inv(pose.adjoint())

        # The relationship Ad(T^-1) == Ad(T)^-1 should hold
        np.testing.assert_allclose(Ad_inv_pose, Ad_inv_from_adj, atol=1e-15)

    def test_adjoint_inverse_vs_pseudoinverse(self):
        """
        Tests if the adjoint of the inverse is equivalent to the pseudoinverse.
        For SE(3), the adjoint is always invertible, so inv == pinv.
        """
        rot = SO3.from_rpy(10 / 180 * pi, 20 / 180 * pi, 40 / 180 * pi)
        trans = np.array([1, 2, 3])
        pose = SE3(rot, trans)

        # Adjoint of the inverse pose
        Ad_inv = pose.inv().adjoint()

        # Pseudoinverse of the original adjoint
        pinv_Ad = np.linalg.pinv(pose.adjoint())

        # They should be equal
        np.testing.assert_allclose(Ad_inv, pinv_Ad, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
