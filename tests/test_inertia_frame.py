import unittest
from math import pi

import mujoco
import numpy as np
from dm_control import mjcf
from mujoco._structs import MjModel
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat


class TestInertiaFrame(unittest.TestCase):
    def test_inertia_frame_from_fullinertia(self):
        """
        Tests if MuJoCo correctly recovers the principal inertia tensor and
        the inertial frame orientation when given a `fullinertia` tensor.
        """
        # --- 1. Define Inertial Properties in a Rotated Frame ---
        l, w, h = 0.1, 0.2, 0.4
        mass_density = 2700
        mass = mass_density * l * w * h
        pixx = mass * (w**2 + h**2) / 12
        piyy = mass * (l**2 + h**2) / 12
        pizz = mass * (l**2 + w**2) / 12
        diaginertia_gt = np.array([pixx, piyy, pizz])
        imat_bodyi_gt = np.diag(diaginertia_gt)

        s_rx, s_ry, s_rz = pi / 12, pi / 9, pi / 4
        rot_body_bodyi_gt = euler2mat(s_rx, s_ry, s_rz, "sxyz")

        imat_body_gt = rot_body_bodyi_gt @ imat_bodyi_gt @ rot_body_bodyi_gt.T
        ixx, iyy, izz = imat_body_gt[0, 0], imat_body_gt[1, 1], imat_body_gt[2, 2]
        ixy, ixz, iyz = imat_body_gt[0, 1], imat_body_gt[0, 2], imat_body_gt[1, 2]
        fullinertia = [ixx, iyy, izz, ixy, ixz, iyz]

        # --- 2. Setup MJCF Model ---
        mjcf_model = mjcf.RootElement()
        body = mjcf_model.worldbody.add("body", name="cuboid")
        body.add(
            "inertial",
            pos=[0, 0, 0],
            mass=mass,
            fullinertia=fullinertia,
        )

        # --- 3. Get MuJoCo Computed Data ---
        m = MjModel.from_xml_string(mjcf_model.to_xml_string())
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cuboid")

        diaginertia_mj = m.body_inertia[body_id]
        iquat_mj = m.body_iquat[body_id]

        # --- 4. Assertions ---
        np.testing.assert_allclose(np.sort(diaginertia_gt), np.sort(diaginertia_mj), atol=1e-9)

        rot_body_bodyi_mj = quat2mat(iquat_mj)
        imat_bodyi_mj = np.diag(diaginertia_mj)
        imat_body_mj = rot_body_bodyi_mj @ imat_bodyi_mj @ rot_body_bodyi_mj.T

        np.testing.assert_allclose(imat_body_gt, imat_body_mj, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
