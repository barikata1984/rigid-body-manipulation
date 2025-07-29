import unittest
from math import radians

import mujoco
import numpy as np
from dm_control import mjcf
from mujoco._structs import MjModel
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat


class TestInertia(unittest.TestCase):
    def test_mujoco_inertia_calculation(self):
        """
        Tests if MuJoCo correctly calculates and decomposes inertial properties
        from a `fullinertia` specification.
        """
        # --- 1. Define Ground Truth Inertial Properties ---
        cf = mjcf.RootElement()
        body = cf.worldbody.add("body", name="test_body")

        body_mass = 10
        pos_obj_obji = np.array([0, 0, 0])
        rx, ry, rz = radians(14), radians(24), radians(36)
        rot_body_bodyi_gt = euler2mat(rx, ry, rz, "rxyz")

        diaginertia_gt = np.array([0.04, 0.05, 0.06])
        diaginertia_tensor_gt = np.diag(diaginertia_gt)

        inertia_tensor_gt = rot_body_bodyi_gt @ diaginertia_tensor_gt @ rot_body_bodyi_gt.T
        ixx, iyy, izz = inertia_tensor_gt[0, 0], inertia_tensor_gt[1, 1], inertia_tensor_gt[2, 2]
        ixy, ixz, iyz = inertia_tensor_gt[0, 1], inertia_tensor_gt[0, 2], inertia_tensor_gt[1, 2]
        fullinertia = np.array([ixx, iyy, izz, ixy, ixz, iyz])

        body.add(
            "inertial",
            mass=body_mass,
            pos=pos_obj_obji,
            fullinertia=fullinertia,
        )

        # --- 2. Spawn MuJoCo Model and Get Computed Data ---
        m = MjModel.from_xml_string(cf.to_xml_string())
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "test_body")

        mj_mass = m.body_mass[body_id]
        mj_diaginertia = m.body_inertia[body_id]
        mj_iquat = m.body_iquat[body_id]

        # --- 3. Assertions ---
        self.assertAlmostEqual(body_mass, mj_mass, places=9)

        # Compare principal moments of inertia (sorted, as order can change)
        np.testing.assert_allclose(np.sort(diaginertia_gt), np.sort(mj_diaginertia), atol=1e-9)

        # Compare the resulting inertia tensors
        rot_body_bodyi_mj = quat2mat(mj_iquat)
        diaginertia_tensor_mj = np.diag(mj_diaginertia)
        inertia_tensor_mj = rot_body_bodyi_mj @ diaginertia_tensor_mj @ rot_body_bodyi_mj.T

        np.testing.assert_allclose(inertia_tensor_gt, inertia_tensor_mj, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
