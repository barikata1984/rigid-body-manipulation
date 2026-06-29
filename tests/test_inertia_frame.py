from math import pi

import numpy as np
import pytest
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat

mjcf = pytest.importorskip("dm_control.mjcf")
MjModel = pytest.importorskip("mujoco._structs").MjModel


def test_mujoco_recovers_principal_inertia_frame_from_fullinertia():
    # Principal inertia tensor of a cuboid
    l = 0.1  # side length along X
    w = 0.2  # side length along Y
    h = 0.4  # side length along Z
    volume = l * h * w
    mass_density = 2700
    mass = mass_density * volume
    pixx = mass * (pow(w, 2) + pow(h, 2)) / 12
    piyy = mass * (pow(l, 2) + pow(h, 2)) / 12
    pizz = mass * (pow(l, 2) + pow(w, 2)) / 12
    diaginertia = np.array([pixx, piyy, pizz])
    imat_bodyi = np.diag(diaginertia)

    # New object orientation represented as static XYZ-euler angles
    s_rx = 15 / 180 * pi
    s_ry = 20 / 180 * pi
    s_rz = 45 / 180 * pi
    # New orientation w.r.t the principal inertia frame
    rot_bodyi_new = euler2mat(s_rx, s_ry, s_rz, "sxyz")
    imat_new = rot_bodyi_new.T @ imat_bodyi @ rot_bodyi_new

    ixx = imat_new[0, 0]
    iyy = imat_new[1, 1]
    izz = imat_new[2, 2]
    ixy = imat_new[0, 1]
    iyz = imat_new[1, 2]
    izx = imat_new[2, 0]

    # Pattern 2: Set fullinertia; MuJoCo should recover the principal moments
    mjcf_model_2 = mjcf.RootElement()
    body = mjcf_model_2.worldbody.add("body", name="cuboid")
    body.add(
        "inertial",
        pos=[0, 0, 0],
        mass=mass,
        fullinertia=[ixx, iyy, izz, ixy, izx, iyz],
    )

    m2 = MjModel.from_xml_string(mjcf_model_2.to_xml_string())

    # MuJoCo body index 1 is the cuboid (index 0 is worldbody)
    np.testing.assert_allclose(np.sort(m2.body_inertia[1]), np.sort(diaginertia), atol=1e-10)
    np.testing.assert_allclose(quat2mat(m2.body_iquat[1]).T, rot_bodyi_new, atol=1e-6)
