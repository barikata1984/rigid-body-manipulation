from math import radians

import numpy as np
import pytest
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat

mjcf = pytest.importorskip("dm_control.mjcf")
MjModel = pytest.importorskip("mujoco._structs").MjModel


def test_mujoco_recovers_inertial_params_from_fullinertia():
    # register the root body and a body to which iprops are assigned
    cf = mjcf.RootElement()
    body = cf.worldbody.add("body", name="test_body")

    # body mass
    body_mass = 10
    # body com
    pos_obj_obji = np.array([0, 0, 0])
    # orientation of the body w.r.t its inertial frame
    rx = radians(14)
    ry = radians(24)
    rz = radians(36)
    # rotation from the body's inertial frame to the body frame
    # 'bodyi' and 'body' mean the inertial frame and the body frame
    rot_body_bodyi = euler2mat(rx, ry, rz, "rxyz")  # intrinsic XYZ euler
    # principal moments of inertia
    pixx = 0.04
    piyy = 0.05
    pizz = 0.06
    diaginertia = np.array([pixx, piyy, pizz])
    # expanded as a tensor
    diaginertia_tensor = np.diag(diaginertia)
    # transfer the reference frame to the body frame
    inertia_tensor = rot_body_bodyi @ diaginertia_tensor @ rot_body_bodyi.T
    ixx = inertia_tensor[0, 0]
    iyy = inertia_tensor[1, 1]
    izz = inertia_tensor[2, 2]
    ixy = inertia_tensor[0, 1]
    iyz = inertia_tensor[1, 2]
    izx = inertia_tensor[2, 0]
    fullinertia = np.array([ixx, iyy, izz, ixy, izx, iyz])
    # register the inertial params to the body
    body.add(
        "inertial",
        mass=body_mass,
        pos=pos_obj_obji,
        fullinertia=fullinertia,
    )

    # spawn model and get mujoco-computed data
    m = MjModel.from_xml_string(cf.to_xml_string())
    mj_mass = m.body_mass[1]
    mj_diaginertia = m.body_inertia[1]
    mj_iquat = m.body_iquat[1]

    assert np.isclose(mj_mass, body_mass)
    np.testing.assert_allclose(np.sort(mj_diaginertia), np.sort(diaginertia), atol=1e-10)
    np.testing.assert_allclose(quat2mat(mj_iquat), rot_body_bodyi, atol=1e-6)
