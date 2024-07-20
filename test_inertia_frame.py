from math import pi

import numpy as np
from dm_control import mjcf
from mujoco._structs import MjModel
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat


# Principal inertia tensor of a ellipsoid
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
print(f"principal inertia tensor:\n{imat_bodyi}")

# New object orientation representated as static XYZ-euler
s_rx = 15 / 180 * pi
s_ry = 20 / 180 * pi
s_rz = 45 / 180 * pi
# New rientation w.r.t the principal inertia frame
rot_bodyi_new = euler2mat(s_rx, s_ry, s_rz, "sxyz")
imat_new = rot_bodyi_new.T @ imat_bodyi @ rot_bodyi_new
print(f"rotated inertia tensor:\n{imat_new}")

ixx = imat_new[0, 0]
iyy = imat_new[1, 1]
izz = imat_new[2, 2]
ixy = imat_new[0, 1]
iyz = imat_new[1, 2]
izx = imat_new[2, 0]

mjcf_model_1 = mjcf.RootElement()
mjcf_model_1.worldbody.add("body", name="cuboid")
print(f"{dir(mjcf_model_1.worldbody.body[0].inertial)=}")
#                                   pos=[0, 0, 0],
#                                   mass=mass,
#                                   diaginertia=diaginertia,
#                                   quat=mat2quat(rot_bodyi_new),
#                                   #fullinertia=[ixx, iyy, izz, ixy, izx, iyz],
#                                   )


m1 = MjModel.from_xml_string(mjcf_model_1.to_xml_string())
print(f"Original rot_x_new in quat: {mat2quat(rot_bodyi_new)}")
print(f"Original rot_new_x in quat: {mat2quat(rot_bodyi_new.T)}")
print(f"===============================================")
print(f"Pattern 1: Set 'diaginertia'")
print(f"Recoverd rot_x_new in quat: {m1.body_iquat[0]}")
print(f"{m1.body_inertia[0]=}")
print(f"- - - - - - - - - - - - - - - - - - - - - - - -")


mjcf_model_2 = mjcf.RootElement()
body = mjcf_model_2.worldbody.add("body", name="cuboid")
mjcf_model_1.worldbody.body[0].add("inertial",
                                   pos=[0, 0, 0],
                                   mass=mass,
                                   #diaginertia=diaginertia,
                                   #quat=mat2quat(rot_bodyi_new),
                                   fullinertia=[ixx, iyy, izz, ixy, izx, iyz],
                                   )


m2 = MjModel.from_xml_string(mjcf_model_2.to_xml_string())
print(f"Pattern 2: 'fullinertia'")
print(f"Recoverd rot_x_new in quat: {m2.body_iquat[0]}")
