from math import radians

import numpy as np
import pandas as pd
from dm_control import mjcf
from mujoco._structs import MjModel
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat


# register the root body and a body to which iprops are assigned
cf =  mjcf.RootElement()
body = cf.worldbody.add("body", name="test_body")

# body mass
body_mass = 10
# body com
pos_obj_obji = np.array([0, 0, 0])
# orientation of the body w.r.t its inertial frame and its inverse
rx = radians(14)
ry = radians(24)
rz = radians(36)
# rotation from the body's inertial frame to the body frame
# 'bodyi' and 'body' mean the inertial frame and the body frame
rot_body_bodyi = euler2mat(rx, ry, rz, "rxyz")  # intrincic XYZ euler
# quartanion of the inverse
quat_body_bodyi = mat2quat(rot_body_bodyi)
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
body.add("inertial",
         mass=body_mass,
         pos=pos_obj_obji,
         fullinertia=fullinertia,
         )

# spawn model and get mujoco-computed data
m = MjModel.from_xml_string(cf.to_xml_string())
mj_mass = m.body_mass[1]
mj_diaginertia = m.body_inertia[1]
mj_iquat = m.body_iquat[1]
# compare the orignal and mujoco-computed data
cad_gt = [body_mass, *diaginertia, *quat_body_bodyi]
mjc = [mj_mass, *mj_diaginertia, *mj_iquat]
# principal moments of inertia and the inertial frame's orientation w.r.t the body frame
index = ["total_mass", "pixx", "piyy", "pizz", "real", "i", "j", "k"]
print("mass, diaginertia, quat ------------------------------------------------------")
print(pd.DataFrame({"original": cad_gt,
                    "mujoco": mjc,
                    },
                   index=index,
                   ).transpose()
      )
print("")
print("rot_body_bodyi ---------------------------------------------------------------")
print(f"original:\n{rot_body_bodyi}")
print(f"mujoco:\n{quat2mat(mj_iquat)}")
