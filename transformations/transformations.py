from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from liegroups import SO3, SE3  # may be replaced with robotics toolbox for python
from mujoco._enums import mjtObj
from mujoco._functions import mj_name2id
from mujoco._structs import MjModel, MjData
from numpy.typing import NDArray


def tq2se3(t,
           q,
           ) -> SE3:
    rot = SO3.from_quaternion(q)
    return SE3(rot, t)


def tr2se3(t,
           r,
           ) -> SE3:
    rot = SO3.from_matrix(r)
    return SE3(rot, t)


def compose(trans: NDArray,
             rot: NDArray,
             ) -> Union[SE3, list[SE3]]:

    single_trans = False
    single_rot = False

    if 1 == trans.ndim:
        single_trans = True
        trans = np.expand_dims(trans, 0)

    if 1 == rot.ndim:
        single_rot = True
        rot = np.expand_dims(rot, 0)

    assert len(trans) == len(rot), "Numbers of vectors in 'trans' and 'rot' must match."

    poses = []
    for t, r in zip(trans, rot):
        if 4 == len(r):  # quaternion
            poses.append(tq2se3(t, r))
        elif 9 == len(r):  # rotation matrix
            poses.append(tr2se3(t, r.reshape(3, 3)))

    return poses[0] if single_trans and single_rot else poses


