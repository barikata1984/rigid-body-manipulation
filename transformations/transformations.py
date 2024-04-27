from typing import Optional, Union

import numpy as np
from liegroups import SO3, SE3
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
             rot: Optional[NDArray] = None,
             ) -> Union[SE3, list[SE3]]:

    if rot is None:
        id_quat = [1, 0, 0, 0]
        rot = np.array([id_quat for _ in trans])

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


def homogenize(coord, forth_val=1):
    homog = forth_val * np.ones(4)
    homog[:3] = coord
    return homog
