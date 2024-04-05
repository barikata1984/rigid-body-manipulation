import numpy as np
from transforms3d import affines
from liegroups import SO3, SE3  # may be replaced with robotics toolbox for python
from numpy.typing import NDArray


def trzs2SE3(t: NDArray,
             r: NDArray,
             z: NDArray = np.ones(3),
             s: NDArray = np.zeros(3),
             ) -> SE3:
    """Compose an SE3 object by applying affine.compose of transforms3d and
    SE3.from_matrix module of liegrops

    Args:
        t (np.ndarray): 3D translation vector.
        r (np.ndarray): 3x3 rotation matrix.
        z (np.ndarray, optional): 3D zoom vector. Defaults to np.ones(3).
        s (np.ndarray, optional): 3D shear vector. Defaults to np.zeros(3).

    Returns:
        liegroups.SE3: single SE3 entity
    """

    return SE3.from_matrix(affines.compose(t, r.reshape((3, 3)), z, s))


def tquat2SE3(t: NDArray,
              quat: NDArray,
              ) -> SE3:
    """Compose a SE3 object from a translation vector and a quaternion using
    the liegroups module.

    Args:
        t (np.ndarray): 3D translation vector.
        quat (np.ndarray): unit quaternion

    Returns:
        _type_: SE3 object
    """

    return SE3(SO3.from_quaternion(quat), t)


def posquat2SE3(p: NDArray,
                q: NDArray,
                ) -> SE3:
    """Compose a SE3 object from a translation vector and a quaternion using
    the liegroups module.

    Args:
        t (np.ndarray): 3D translation vector.
        quat (np.ndarray): unit quaternion

    Returns:
        SE3: instance of liegroups.SE3
    """
    return SE3(SO3.from_quaternion(q), p)


def posquat2SE3s(pos: NDArray,
                 quat: NDArray,
                 ) -> list[NDArray]:
    """Compose a SE3 object from a translation vector and a quaternion using
    the liegroups module following the shapes of input pos and quat.

    Args:
        t (np.ndarray): 3D translation vector.
        quat (np.ndarray): unit quaternion

    Returns:
        SE3: instance of liegroups.SE3
    """
    assert len(pos) == len(quat), "Length of 'pos' and 'quat' must be the same."

    return [posquat2SE3(p, q) for p, q in zip(pos, quat)]


