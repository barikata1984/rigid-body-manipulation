from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np
from liegroups import SE3
from mujoco._functions import mjd_transitionFD
from mujoco._structs import MjModel, MjData
from numpy.typing import NDArray

from transformations import homogenize


@dataclass
class StateSpaceConfig:
    epsilon: float = 1e-8
    centered: bool = True


class StateSpace:
    def __init__(self,
                 cfg: StateSpaceConfig,
                 m: MjModel,
                 d: MjData,
                 ) -> None:
        self.epsilon = cfg.epsilon
        self.centered = cfg.centered

        self.ns = 2 * m.nv + m.na  # Number of dimensions of state space
        self.nsensordata = m.nsensordata  # Number of sensor ourputs

        self.A = np.zeros((self.ns, self.ns))  # State transition matrix
        self.B = np.zeros((self.ns, m.nu))  # Input2state matrix
        self.C = np.zeros((m.nsensordata, self.ns))  # State2output matrix
        self.D = np.zeros((m.nsensordata, m.nu))  # Input2output matrix

        # Populate the matrices
        self.update_matrices(m, d)

    def update_matrices(self,
                        m: MjModel,
                        d: MjData,
                        ) -> None:
        mjd_transitionFD(m, d, self.epsilon, self.centered, self.A, self.B, self.C, self.D)


def _get_simat(mass: float,
               diag_i: Union[list[float], NDArray],
               ) -> NDArray:
    imat = np.diag(diag_i)  # inertia matrix
    return np.block([[mass * np.eye(3), np.zeros((3, 3))],
                     [np.zeros((3, 3)), imat]])


def get_spatial_inertia_matrix(mass,
                               diagonal_inertia,
                               ):
    assert len(mass) == len(diagonal_inertia), "Lenght of 'mass' of the bodies and that of 'diagonal_inertia' vectors must match."
    return np.array([_get_simat(m, di) for m, di in zip(mass, diagonal_inertia)])


def transfer_simat(pose: Union[SE3, Sequence[SE3]],
                   simat: NDArray,
                   ) -> NDArray:
    single_pose = False
    single_simat = False

    # Add a batch dimension to handle a set of single pose and simat
    if isinstance(pose, SE3):
        # poses is an instance of liegroups.numpy.se3.SE3Matrix if this block hit
        single_pose = True
        pose = [pose]

    if 2 == simat.ndim:
        single_simat = True
        simat = np.expand_dims(simat, 0)

    assert len(pose) == len(simat), "The numbers of spatial inertia tensors and SE3 instances do not match."

    adjoint = [p.inv().adjoint() for p in pose]
    transfered = np.array([adj.T @ sim @ adj for adj, sim in zip(adjoint, simat)])

    return transfered[0] if single_pose and single_simat else transfered


def inverse(traj: np.ndarray,
            hposes_body_parent,
            simats_body: np.ndarray,
            uscrews_body: np.ndarray,
            twist_0: np.ndarray,
            dtwist_0: np.ndarray,
            wrench_tip: np.ndarray = np.zeros(6),
            pose_tip_ee: NDArray = SE3.identity(),
            ):

    # Prepare lie group, twist, and dtwist storage arrays
    poses = []  # T_{i, i - 1} in Modern Robotics
    twists = [twist_0]  # \mathcal{V}
    dtwists = [dtwist_0]  # \dot{\mathcal{V}}

    # Forward iterations
    for i, (h_p, us) in enumerate(zip(hposes_body_parent[1:], uscrews_body)):
        poses.append(SE3.exp(-1 * us * traj[0, i]).dot(h_p))  # Eq. 8.50
        # Compute twist (Eq. 8.51 in Modern Robotics)
        tw = poses[-1].adjoint() @ twists[-1] \
           + us * traj[1, i]
        # Compute the derivatife of twist (Eq. 8.52 in Modern Robotics)
        dtw = poses[-1].adjoint() @ dtwists[-1] \
            + SE3.curlywedge(tw) @ us * traj[1, i] \
            + us * traj[2, i]
        # Add the twist and its derivative to their storage arrays
        twists.append(tw)
        dtwists.append(dtw)

    # Backward iterations
    wrench = [wrench_tip]
    poses.append(pose_tip_ee)
    # Let m the # of joint/actuator axes, the backward iteration should be
    # performed from index m to 1. So, the range is set like below.
    for i in range(len(uscrews_body), 0, -1):
        # Compute wrench (Eq. 8.53 in Modern Robotics)
        w = poses[i].adjoint().T @ wrench[-1] \
          + simats_body[i] @ dtwists[i] \
          + -1 * SE3.curlywedge(twists[i]).T @ simats_body[i] @ twists[i]
        wrench.append(w)

    wrench.reverse()

    # A uscrew is a set of one-hot vectors, where the hot flag indicates the force
    # or torque element of the wrench that each screw corresponds to. Therefore,
    # the hadamarrd product below extracts the control signals, which are the
    # magnitude of target force or torque signals, from wrench array
    ctrl_mat = wrench[:-1] * uscrews_body  # Eq. 8.54

    return ctrl_mat.sum(axis=1), poses, twists, dtwists


def coordinate_transform_dtwist(pose, twist, dtwist, coord_xfer_twist=False):
    if coord_xfer_twist:  #  if twist is not coordinate transfered beforehand
        twist = pose.adjoint() @ twist

    # curlywedge() computes lie brackets
    return SE3.curlywedge(twist) @ twist + pose.adjoint() @ dtwist


def get_linear_velocity(twist, pose):
    """
    Refer to Modern Robotics Chapt. 8.2.1 with homogeneous coordinate representation
    """
    _linvel = SE3.wedge(twist) @ homogenize(pose.trans)
    return _linvel[:3]


def get_linear_acceleration(twist, dtwist, pose):
    """
    Refer to Modern Robotics Chapt. 8.2.1 with homogeneous coordinate representation
    """
    linvel = get_linear_velocity(twist, pose)
    _linacc = SE3.wedge(dtwist) @ homogenize(pose.trans) \
            + SE3.wedge(twist) @ homogenize(linvel, 0)

    return _linacc[:3]

