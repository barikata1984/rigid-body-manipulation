from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np
from liegroups import SE3
from mujoco._functions import mjd_transitionFD
from mujoco._structs import MjModel, MjData
from numpy.typing import NDArray


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


def _compose_simat(mass: float,
                   diag_i: Union[list[float], NDArray],
                   ) -> NDArray:
    imat = np.diag(diag_i)  # inertia matrix
    return np.block([[mass * np.eye(3), np.zeros((3, 3))],
                     [np.zeros((3, 3)), imat]])


def compose_spatial_inertia_matrix(mass,
                                   diagonal_inertia,
                                   ):
    assert len(mass) == len(diagonal_inertia), "Lenght of 'mass' of the bodies and 'diagonal_inertia' vectors must match."
    return np.array([_compose_simat(m, di) for m, di in zip(mass, diagonal_inertia)])


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


def inverse(
        traj: np.ndarray,
        pose_home,
        body_spati: np.ndarray,
        uscrew: np.ndarray,
        twist_0: np.ndarray,
        dtwist_0: np.ndarray,
        wrench_tip: np.ndarray = np.zeros(6),
        pose_tip_ee: NDArray = SE3.identity()):

    # Prepare lie group, twist, and dtwist storage arrays
    poses = []  # T_{i, i - 1} in Modern Robotics
    twists = [twist_0]  # \mathcal{V}
    dtwists = [dtwist_0]  # \dot{\mathcal{V}}

    # Forward iterations
    for i, (p_h, us) in enumerate(zip(pose_home[1:], uscrew)):
        poses.append(SE3.exp(-us * traj[0, i]).dot(p_h))  # Eq. 8.50
        # Compute twist (Eq. 8.51 in Modern Robotics)
        tw  = poses[-1].adjoint() @ twists[-1]
        tw += us * traj[1, i]
        # Compute the derivatife of twist (Eq. 8.52 in Modern Robotics)
        dtw  = poses[-1].adjoint() @ dtwists[-1]
        dtw += SE3.curlywedge(tw) @ us * traj[1, i]
        dtw += us * traj[2, i]
        # Add the twist and its derivative to their storage arrays
        twists.append(tw)
        dtwists.append(dtw)

    # Backward iterations
    wrench = [wrench_tip]
    poses.append(pose_tip_ee)
    # Let m the # of joint/actuator axes, the backward iteration should be
    # performed from index m to 1. So, the range is set like below.
    for i in range(len(uscrew), 0, -1):
        w  = poses[i].adjoint().T @ wrench[-1]
        w += body_spati[i] @ dtwists[i]
        w += -1 * SE3.curlywedge(twists[i]).T @ body_spati[i] @ twists[i]
        wrench.append(w)

    wrench.reverse()

    # A uscrew is a set of one-hot vectors, where the hot flag indicates the force
    # or torque element of the wrench that each screw corresponds to. Therefore,
    # the hadamarrd product below extracts the control signals, which are the
    # magnitude of target force or torque signals, from wrench array
    ctrl_mat = wrench[:-1] * uscrew  # Eq. 8.53

    return ctrl_mat.sum(axis=1), poses, twists, dtwists


def compute_linvel(pose, twist, coord_xfer_twist=False):
    if coord_xfer_twist:  #  if twist is not coordinate transfered beforehand
        twist = pose.adjoint() @ twist

    htrans = np.array([*pose.trans, 1])  # convert into homogeneous coordinates
    hlinvel = SE3.wedge(twist) @ htrans

    return hlinvel[:3]


def coordinate_transform_dtwist(pose, twist, dtwist, coord_xfer_twist=False):
    if coord_xfer_twist:  #  if twist is not coordinate transfered beforehand
        twist = pose.adjoint() @ twist

    # curlywedge() computes lie brackets
    return SE3.curlywedge(twist) @ twist + pose.adjoint() @ dtwist


def compute_linacc(pose, twist, dtwist, coord_xfer_tdtwist=False):
    if coord_xfer_tdtwist:  #  if twist is not coordinate transfered beforehand
        twist = pose.adjoint() @ twist
        dtwist = coordinate_transform_dtwist(pose, twist, dtwist, coord_xfer_twist=False)

    htrans = np.array([*pose.trans, 1])
    linvel = compute_linvel(pose, twist)

    return (SE3.wedge(dtwist) @ htrans)[:3] + np.cross(twist[3:], linvel)

