from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

import numpy as np
#import mujoco as mj
#from attrdict import AttrDict
from liegroups import SE3
from mujoco._functions import mjd_transitionFD
from mujoco._structs import MjModel, MjData
from numpy.typing import NDArray
from scipy import linalg


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
        self.B = np.zeros((self.ns, m.nu))  # Input state matrix
        self.C = np.zeros((m.nsensordata, self.ns))  # State output matrix
        self.D = np.zeros((m.nsensordata, m.nu))  # Input output matrix

        # Populate the matrices
        self.update_matrices(m, d)

    def update_matrices(self,
                        m: MjModel,
                        d: MjData,
                        ) -> None:
        mjd_transitionFD(m, d, self.epsilon, self.centered,
                         self.A, self.B, self.C, self.D)



def _compose_simat(mass: float,
                   diag_i: Union[list[float], NDArray],
                   ) -> NDArray:
    imat = np.diag(diag_i)  # inertia matrix
    return np.block([[mass * np.eye(3), np.zeros((3, 3))],
                     [np.zeros((3, 3)), imat]])


def compose_spatial_inertia_matrices(mass,
                                     diagonal_inertia,
                                     ):
    assert len(mass) == len(diagonal_inertia), "Lenght of 'mass' of the bodies and 'diagonal_inertia' vectors must match."
    return np.array([_compose_simat(m, di) for m, di in zip(mass, diagonal_inertia)])


def transfer_simats(poses,  # TODO: annotate later...
                    simats,  # TODO: annotate later...
                    ) -> NDArray:
    poses_is_iterable = True

    # Add a batch dimension to handle a set of single pose and simat
    if not isinstance(poses, Iterable):
        # poses is an instance of liegroups.numpy.se3.SE3Matrix if this block hit
        poses_is_iterable = False
        poses = [poses]

    if 2 == simats.ndim:
        simats = [simats]

    assert len(poses) == len(simats), "The numbers of spatial inertia tensors and SE3 instances do not match."

    adjoints = [p.inv().adjoint() for p in poses]
    transfered = np.array([adj.T @ sim @ adj for adj, sim in zip(adjoints, simats)])

    return transfered if poses_is_iterable else transfered[0]


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
    twists = [twist_0]
    dtwists = [dtwist_0]

    # Forward iterations
    for i, (p_h, us) in enumerate(zip(pose_home[1:], uscrew)):
        poses.append(SE3.exp(-us * traj[0, i]).dot(p_h))
        Ad_se3 = poses[-1].adjoint()
        # Compute twist
        prior_tw = twists[-1]
        tw = Ad_se3 @ prior_tw
        tw += us * traj[1, i]
        # Compute the derivatife of twist
        prior_dtw = dtwists[-1]
        dtw = Ad_se3 @ prior_dtw
        dtw += us * traj[2, i]
        dtw += SE3.curlywedge(tw) @ us * traj[1, i]
        # Add the twist and its derivative to their storage arrays
        twists.append(tw)
        dtwists.append(dtw)

    # Backward iterations
    wrench = [wrench_tip]
    poses.append(pose_tip_ee)
    # Let m the # of joint/actuator axes, the backward iteration should be
    # performed from index m to 1. So, the range is set like below.
    for i in range(len(uscrew), 0, -1):
        prior_w = wrench[-1]
        w = poses[i].adjoint().T @ prior_w
        w += body_spati[i] @ dtwists[i]
        w += -SE3.curlywedge(twists[i]).T @ body_spati[i] @ twists[i]
        wrench.append(w)

    wrench = wrench[::-1]
    ctrl_mat = wrench[:-1] * uscrew

    return np.sum(ctrl_mat, axis=0), poses, twists, dtwists


def compute_linvel(pose, twist, coord_xfer_twist=False):
    if coord_xfer_twist:  #  if twist is not coordinate transfered beforehand
        twist = pose.adjoint() @ twist

    htrans = np.array([*pose.trans, 1])  # convert into homogeneous coordinates
    hlinvel = SE3.wedge(twist) @ htrans

    return hlinvel[:3]


def coordinate_transform_dtwist(pose, twist, dtwist, coord_xfer_twist=False):
    if coord_xfer_twist:  #  if twist is not coordinate transfered beforehand
        twist = pose.adjoint() @ twist

    return SE3.curlywedge(twist) @ twist + pose.adjoint() @ dtwist


def compute_linacc(pose, twist, dtwist, coord_xfer_tdtwist=False):
    if coord_xfer_tdtwist:  #  if twist is not coordinate transfered beforehand
        twist = pose.adjoint() @ twist
        dtwist = coordinate_transform_dtwist(pose, twist, dtwist, coord_xfer_twist=False)

    htrans = np.array([*pose.trans, 1])
    linvel = compute_linvel(pose, twist)

    return (SE3.wedge(dtwist) @ htrans)[:3] + np.cross(twist[3:], linvel)

