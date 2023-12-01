import numpy as np
import mujoco as mj
from liegroups import SE3
from attrdict import AttrDict
from scipy import linalg


class StateSpace:
    def __init__(self, m: mj.MjModel, d: mj.MjData, config: AttrDict):
        self.nv = m.nv  # Number of degree of freedom
        self.na = m.na  # Number of activations
        self.nu = m.nu  # Number of inputs
        self.ns = 2 * self.nv + self.na  # Number of dimensions of state space
        self.nsensordata = m.nsensordata  # Number of sensor ourputs

        self.A = np.zeros((self.ns, self.ns))  # State transition matrix
        self.B = np.zeros((self.ns, self.nu))  # Input state matrix
        self.C = np.zeros((self.nsensordata, self.ns))  # State output matrix
        self.D = np.zeros((self.nsensordata, self.nu))  # input output matrix

        self.eps = config.lqr.epsilon
        self.flg_centered = config.lqr.centered

        self.input_weights = config.lqr.input_weights

        mj.mjd_transitionFD(
            m, d, self.eps, self.flg_centered, self.A, self.B, self.C, self.D)


def compute_gain_matrix(m, d, ss: StateSpace):
    mj.mjd_transitionFD(m, d, ss.eps, ss.flg_centered, ss.A, ss.B, ss.C, ss.D)

    Q = np.eye(ss.ns)  # State cost matrix
    R = np.diag(ss.input_weights)  # Input cost matrix

    # Compute the feedback gain matrix K
    P = linalg.solve_discrete_are(ss.A, ss.B, Q, R)
    K = linalg.pinv(R + ss.B.T @ P @ ss.B) @ ss.B.T @ P @ ss.A

    return K


def compose_spati_i(mass, principal_inertia):
    return np.block([
        [mass * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.diag(principal_inertia)]])


def transfer_sinert(pose, spati):
    assert len(pose) == len(spati), "The numbers of spatial inertia tensors and SE3 instances do not match."

    pose_adjoint = [p.inv().adjoint() for p in pose]
    transfered = [adj.T @ si @ adj for adj, si in zip(pose_adjoint, spati)]

    return np.array(transfered)


def inverse(
        traj: np.ndarray,
        pose_home,
        body_spati: np.ndarray,
        uscrew: np.ndarray,
        twist_0: np.ndarray,
        dtwist_0: np.ndarray,
        wrench_tip: np.ndarray = np.zeros(6),
        pose_tip_ee=SE3.identity()):

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

