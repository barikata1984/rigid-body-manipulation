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
    Q = np.eye(ss.ns)  # State cost matrix
    R = np.diag(ss.input_weights)  # Input cost matrix

    # Compute the feedback gain matrix K
    P = linalg.solve_discrete_are(ss.A, ss.B, Q, R)
    K = linalg.pinv(R + ss.B.T @ P @ ss.B) @ ss.B.T @ P @ ss.A

    return K


def _compose_simat(mass, diag_i):
    imat = np.diag(diag_i)  # inertia matrix
    return np.block([[mass * np.eye(3), np.zeros((3, 3))],
                     [np.zeros((3, 3)), imat]])


def compose_spatial_inertia_matrix(mass, diagonal_inertia):
    assert len(mass) == len(diagonal_inertia), "Lenght of 'mass' of the bodies and 'diagonal_inertia' vectors must match."
    return np.array([_compose_simat(m, di) for m, di in zip(mass, diagonal_inertia)])


def transfer_sinert(pose, spatial_inertia_matrices):
    assert len(pose) == len(spatial_inertia_matrices), "Lenght of 'pose' and 'spatial_inertia_matrices' must match."

    pose_adjoint = [p.inv().adjoint() for p in pose]
    transfered = [adj.T @ simat @ adj for adj, simat in zip(pose_adjoint, spatial_inertia_matrices)]

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
    pose = []  # T_{i, i - 1} in Modern Robotics
    twist = [twist_0]
    dtwist = [dtwist_0]

    # Forward iterations
    for i, (p_h, us) in enumerate(zip(pose_home[1:], uscrew)):
        pose.append(SE3.exp(-us * traj[0, i]).dot(p_h))
        Ad_se3 = pose[-1].adjoint()
        # Compute twist
        prior_tw = twist[-1]
        tw = Ad_se3 @ prior_tw
        tw += us * traj[1, i]
        # Compute the derivatife of twist
        prior_dtw = dtwist[-1]
        dtw = Ad_se3 @ prior_dtw
        dtw += us * traj[2, i]
        dtw += SE3.curlywedge(tw) @ us * traj[1, i]
        # Add the twist and its derivative to their storage arrays
        twist.append(tw)
        dtwist.append(dtw)

    # Backward iterations
    wrench = [wrench_tip]
    pose.append(pose_tip_ee)

    # Let m the # of joint/actuator axes, the backward iteration should be
    # performed from index m to 1. So, the range is set like below.
    for i in range(len(uscrew), 0, -1):
        prior_w = wrench[-1]
        w = pose[i].adjoint().T @ prior_w
        w += body_spati[i] @ dtwist[i]
        w += -SE3.curlywedge(twist[i]).T @ body_spati[i] @ twist[i]
        wrench.append(w)

    wrench = wrench[::-1]
    ctrl_mat = wrench[:-1] * uscrew

    return np.sum(ctrl_mat, axis=0)

