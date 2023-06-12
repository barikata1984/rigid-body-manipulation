import numpy as np
from utilities import store
from liegroups import SE3


def compose_sinert_i(mass, principal_inertia):
    return np.block([
        [mass * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.diag(principal_inertia)]])


def inverse(
        traj: np.ndarray,
        se3_home,
        sinert: np.ndarray,
        uscrew: np.ndarray,
        twist_0: np.ndarray,
        dtwist_0: np.ndarray,
        wrench_tip: np.ndarray = np.zeros(6),
        se3_tip_ee=SE3.identity()):

    # Prepare lie group, twist, and dtwist storage arrays
    se3 = []  # T_{i, i - 1} in Modern Robotics
    twist = np.atleast_2d(twist_0)
    dtwist = np.atleast_2d(dtwist_0)

    # Forward iterations
    for i, (se3_h, us) in enumerate(zip(se3_home[1:], uscrew)):
        se3.append(SE3.exp(-us * traj[0, i]).dot(se3_h))
        Ad_se3 = se3[-1].adjoint()
        # Compute twist
        prior_tw = twist[-1]
        tw_1 = Ad_se3 @ prior_tw
        tw_2 = us * traj[1, i]
        tw = tw_1 + tw_2
        # Compute the derivatife of twist
        prior_dtw = dtwist[-1]
        dtw_1 = Ad_se3 @ prior_dtw
        dtw_2 = us * traj[2, i]
        dtw_3 = SE3.curlywedge(tw) @ us * traj[1, i]
        dtw = dtw_1 + dtw_2 + dtw_3
        # Add the twist and its derivative to their storage arrays
        twist = store(tw, twist)
        dtwist = store(dtw, dtwist)

    # Backward iterations
    wrench = np.atleast_2d(wrench_tip)
    se3.append(se3_tip_ee)
    # Let m the # of joint/actuator axes, the backward iteration should be
    # performed from index m to 1. So, the range is set like below.
    for i in range(len(uscrew), 0, -1):
        prior_w = wrench[-1]
        w_1 = se3[i].adjoint().T @ prior_w
        w_2 = sinert[i] @ dtwist[i]
        w_3 = -SE3.curlywedge(twist[i]).T @ sinert[i] @ twist[i]
        w = w_1 + w_2 + w_3
        wrench = store(w, wrench)

    wrench = wrench[::-1]
    ctrl_mat = wrench[:-1] * uscrew

    return np.sum(ctrl_mat, axis=0)

