import numpy as np
from utilities import store
from liegroups import SE3


def compose_sinert_i(mass, principal_inertia):
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
    pose = []  # T_{i, i - 1} in Modern Robotics
    twist = np.atleast_2d(twist_0)
    dtwist = np.atleast_2d(dtwist_0)

    # Forward iterations
    for i, (p_h, us) in enumerate(zip(pose_home[1:], uscrew)):
        pose.append(SE3.exp(-us * traj[0, i]).dot(p_h))
        Ad_se3 = pose[-1].adjoint()
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
    pose.append(pose_tip_ee)
    # Let m the # of joint/actuator axes, the backward iteration should be
    # performed from index m to 1. So, the range is set like below.
    for i in range(len(uscrew), 0, -1):
        prior_w = wrench[-1]
        w_1 = pose[i].adjoint().T @ prior_w
        w_2 = body_spati[i] @ dtwist[i]
        w_3 = -SE3.curlywedge(twist[i]).T @ body_spati[i] @ twist[i]
        w = w_1 + w_2 + w_3
        wrench = store(w, wrench)

    wrench = wrench[::-1]
    ctrl_mat = wrench[:-1] * uscrew

    return np.sum(ctrl_mat, axis=0)

