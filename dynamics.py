import scipy
import numpy as np
import mujoco as mj
from liegroups import SO3, SE3
from numpy import linalg as la


def compose_sinert_i(mass, principal_inertia):
    return np.block([
        [mass * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), np.diag(principal_inertia)]])


def inverse(traj,
            SE3_home_ba,
            sinert_b,
            screw_bb,
            twist_00,
            dtwist_00,
            wrench_cc=np.zeros(6),
            SE3_tip_eefxr=SE3.identity()):
    # Forward iterations
    SE3_ba = []  # T_{ba} (ba = 10, 21, ..., 65), T_{i, i - 1} in MR
    twist_bb = np.atleast_2d(twist_00)  # First row set here is twist_00
    dtwist_bb = np.atleast_2d(dtwist_00)
    for idx, (SE3_h_ba, s_bb) in enumerate(zip(SE3_home_ba[1:], screw_bb)):
        # Index transitions:
        # ba = (1, 0), ..., (6, 5)
        # bb = (1, 1), ..., (6, 6)
        # idx = 0, 1, ..., 5, which corresponds to b = 1, 2, ..., 6
        SE3_ba.append(SE3.exp(-s_bb * traj[0, idx]).dot(SE3_h_ba))
        Ad_SE3_ba = SE3_ba[-1].adjoint()
        # =*=*=*=*=*=*=
        t_aa = twist_bb[-1]
        t_bb_1 = Ad_SE3_ba @ t_aa
        t_bb_2 = s_bb * traj[1, idx]
        t_bb = t_bb_1 + t_bb_2
        twist_bb = np.append(twist_bb, t_bb[np.newaxis, :], axis=0)
        # =*=*=*=*=*=*=
        dt_aa = dtwist_bb[-1]
        dt_bb_1 = Ad_SE3_ba @ dt_aa
        dt_bb_2 = SE3.curlywedge(t_bb) @ s_bb * traj[1, idx]
        dt_bb_3 = s_bb * traj[2, idx]
        dt_bb = dt_bb_1 + dt_bb_2 + dt_bb_3
        dtwist_bb = np.append(dtwist_bb, dt_bb[np.newaxis, :], axis=0)

    # Backward iterations
    wrench_bb = np.atleast_2d(wrench_cc)
    SE3_cb = SE3_ba + [SE3_tip_eefxr]
    reversed_data = [
        reversed(d) for d in (SE3_cb, sinert_b, twist_bb, dtwist_bb, screw_bb)]
    for SE3_cb, si_b, t_bb, dt_bb, s_bb in zip(*reversed_data):
        # Index transitions:
        # cb = 76, 65, 54, 43, 32, 21
        # bb = 66, 55, 44, 33, 22, 11
        w_cc = wrench_bb[-1]
        w_bb_1 = SE3_cb.adjoint().T @ w_cc
        w_bb_2 = si_b @ dt_bb
        w_bb_3 = - SE3.curlywedge(t_bb).T @ si_b @ t_bb
        w_bb = w_bb_1 + w_bb_2 + w_bb_3
        wrench_bb = np.append(wrench_bb, w_bb[np.newaxis, :], axis=0)

    wrench_bb = wrench_bb[::-1]
    ctrl_mat = wrench_bb[:-1] * screw_bb

    return np.sum(ctrl_mat, axis=0)
