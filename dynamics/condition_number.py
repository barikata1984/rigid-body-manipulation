import numpy as np
from liegroups.numpy import SE3
from mujoco._structs import MjData, MjModel
from numpy.typing import NDArray

from .dynamics import _setup_robot_dynamics_parameters, get_regressor_matrix


def _calculate_frame_dynamics(
    act_traj: NDArray,
    inverse_dynamics_partial_func,
    id_ll: int,
    pose_x_ll: SE3,
    pose_ll_llj: SE3,
    pose_x_sen: SE3,
) -> tuple[NDArray, NDArray, NDArray]:
    _, _, twists_lj_l, dtwists_lj_l = inverse_dynamics_partial_func(act_traj)
    twist_llj = twists_lj_l[id_ll]
    pose_sen_llj = pose_x_sen.inv().dot(pose_x_ll.dot(pose_ll_llj))
    twist_sen = pose_sen_llj.adjoint() @ twist_llj
    dtwist_llj = dtwists_lj_l[id_ll]
    pose_sen_llj_dadjoint = SE3.curlywedge(twist_sen) @ pose_sen_llj.adjoint()
    dtwist_sen = pose_sen_llj_dadjoint @ twist_llj + pose_sen_llj.adjoint() @ dtwist_llj

    regressor = get_regressor_matrix(twist_sen, dtwist_sen)
    return twist_sen, dtwist_sen, regressor


def calculate_condition_number(
    m: MjModel,
    d: MjData,
    joint_trajectory: NDArray,
    ee_body_name: str = "link6",
) -> float:
    """
    Calculates the condition number of the regressor matrix for a given trajectory.

    Args:
        m: MuJoCo MjModel object.
        d: MuJoCo MjData object.
        joint_trajectory:
            A numpy array of shape (n_frames, 3, n_dof) containing the joint positions, velocities, and accelerations.

    Returns:
        The condition number of the stacked regressor matrix.
    """
    poses, id_ll, pose_ll_llj, _, _, _, inverse_dynamics = _setup_robot_dynamics_parameters(m, d, ee_body_name)
    pose_x_ll = poses.x_b[id_ll]
    pose_x_sen = poses.get_x_("site", "target/ft_sensor")

    regressors = []
    for i in range(joint_trajectory.shape[0]):
        act_traj = joint_trajectory[i, :, :]
        _, _, regressor = _calculate_frame_dynamics(
            act_traj, inverse_dynamics, id_ll, pose_x_ll, pose_ll_llj, pose_x_sen
        )
        regressors.append(regressor)

    stacked_regressors = np.vstack(regressors)
    correlation_matrix = stacked_regressors.T @ stacked_regressors
    return np.linalg.cond(correlation_matrix)