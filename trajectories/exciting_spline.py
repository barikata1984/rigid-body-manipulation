from datetime import datetime

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.signal.windows import tukey

from dynamics.condition_number import calculate_condition_number
from trajectories.excitation import generate_sinusoidal_trajectory
from trajectories.spline_interpolation import (
    BoundaryCondition,
    generate_spline_trajectory,
)


def _windowed_generate_sinusoidal_trajectory(
    duration: float,
    fps: int,
    coeffs: np.ndarray,
    base_frequency: float,
    window_func=tukey,
    alpha=0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a sinusoidal trajectory multiplied by a window function.
    Derivatives are calculated numerically from the resulting position trajectory
    to ensure boundary conditions are met.
    """
    n_frames = int(duration * fps)

    # Generate the core sinusoidal trajectory (without any offset)
    t_vec, s_pos, _, _, _ = generate_sinusoidal_trajectory(
        duration=duration,
        fps=fps,
        coeffs=coeffs,
        base_frequency=base_frequency,
        jointpos_offset=np.zeros(coeffs.shape[0]),  # No offset for the excitation part
    )

    # Generate window function
    window = window_func(n_frames, alpha=alpha)

    # Apply window to the position trajectory
    qpos = window * s_pos

    # Numerically differentiate the windowed position trajectory to get velocity, acceleration, and jerk
    dt = 1.0 / fps
    qvel = np.gradient(qpos, dt, axis=1, edge_order=2)
    qacc = np.gradient(qvel, dt, axis=1, edge_order=2)
    qjerk = np.gradient(qacc, dt, axis=1, edge_order=2)

    # Force boundary derivatives to zero to ensure smooth connection
    qvel[:, 0] = 0.0
    qvel[:, -1] = 0.0
    qacc[:, 0] = 0.0
    qacc[:, -1] = 0.0
    qjerk[:, 0] = 0.0
    qjerk[:, -1] = 0.0

    return t_vec, qpos, qvel, qacc, qjerk


def _generate_combined_trajectory(
    coeffs: np.ndarray,
    q_base_pos: np.ndarray,
    q_base_vel: np.ndarray,
    q_base_acc: np.ndarray,
    q_base_jerk: np.ndarray,
    duration: float,
    fps: int,
    base_frequency: float,
) -> dict:
    """Helper function to generate and combine base and excitation trajectories."""
    # Generate the windowed excitation trajectory
    t, exc_pos, exc_vel, exc_acc, exc_jerk = _windowed_generate_sinusoidal_trajectory(
        duration, fps, coeffs, base_frequency, window_func=tukey, alpha=0.2
    )

    # Combine base trajectory with excitation trajectory
    full_qpos = q_base_pos + exc_pos
    full_qvel = q_base_vel + exc_vel
    full_qacc = q_base_acc + exc_acc
    full_qjerk = q_base_jerk + exc_jerk

    return {"t": t, "qpos": full_qpos, "qvel": full_qvel, "qacc": full_qacc, "qjerk": full_qjerk}


def _exciting_spline_objective(coeffs_flat: np.ndarray, *opt_args) -> float:
    """Objective function for task-oriented optimization. Calculates condition number."""
    (
        q_base_pos,
        q_base_vel,
        q_base_acc,
        q_base_jerk,
        m,
        d,
        duration,
        fps,
        base_frequency,
        ee_body_name,
    ) = opt_args
    n_joints = m.njnt
    n_harmonics = coeffs_flat.shape[0] // (n_joints * 2)
    coeffs = coeffs_flat.reshape(n_joints, n_harmonics, 2)

    traj = _generate_combined_trajectory(
        coeffs, q_base_pos, q_base_vel, q_base_acc, q_base_jerk, duration, fps, base_frequency
    )
    joint_traj = np.stack([traj["qpos"].T, traj["qvel"].T, traj["qacc"].T], axis=1)

    return calculate_condition_number(m, d, joint_traj, ee_body_name)


def _joint_limit_constraint(coeffs_flat: np.ndarray, *opt_args) -> np.ndarray:
    """Constraint function for joint limits."""
    (
        q_base_pos,
        q_base_vel,
        q_base_acc,
        q_base_jerk,
        m,
        d,
        duration,
        fps,
        base_frequency,
        ee_body_name,
    ) = opt_args
    n_joints = m.njnt
    n_harmonics = coeffs_flat.shape[0] // (n_joints * 2)
    coeffs = coeffs_flat.reshape(n_joints, n_harmonics, 2)

    traj_data = _generate_combined_trajectory(
        coeffs, q_base_pos, q_base_vel, q_base_acc, q_base_jerk, duration, fps, base_frequency
    )
    qpos = traj_data["qpos"]

    qpos_min = m.jnt_range[:, 0]
    qpos_max = m.jnt_range[:, 1]

    # Inequality constraints for SLSQP must be non-negative (>= 0)
    lower_bound_violation = qpos - qpos_min[:, np.newaxis]
    upper_bound_violation = qpos_max[:, np.newaxis] - qpos

    return np.concatenate((lower_bound_violation.flatten(), upper_bound_violation.flatten()))


def generate_exciting_spline_trajectory(
    start_conditions: BoundaryCondition,
    end_conditions: BoundaryCondition,
    duration: float,
    fps: int,
    n_harmonics: int,
    base_frequency: float,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    ee_body_name: str,
    optimization_max_iter: int = 10,
) -> dict:
    """
    Generates a task-oriented excitation trajectory from start_qpos to end_qpos.
    The entire trajectory is optimized to be persistently exciting.
    """
    n_joints = m.njnt

    # Extract values from BoundaryCondition objects
    start_qpos = np.array(start_conditions.qpos)
    start_qvel = np.array(start_conditions.qvel)
    start_qacc = np.array(start_conditions.qacc)
    start_qjerk = np.array(start_conditions.qjerk)

    end_qpos = np.array(end_conditions.qpos)
    end_qvel = np.array(end_conditions.qvel)
    end_qacc = np.array(end_conditions.qacc)
    end_qjerk = np.array(end_conditions.qjerk)

    # 1. Generate the base trajectory (7th order spline)
    start_cond = BoundaryCondition(
        qpos=start_qpos.tolist(), qvel=start_qvel.tolist(), qacc=start_qacc.tolist(), qjerk=start_qjerk.tolist()
    )
    end_cond = BoundaryCondition(
        qpos=end_qpos.tolist(), qvel=end_qvel.tolist(), qacc=end_qacc.tolist(), qjerk=end_qjerk.tolist()
    )
    base_traj_data, _ = generate_spline_trajectory("seventh", duration, fps, start_cond, end_cond)

    q_base_pos = base_traj_data[:, 0, :].T
    q_base_vel = base_traj_data[:, 1, :].T
    q_base_acc = base_traj_data[:, 2, :].T
    q_base_jerk = base_traj_data[:, 3, :].T

    # Calculate and print the condition number of the base trajectory
    print("\nCalculating condition number for the base spline trajectory...")
    base_condition_number = calculate_condition_number(
        m=m,
        d=d,
        joint_trajectory=base_traj_data[:, :3, :],  # Slice to get pos, vel, acc
        ee_body_name=ee_body_name,
    )
    print(f"  Condition Number (Base Trajectory): {base_condition_number:.4e}")

    # 2. Optimize the coefficients of the excitation component
    initial_coeffs = np.random.rand(n_joints, n_harmonics, 2) * 0.1  # Start with small random amplitudes
    opt_args = (
        q_base_pos,
        q_base_vel,
        q_base_acc,
        q_base_jerk,
        m,
        d,
        duration,
        fps,
        base_frequency,
        ee_body_name,
    )

    # Define a callback function to print progress
    iteration_count = [0]  # Use a list to make it mutable inside the callback

    def _optimization_callback(coeffs_flat):
        iteration_count[0] += 1
        current_condition_number = _exciting_spline_objective(coeffs_flat, *opt_args)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {timestamp} - Iteration: {iteration_count[0]:>3}, Condition Number: {current_condition_number:.4e}")

    print("\nStarting trajectory optimization...")
    result = minimize(
        fun=_exciting_spline_objective,
        x0=initial_coeffs.flatten(),
        args=opt_args,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": _joint_limit_constraint, "args": opt_args}],
        options={"maxiter": optimization_max_iter, "disp": False},
        callback=_optimization_callback,
    )
    print("...Optimization finished.")
    final_cond_num = result.fun
    print(f"Final condition number: {final_cond_num:.4e}")
    optimal_coeffs = result.x.reshape(n_joints, n_harmonics, 2)

    # 3. Generate the final, optimized trajectory by combining the base and excitation parts
    final_trajectory = _generate_combined_trajectory(
        optimal_coeffs,
        q_base_pos,
        q_base_vel,
        q_base_acc,
        q_base_jerk,
        duration,
        fps,
        base_frequency,
    )

    final_trajectory["condition_number"] = final_cond_num.item()

    return final_trajectory
