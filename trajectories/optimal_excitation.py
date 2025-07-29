import mujoco
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from dynamics.dynamics import calculate_condition_number
from trajectories.spline_interpolation import generate_spline_trajectory


def generate_optimal_excitation_trajectory(
    duration: float,
    fps: int,
    n_harmonics: int,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    base_frequency: float,
    jointpos_offset: ArrayLike = (0, 0, 0, 0, 0, 0),
    ee_body_name: str = "link6",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates an optimal excitation trajectory by minimizing the condition number
    of the regressor matrix.

    Args:
        duration: The total duration of the trajectory in seconds.
        fps: The frequency (frames per second) to generate trajectory points.
        n_harmonics: The number of harmonics to use for the sinusoidal trajectory.
        m: MuJoCo MjModel object.
        d: MuJoCo MjData object.
        base_frequency: The base frequency 'f' [Hz] for the harmonics.
        jointpos_offset: A 1D numpy array representing the constant offset angle for each joint.
        ee_body_name: The name of the end-effector body in the MuJoCo model.

    Returns:
        A tuple (t_vec, qpos, qvel, qacc) representing the optimized trajectory.
    """
    n_joints = m.njnt
    # Initial guess for coeffs: small random values
    # Shape: (n_joints, n_harmonics, 2)
    initial_coeffs = np.random.rand(n_joints, n_harmonics, 2) * 0.01

    # Define the arguments to be passed to the objective function
    objective_args = (
        m,
        d,
        duration,
        fps,
        jointpos_offset,
        base_frequency,
        ee_body_name,
    )

    # Perform the optimization
    # We need to flatten the coeffs for the optimizer and then reshape inside the objective function
    # Flatten the initial_coeffs for the optimizer
    initial_coeffs_flat = initial_coeffs.flatten()

    # Define a wrapper objective function that reshapes coeffs
    def _objective_function_wrapper(coeffs_flat, *args):
        _m, _d, _duration, _fps, _jointpos_offset, _base_frequency, _ee_body_name = args
        _coeffs = coeffs_flat.reshape(n_joints, n_harmonics, 2)
        return objective_function(
            coeffs=_coeffs,
            m=_m,
            d=_d,
            duration=_duration,
            fps=_fps,
            jointpos_offset=_jointpos_offset,
            base_frequency=_base_frequency,
            ee_body_name=_ee_body_name,
        )

    # Perform the optimization
    result = minimize(
        fun=_objective_function_wrapper,
        x0=initial_coeffs_flat,
        args=objective_args,
        method="Nelder-Mead",  # A simple, derivative-free method
        options={"maxiter": 100, "disp": True},  # Limited iterations for testing
    )

    # Reshape the optimized coefficients back to their original shape
    optimized_coeffs = result.x.reshape(n_joints, n_harmonics, 2)

    # Generate the final trajectory with optimized coefficients
    t_vec, qpos, qvel, qacc = generate_sinusoidal_trajectory(
        duration=duration,
        fps=fps,
        coeffs=optimized_coeffs,
        base_frequency=base_frequency,
        jointpos_offset=jointpos_offset,
    )

    return t_vec, qpos, qvel, qacc, optimized_coeffs


def generate_full_trajectory(
    main_duration: float,
    transition_duration: float,
    fps: int,
    n_harmonics: int,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    base_frequency: float,
    start_qpos: ArrayLike,
    ee_body_name: str = "link6",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a full excitation trajectory including transition splines.

    This function orchestrates the generation of:
    1. An optimal excitation (main) trajectory.
    2. A transition spline from a starting configuration to the main trajectory.
    3. A transition spline from the main trajectory back to the starting configuration.

    Args:
        main_duration: Duration of the main excitation part in seconds.
        transition_duration: Duration of each transition spline in seconds.
        fps: Frames per second for the trajectory.
        n_harmonics: Number of harmonics for the sinusoidal trajectory.
        m: MuJoCo MjModel object.
        d: MuJoCo MjData object.
        base_frequency: Base frequency for the harmonics.
        start_qpos: The starting and ending joint positions for the trajectory.
        ee_body_name: The name of the end-effector body.

    Returns:
        A tuple (t_vec, qpos, qvel, qacc) for the complete, combined trajectory.
    """
    # 1. Generate the optimal excitation (main) trajectory
    # The main trajectory starts from a zero-offset position.
    main_t, main_qpos, main_qvel, main_qacc, _ = generate_optimal_excitation_trajectory(
        duration=main_duration,
        fps=fps,
        n_harmonics=n_harmonics,
        m=m,
        d=d,
        base_frequency=base_frequency,
        jointpos_offset=start_qpos,  # Use start_qpos as the offset
        ee_body_name=ee_body_name,
    )

    # 2. Generate the first transition spline (start -> main)
    start_conditions = {"qpos": start_qpos, "qvel": [0] * m.njnt, "qacc": [0] * m.njnt}
    end_conditions_t1 = {"qpos": main_qpos[:, 0], "qvel": main_qvel[:, 0], "qacc": main_qacc[:, 0]}

    # Note: generate_spline_trajectory returns (n_frames, 3, n_dof)
    # We need to transpose it to (3, n_dof, n_frames) and then unpack
    t1_data = generate_spline_trajectory(
        duration=transition_duration,
        fps=fps,
        start_conditions=start_conditions,
        end_conditions=end_conditions_t1,
        trajectory_type="fifth",
    )
    # Transpose from (n_frames, 3, n_dof) to (n_dof, 3, n_frames) then unpack
    t1_qpos = t1_data.transpose(2, 1, 0)[:, 0, :]
    t1_qvel = t1_data.transpose(2, 1, 0)[:, 1, :]
    t1_qacc = t1_data.transpose(2, 1, 0)[:, 2, :]

    # 3. Generate the second transition spline (main -> end)
    start_conditions_t2 = {"qpos": main_qpos[:, -1], "qvel": main_qvel[:, -1], "qacc": main_qacc[:, -1]}
    end_conditions = {"qpos": start_qpos, "qvel": [0] * m.njnt, "qacc": [0] * m.njnt}

    t2_data = generate_spline_trajectory(
        duration=transition_duration,
        fps=fps,
        start_conditions=start_conditions_t2,
        end_conditions=end_conditions,
        trajectory_type="fifth",
    )
    # Transpose from (n_frames, 3, n_dof) to (n_dof, 3, n_frames) then unpack
    t2_qpos = t2_data.transpose(2, 1, 0)[:, 0, :]
    t2_qvel = t2_data.transpose(2, 1, 0)[:, 1, :]
    t2_qacc = t2_data.transpose(2, 1, 0)[:, 2, :]

    # 4. Concatenate trajectories
    full_qpos = np.hstack((t1_qpos, main_qpos, t2_qpos))
    full_qvel = np.hstack((t1_qvel, main_qvel, t2_qvel))
    full_qacc = np.hstack((t1_qacc, main_qacc, t2_qacc))

    total_duration = 2 * transition_duration + main_duration
    n_total_frames = full_qpos.shape[1]
    full_t_vec = np.linspace(0, total_duration, n_total_frames)

    return full_t_vec, full_qpos, full_qvel, full_qacc


def generate_sinusoidal_trajectory(
    duration: float,
    fps: int,
    coeffs: ArrayLike,
    base_frequency: float,
    jointpos_offset: ArrayLike = (0, 0, 0, 0, 0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a trajectory for multiple joints based on a sum of sinusoids.

    This function implements the trajectory generation described in Eq. (10) and (11)
    of the reference paper. The trajectory for each joint is a sum of N weighted
    sine and cosine functions plus a constant offset.

    Args:
        coeffs (np.ndarray):
            A 3D numpy array of shape (n_joints, n_harmonics, 2). It holds the coefficients for the sinusoids.
            coeffs[i, k, 0] corresponds to p_ik (sine coefficient for joint i, harmonic k). coeffs[i, k, 1]
            corresponds to d_ik (cosine coefficient for joint i, harmonic k).
        jointpos_offset (np.ndarray):
            A 1D numpy array of shape (n_joints,) representing the constant offset angle for each joint (q_i,0).
        base_frequency (float):
            The base frequency 'f' [Hz] for the harmonics. The k-th harmonic will have a frequency of k * f.
        duration (float):
            The total duration of the trajectory in seconds.
        fps (int):
            The frequency (frames per second) to generate trajectory points.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        - t_vec: 1D array of time points.
        - qpos: 2D array of joint positions `q(t)` of shape (n_joints, n_timesteps).
        - qvel: 2D array of joint velocities `q_dot(t)` of shape (n_joints, n_timesteps).
        - qacc: 2D array of joint accelerations `q_ddot(t)` of shape (n_joints, n_timesteps).
    """
    n_joints, n_harmonics, _ = coeffs.shape
    if np.array(jointpos_offset).shape[0] != n_joints:
        raise ValueError("Shape mismatch between coeffs and jointpos_offset.")

    # 1. Create the time vector
    n_timesteps = int(duration * fps)
    t_vec = np.arange(n_timesteps) / fps

    # 2. Initialize output arrays
    qpos = np.zeros((n_joints, n_timesteps))
    qvel = np.zeros((n_joints, n_timesteps))
    qacc = np.zeros((n_joints, n_timesteps))

    # 3. Calculate the sum of sinusoids for each harmonic
    for k in range(1, n_harmonics + 1):
        omega = 2 * np.pi * k * base_frequency

        p_k = coeffs[:, k - 1, 0]
        d_k = coeffs[:, k - 1, 1]

        sin_t = np.sin(omega * t_vec)
        cos_t = np.cos(omega * t_vec)

        # Position contribution from this harmonic
        qpos_k = p_k[:, np.newaxis] * sin_t + d_k[:, np.newaxis] * cos_t
        qpos += qpos_k

        # Velocity contribution (derivative of position)
        qvel_k = omega * (p_k[:, np.newaxis] * cos_t - d_k[:, np.newaxis] * sin_t)
        qvel += qvel_k

        # Acceleration contribution (derivative of velocity)
        qacc_k = -(omega**2) * qpos_k
        qacc += qacc_k

    # 4. Add the constant offset to the final position trajectory
    qpos += np.array(jointpos_offset)[:, np.newaxis]

    return t_vec, qpos, qvel, qacc


def objective_function(
    coeffs: np.ndarray,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    duration: float,
    fps: int,
    jointpos_offset: ArrayLike,
    base_frequency: float,
    ee_body_name: str,
) -> float:
    """
    Calculates the condition number of the regressor matrix for a given set of
    excitation trajectory coefficients. This function serves as the objective
    function for the optimization process.

    Args:
        coeffs: A 3D numpy array of shape (n_joints, n_harmonics, 2) representing
                the coefficients (p_ik, d_ik) for the sinusoidal trajectory.
                This is the variable to be optimized.
        m: MuJoCo MjModel object.
        d: MuJoJo MjData object.
        duration: The total duration of the trajectory in seconds.
        fps: The frequency (frames per second) to generate trajectory points.
        jointpos_offset: A 1D numpy array representing the constant offset angle for each joint.
        base_frequency: The base frequency 'f' [Hz] for the harmonics.
        ee_body_name: The name of the end-effector body in the MuJoCo model.

    Returns:
        The condition number of the stacked regressor matrix, which is to be minimized.
    """
    # 1. Generate the excitation trajectory
    t_vec, qpos, qvel, qacc = generate_sinusoidal_trajectory(
        duration=duration,
        fps=fps,
        coeffs=coeffs,
        base_frequency=base_frequency,
        jointpos_offset=jointpos_offset,
    )

    # Reshape qpos, qvel, qacc from (n_joints, n_timesteps) to (n_timesteps, 3, n_joints)
    # as expected by calculate_condition_number
    joint_trajectory = np.stack([qpos.T, qvel.T, qacc.T], axis=1)

    # 2. Calculate the condition number
    condition_number = calculate_condition_number(
        m=m,
        d=d,
        joint_trajectory=joint_trajectory,
        ee_body_name=ee_body_name,
    )

    return condition_number
