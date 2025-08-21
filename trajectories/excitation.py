import mujoco
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from dynamics.condition_number import calculate_condition_number
from trajectories.spline_interpolation import (
    BoundaryCondition,
    generate_spline_trajectory,
)


def _find_optimal_coeffs(
    n_joints: int,
    n_harmonics: int,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    main_duration: float,
    fps: int,
    start_qpos: ArrayLike,
    base_frequency: float,
    ee_body_name: str,
    optimization_max_iter: int,
) -> np.ndarray:
    """
    Finds the optimal coefficients for the sinusoidal trajectory by minimizing the condition number.
    """
    initial_coeffs = np.random.rand(n_joints, n_harmonics, 2) * 0.25
    initial_coeffs_flat = initial_coeffs.flatten()

    objective_args = (
        m,
        d,
        main_duration,
        fps,
        start_qpos,
        base_frequency,
        ee_body_name,
    )

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

    # コールバック関数を定義
    def _optimization_callback(coeffs_flat):
        _m, _d, _duration, _fps, _jointpos_offset, _base_frequency, _ee_body_name = objective_args
        _coeffs = coeffs_flat.reshape(n_joints, n_harmonics, 2)
        current_condition_number = objective_function(
            coeffs=_coeffs,
            m=_m,
            d=_d,
            duration=_duration,
            fps=_fps,
            jointpos_offset=_jointpos_offset,
            base_frequency=_base_frequency,
            ee_body_name=_ee_body_name,
        )
        print(f"  Optimization Iteration Condition Number: {current_condition_number:.4e}")

    def _joint_position_constraint(coeffs_flat, *args):
        _m, _d, _duration, _fps, _jointpos_offset, _base_frequency, _ee_body_name = args
        _coeffs = coeffs_flat.reshape(n_joints, n_harmonics, 2)

        _, qpos, _, _, _ = generate_sinusoidal_trajectory(
            duration=_duration,
            fps=_fps,
            coeffs=_coeffs,
            base_frequency=_base_frequency,
            jointpos_offset=_jointpos_offset,
        )

        # Extract qpos_min and qpos_max from m.jnt_range
        _qpos_min = _m.jnt_range[:, 0]
        _qpos_max = _m.jnt_range[:, 1]

        lower_bound_violation = qpos - _qpos_min[:, np.newaxis]
        upper_bound_violation = _qpos_max[:, np.newaxis] - qpos

        return np.concatenate((lower_bound_violation.flatten(), upper_bound_violation.flatten()))

    constraints = {"type": "ineq", "fun": _joint_position_constraint, "args": objective_args}

    result = minimize(
        fun=_objective_function_wrapper,
        x0=initial_coeffs_flat,
        args=objective_args,
        method="SLSQP",
        options={"disp": False, "maxiter": optimization_max_iter},
        constraints=constraints,
        callback=_optimization_callback,
    )

    return result.x.reshape(n_joints, n_harmonics, 2)


def generate_optimal_excitation_trajectory(
    main_duration: float,
    transition_duration: float,
    fps: int,
    n_harmonics: int,
    m: mujoco.MjModel | None,
    d: mujoco.MjData | None,
    base_frequency: float,
    start_qpos: ArrayLike,
    ee_body_name: str = "link6",
    manipulator_path: str | None = None,
    optimization_max_iter: int = 10,
) -> dict:
    """
    Generates an optimal excitation trajectory by minimizing the condition number
    of the regressor matrix, including transition splines.

    Returns:
        A dictionary containing the trajectory data and metadata.
        - t: Time vector
        - qpos, qvel, qacc, qjerk: Joint trajectory data
        - excitation: Dictionary with start and end indices of the excitation phase.
    """
    if m is None:
        if manipulator_path is None:
            raise ValueError("Either m or manipulator_path must be provided.")
        with open(manipulator_path) as f:
            xml_string = f.read()
        import re

        xml_string_no_sensors = re.sub(r"<sensor>.*?</sensor>", "", xml_string, flags=re.DOTALL)
        m = mujoco.MjModel.from_xml_string(xml_string_no_sensors)
        d = mujoco.MjData(m)

    optimized_coeffs = _find_optimal_coeffs(
        n_joints=m.njnt,
        n_harmonics=n_harmonics,
        m=m,
        d=d,
        main_duration=main_duration,
        fps=fps,
        start_qpos=start_qpos,
        base_frequency=base_frequency,
        ee_body_name=ee_body_name,
        optimization_max_iter=optimization_max_iter,
    )

    main_t, main_qpos, main_qvel, main_qacc, main_qjerk = generate_sinusoidal_trajectory(
        duration=main_duration,
        fps=fps,
        coeffs=optimized_coeffs,
        base_frequency=base_frequency,
        jointpos_offset=start_qpos,
    )

    if transition_duration < 1e-6:
        return {
            "t": main_t,
            "qpos": main_qpos,
            "qvel": main_qvel,
            "qacc": main_qacc,
            "qjerk": main_qjerk,
            "excitation": {
                "start_index": 0,
                "end_index": main_qpos.shape[1],
            },
        }

    # "+ 1.0 / fps" is to add a buffer frame that is sliced out later to make the full trajectory smooth
    _transition_duration = transition_duration + 1.0 / fps

    start_cond_t1 = BoundaryCondition(
        qpos=start_qpos.tolist(), qvel=[0.0] * m.njnt, qacc=[0.0] * m.njnt, qjerk=[0.0] * m.njnt
    )
    end_cond_t1 = BoundaryCondition(
        qpos=main_qpos[:, 0].tolist(),
        qvel=main_qvel[:, 0].tolist(),
        qacc=main_qacc[:, 0].tolist(),
        qjerk=main_qjerk[:, 0].tolist(),
    )
    t1_data, _ = generate_spline_trajectory(
        trajectory_type="seventh",
        duration=_transition_duration,
        fps=fps,
        start_conditions=start_cond_t1,
        end_conditions=end_cond_t1,
    )
    t1_qpos, t1_qvel, t1_qacc, t1_qjerk = (
        t1_data[:, 0, :].T,
        t1_data[:, 1, :].T,
        t1_data[:, 2, :].T,
        t1_data[:, 3, :].T,
    )

    start_cond_t2 = BoundaryCondition(
        qpos=main_qpos[:, -1].tolist(),
        qvel=main_qvel[:, -1].tolist(),
        qacc=main_qacc[:, -1].tolist(),
        qjerk=main_qjerk[:, -1].tolist(),
    )
    end_cond_t2 = BoundaryCondition(
        qpos=start_qpos.tolist(), qvel=[0.0] * m.njnt, qacc=[0.0] * m.njnt, qjerk=[0.0] * m.njnt
    )
    t2_data, _ = generate_spline_trajectory(
        trajectory_type="seventh",
        duration=_transition_duration,
        fps=fps,
        start_conditions=start_cond_t2,
        end_conditions=end_cond_t2,
    )
    t2_qpos, t2_qvel, t2_qacc, t2_qjerk = (
        t2_data[:, 0, :].T,
        t2_data[:, 1, :].T,
        t2_data[:, 2, :].T,
        t2_data[:, 3, :].T,
    )

    # slice out the first and last frame of the fore and rear transition directory, respectively
    transition1_qpos = t1_qpos[:, :-1]
    full_qpos = np.hstack((transition1_qpos, main_qpos, t2_qpos[:, 1:]))
    full_qvel = np.hstack((t1_qvel[:, :-1], main_qvel, t2_qvel[:, 1:]))
    full_qacc = np.hstack((t1_qacc[:, :-1], main_qacc, t2_qacc[:, 1:]))
    full_qjerk = np.hstack((t1_qjerk[:, :-1], main_qjerk, t2_qjerk[:, 1:]))

    total_duration = 2 * transition_duration + main_duration
    n_total_frames = full_qpos.shape[1]
    full_t_vec = np.linspace(0, total_duration, n_total_frames)

    start_index = transition1_qpos.shape[1]
    end_index = start_index + main_qpos.shape[1]

    return {
        "t": full_t_vec,
        "qpos": full_qpos,
        "qvel": full_qvel,
        "qacc": full_qacc,
        "qjerk": full_qjerk,
        "excitation": {
            "start_index": start_index,
            "end_index": end_index,
        },
    }


def generate_sinusoidal_trajectory(
    duration: float,
    fps: int,
    coeffs: ArrayLike,
    base_frequency: float,
    jointpos_offset: ArrayLike = (0, 0, 0, 0, 0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        - t_vec: 1D array of time points.
        - qpos: 2D array of joint positions `q(t)` of shape (n_joints, n_frames).
        - qvel: 2D array of joint velocities `q_dot(t)` of shape (n_joints, n_frames).
        - qacc: 2D array of joint accelerations `q_ddot(t)` of shape (n_joints, n_frames).
        - qjerk: 2D array of joint jerks `q_dddot(t)` of shape (n_joints, n_frames).
    """
    n_joints, n_harmonics, _ = coeffs.shape
    if np.array(jointpos_offset).shape[0] != n_joints:
        raise ValueError("Shape mismatch between coeffs and jointpos_offset.")

    # 1. Create the time vector
    n_frames = int(duration * fps)
    t_vec = np.arange(n_frames) / fps

    # 2. Initialize output arrays
    qpos = np.zeros((n_joints, n_frames))
    qvel = np.zeros((n_joints, n_frames))
    qacc = np.zeros((n_joints, n_frames))
    qjerk = np.zeros((n_joints, n_frames))  # Added qjerk initialization

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

        # Jerk contribution (derivative of acceleration)
        qjerk_k = -(omega**2) * qvel_k
        qjerk += qjerk_k

    # 4. Add the constant offset to the final position trajectory
    qpos += np.array(jointpos_offset)[:, np.newaxis]

    return t_vec, qpos, qvel, qacc, qjerk


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
    t_vec, qpos, qvel, qacc, _ = generate_sinusoidal_trajectory(
        duration=duration,
        fps=fps,
        coeffs=coeffs,
        base_frequency=base_frequency,
        jointpos_offset=jointpos_offset,
    )

    # Reshape qpos, qvel, qacc from (n_joints, n_frames) to (n_frames, 3, n_joints)
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
