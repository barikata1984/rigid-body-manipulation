import numpy as np
from numpy.typing import ArrayLike


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
    if jointpos_offset.shape[0] != n_joints:
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
    qpos += jointpos_offset[:, np.newaxis]

    return t_vec, qpos, qvel, qacc
