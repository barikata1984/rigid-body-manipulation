from dataclasses import dataclass, field

import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray


@dataclass
class BoundaryCondition:
    qpos: list[float] = field(default_factory=lambda: [0.0] * 6)
    qvel: list[float] = field(default_factory=lambda: [0.0] * 6)
    qacc: list[float] = field(default_factory=lambda: [0.0] * 6)
    qjerk: list[float] = field(default_factory=lambda: [0.0] * 6)


def _generate_fifth_order_spline_coeffs(t_s: float, t_e: float, boundary_conditions: NDArray) -> NDArray:
    """Generates coefficients for a 5th-order spline with specified boundary conditions."""
    boundary_matrix = np.array(
        [
            # start pos
            [t_s**5, t_s**4, t_s**3, t_s**2, t_s, 1],
            # end pos
            [t_e**5, t_e**4, t_e**3, t_e**2, t_e, 1],
            # start vel
            [5 * t_s**4, 4 * t_s**3, 3 * t_s**2, 2 * t_s**1, 1, 0],
            # end vel
            [5 * t_e**4, 4 * t_e**3, 3 * t_e**2, 2 * t_e**1, 1, 0],
            # start acc
            [20 * t_s**3, 12 * t_s**2, 6 * t_s**1, 2, 0, 0],
            # end acc
            [20 * t_e**3, 12 * t_e**2, 6 * t_e**1, 2, 0, 0],
        ],
        dtype=float,
    )
    return la.solve(boundary_matrix, boundary_conditions)


def _generate_seventh_order_spline_coeffs(t_s: float, t_e: float, boundary_conditions: NDArray) -> NDArray:
    """Generates coefficients for a 7th-order spline with specified boundary conditions."""
    boundary_matrix = np.array(
        [
            # start pos
            [t_s**7, t_s**6, t_s**5, t_s**4, t_s**3, t_s**2, t_s, 1],
            # end pos
            [t_e**7, t_e**6, t_e**5, t_e**4, t_e**3, t_e**2, t_e, 1],
            # start vel
            [7 * t_s**6, 6 * t_s**5, 5 * t_s**4, 4 * t_s**3, 3 * t_s**2, 2 * t_s**1, 1, 0],
            # end vel
            [7 * t_e**6, 6 * t_e**5, 5 * t_e**4, 4 * t_e**3, 3 * t_e**2, 2 * t_e**1, 1, 0],
            # start acc
            [42 * t_s**5, 30 * t_s**4, 20 * t_s**3, 12 * t_s**2, 6 * t_s**1, 2, 0, 0],
            # end acc
            [42 * t_e**5, 30 * t_e**4, 20 * t_e**3, 12 * t_e**2, 6 * t_e**1, 2, 0, 0],
            # start jerk
            [210 * t_s**4, 120 * t_s**3, 60 * t_s**2, 24 * t_s**1, 6, 0, 0, 0],
            # end jerk
            [210 * t_e**4, 120 * t_e**3, 60 * t_e**2, 24 * t_e**1, 6, 0, 0, 0],
        ],
        dtype=float,
    )
    return la.solve(boundary_matrix, boundary_conditions)


def generate_spline_trajectory(
    trajectory_type: str,
    duration: float,  # [s]
    fps: float,  # [Hz]
    start_conditions: BoundaryCondition,
    end_conditions: BoundaryCondition,
) -> NDArray:
    """
    Generates a spline trajectory for multiple joints with specified start and end conditions.

    Args:
        duration: The duration of the trajectory in seconds.
        fps: The frequency of the trajectory in Hz.
        start_conditions: A dictionary with "qpos", "qvel", "qacc", and optionally "qjerk" at the start.
        end_conditions: A dictionary with "qpos", "qvel", "qacc", and optionally "qjerk" at the end.
        trajectory_type: The type of spline to use, either "fifth" or "sixth".

    Returns:
        A numpy array of shape (n_frames, 4, n_dof) containing the joint positions,
        velocities, accelerations, and jerks at each frame.
    """
    n_frames = int(duration * fps)
    t_s = 0.0
    t_e = duration

    start_qpos = np.array(start_conditions.qpos)
    start_qvel = np.array(start_conditions.qvel)
    start_qacc = np.array(start_conditions.qacc)
    end_qpos = np.array(end_conditions.qpos)
    end_qvel = np.array(end_conditions.qvel)
    end_qacc = np.array(end_conditions.qacc)

    n_dof = len(start_qpos)
    qposs = np.zeros((n_frames, n_dof))
    qvels = np.zeros((n_frames, n_dof))
    qaccs = np.zeros((n_frames, n_dof))
    qjerks = np.zeros((n_frames, n_dof))  # Added qjerks initialization

    time_points = np.linspace(t_s, t_e, n_frames)

    for i in range(n_dof):
        if "fifth" in trajectory_type:
            boundary_conditions = np.array(
                [
                    start_qpos[i],
                    end_qpos[i],
                    start_qvel[i],
                    end_qvel[i],
                    start_qacc[i],
                    end_qacc[i],
                ]
            )
            coeffs = _generate_fifth_order_spline_coeffs(t_s, t_e, boundary_conditions)
            poly_t = np.array([time_points**j for j in range(5, -1, -1)]).T
            poly_vel_t = np.array([j * time_points ** (j - 1) for j in range(5, 0, -1)]).T
            poly_acc_t = np.array([j * (j - 1) * time_points ** (j - 2) for j in range(5, 1, -1)]).T
            poly_jerk_t = np.array(
                [j * (j - 1) * (j - 2) * time_points ** (j - 3) for j in range(5, 2, -1)]
            ).T  # Added jerk poly

            qposs[:, i] = poly_t @ coeffs
            qvels[:, i] = poly_vel_t @ coeffs[:-1]
            qaccs[:, i] = poly_acc_t @ coeffs[:-2]
            qjerks[:, i] = poly_jerk_t @ coeffs[:-3]  # Added jerk calculation

        elif "seventh" in trajectory_type:
            start_qjerk = start_conditions.qjerk[i]
            end_qjerk = end_conditions.qjerk[i]

            boundary_conditions = np.array(
                [
                    start_qpos[i],
                    end_qpos[i],
                    start_qvel[i],
                    end_qvel[i],
                    start_qacc[i],
                    end_qacc[i],
                    start_qjerk,
                    end_qjerk,
                ],
                dtype=float,
            )
            coeffs = _generate_seventh_order_spline_coeffs(t_s, t_e, boundary_conditions)
            poly_t = np.array([time_points**j for j in range(7, -1, -1)]).T
            poly_vel_t = np.array([j * time_points ** (j - 1) for j in range(7, 0, -1)]).T
            poly_acc_t = np.array([j * (j - 1) * time_points ** (j - 2) for j in range(7, 1, -1)]).T
            poly_jerk_t = np.array([j * (j - 1) * (j - 2) * time_points ** (j - 3) for j in range(7, 2, -1)]).T

            qposs[:, i] = poly_t @ coeffs
            qvels[:, i] = poly_vel_t @ coeffs[:-1]
            qaccs[:, i] = poly_acc_t @ coeffs[:-2]
            qjerks[:, i] = poly_jerk_t @ coeffs[:-3]
        else:
            raise ValueError("Invalid trajectory_type. Must be 'fifth', 'sixth' or 'seventh'.")

    return {
        "t": time_points,
        "qpos": qposs.T,
        "qvel": qvels.T,
        "qacc": qaccs.T,
        "qjerk": qjerks.T,
    }
