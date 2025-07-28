import json

import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray


def _generate_fifth_order_spline_coeffs(t_s: float, t_e: float) -> NDArray:
    """Generates coefficients for a 5th-order spline with zero velocity and acceleration at start/end."""
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

    normalized_boundaries = np.array(
        [
            0,  # start pos
            1,  # end pos
            0,  # start vel
            0,  # end vel
            0,  # start acc
            0,  # end acc
        ]
    )
    return la.solve(boundary_matrix, normalized_boundaries)


def _generate_sixth_order_spline_coeffs(t_s: float, t_e: float) -> NDArray:
    """Generates coefficients for a 6th-order spline with zero velocity, acceleration, and jerk at start/end."""
    boundary_matrix = np.array(
        [
            # start pos
            [t_s**6, t_s**5, t_s**4, t_s**3, t_s**2, t_s, 1],
            # end pos
            [t_e**6, t_e**5, t_e**4, t_e**3, t_e**2, t_e, 1],
            # start vel
            [6 * t_s**5, 5 * t_s**4, 4 * t_s**3, 3 * t_s**2, 2 * t_s**1, 1, 0],
            # end vel
            [6 * t_e**5, 5 * t_e**4, 4 * t_e**3, 3 * t_e**2, 2 * t_e**1, 1, 0],
            # start acc
            [30 * t_s**4, 20 * t_s**3, 12 * t_s**2, 6 * t_s**1, 2, 0, 0],
            # end acc
            [30 * t_e**4, 20 * t_e**3, 12 * t_e**2, 6 * t_e**1, 2, 0, 0],
            # start jerk
            [120 * t_s**3, 60 * t_s**2, 24 * t_s**1, 6, 0, 0, 0],
        ],
        dtype=float,
    )

    normalized_boundaries = np.array(
        [
            0,  # start pos
            1,  # end pos
            0,  # start vel
            0,  # end vel
            0,  # start acc
            0,  # end acc
            0,  # start jerk
        ]
    )
    return la.solve(boundary_matrix, normalized_boundaries)


def generate_spline_trajectory(
    duration: float,  # [s]
    fps: float,  # [Hz]
    displacement: list[float],  # [m, m, m, rad, rad, rad]
    jointpos_offset: list[float],  # [m, m, m, rad, rad, rad]
    trajectory_type: str = "fifth",  # "fifth" or "sixth"
) -> NDArray:
    # Set the time window
    n_frames = int(duration * fps)
    frame_interval = 1.0 / fps
    t_s = 0
    t_e = t_s + n_frames

    if "fifth" in trajectory_type:
        coeffs = _generate_fifth_order_spline_coeffs(t_s, t_e)
        # Polynomial terms for 5th order
        fifth_poly = np.array([[f**i for i in range(5, -1, -1)] for f in range(n_frames)])
        fourth_poly_deriv = np.array([[f**i * (i + 1) for i in range(4, -1, -1)] for f in range(n_frames)])
        third_poly_deriv2 = np.array([[f**i * (i + 1) * (i + 2) for i in range(3, -1, -1)] for f in range(n_frames)])

        qposs = np.outer(fifth_poly.dot(coeffs), displacement) + jointpos_offset
        qvels = np.outer(fourth_poly_deriv.dot(coeffs[:-1]), displacement) / frame_interval
        qaccs = np.outer(third_poly_deriv2.dot(coeffs[:-2]), displacement) / frame_interval**2

    elif "sixth" in trajectory_type:
        coeffs = _generate_sixth_order_spline_coeffs(t_s, t_e)
        # Polynomial terms for 6th order
        sixth_poly = np.array([[f**i for i in range(6, -1, -1)] for f in range(n_frames)])
        fifth_poly_deriv = np.array([[f**i * (i + 1) for i in range(5, -1, -1)] for f in range(n_frames)])
        fourth_poly_deriv2 = np.array([[f**i * (i + 1) * (i + 2) for i in range(4, -1, -1)] for f in range(n_frames)])

        qposs = np.outer(sixth_poly.dot(coeffs), displacement) + jointpos_offset
        qvels = np.outer(fifth_poly_deriv.dot(coeffs[:-1]), displacement) / frame_interval
        qaccs = np.outer(fourth_poly_deriv2.dot(coeffs[:-2]), displacement) / frame_interval**2

    else:
        raise ValueError("Invalid trajectory_type. Must be 'fifth' or 'sixth'.")

    jointvars = []
    for i in range(n_frames):
        jointvar = {
            "qpos": qposs[i].tolist(),
            "qvel": qvels[i].tolist(),
            "qacc": qaccs[i].tolist(),
        }
        jointvars.append(jointvar)

    json_data = {
        "duration": duration,
        "fps": fps,
        "jointpos_offset": jointpos_offset,
        "displacement": displacement,
        "trajectory_type": trajectory_type,
        "jointvars": jointvars,
    }

    # Save to JSON file
    json_file_path = (
        f"experiment_setups/trajectories/_{trajectory_type}.json"  # Hardcoded for now, can be made configurable
    )
    with open(json_file_path, "w") as f:
        json.dump(json_data, f, indent=4)

    return np.stack([qposs, qvels, qaccs], axis=1)
