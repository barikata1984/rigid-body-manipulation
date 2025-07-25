import json
from collections.abc import Callable

import numpy as np
from numpy import linalg as la
from numpy.typing import ArrayLike, NDArray


def get_trajectory_interpolated_with_fifth_order_spline(
    displacement: ArrayLike,  # [m, m, m, rad, rad, rad]
    pos_offset: ArrayLike,  # [m, m, m, rad, rad, rad]
    frame_span: float,  # [s]
    n_frames: int,  # [steps]
    init_step: int = 0,
) -> Callable[[int], NDArray]:
    # Set the time window
    t_s = init_step
    t_e = t_s + n_frames

    # define the parameter matrix for a differentiated fifth-order polynomial
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
            # end vel
            [20 * t_e**3, 12 * t_e**2, 6 * t_e**1, 2, 0, 0],
        ],
        dtype=float,
    )

    # Define normalized boundaries of a spline.
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

    # Find coefficient of the 5th-order spline s.t.
    # boundary_matreix @ coeffs = normalized_boundaries  # shape: (6, 6) @ (6,) = (6,)
    coeffs = la.solve(
        boundary_matrix,
        normalized_boundaries,  # boundary values
    )

    displacement = np.array(displacement)
    pos_offset = np.array(pos_offset)

    fifth = np.array([[f**i for i in range(5, -1, -1)] for f in range(n_frames)])  # 5th to 0th (6 elems in total))
    fourth = np.array(
        [[f**i * (i + 1) for i in range(4, -1, -1)] for f in range(n_frames)]
    )  # 4th to 0th (5 elems in total)
    third = np.array(
        [[f**i * (i + 1) * (i + 2) for i in range(3, -1, -1)] for f in range(n_frames)]
    )  # 3rd to 0th  (4 elems in total))

    # Multipy the trajectory with displacement since the trajectory is normalized
    pos = np.outer(fifth.dot(coeffs[:]), displacement) + pos_offset  # unit: [m] ← [m] * [.]  + [m]
    vel = np.outer(fourth.dot(coeffs[:-1]), displacement) / frame_span  # unit: [m/s] ← ([m] * [.]  + [m]) / [s]
    acc = np.outer(third.dot(coeffs[:-2]), displacement) / frame_span**2  # unit: [m/s^2] ← ([m] * [.]  + [m]) / [s^2]

    trajectories = np.stack([pos, vel, acc], axis=1)

    # Prepare data for JSON
    duration = frame_span
    fps = n_frames / frame_span  # Assuming frame_span is the total duration for n_frames

    frames_data = []
    for i in range(n_frames):
        frame_dict = {
            "qpos": trajectories[i, 0, :].tolist(),
            "qvel": trajectories[i, 1, :].tolist(),
            "qacc": trajectories[i, 2, :].tolist(),
        }
        frames_data.append(frame_dict)

    json_data = {
        "duration": duration,
        "fps": fps,
        "pos_offset": pos_offset.tolist(),
        "displacement": displacement.tolist(),
        "frames": frames_data,
    }

    # Save to JSON file
    json_file_path = "trajectory_data.json"  # Hardcoded for now, can be made configurable
    with open(json_file_path, "w") as f:
        json.dump(json_data, f, indent=4)

    return trajectories
