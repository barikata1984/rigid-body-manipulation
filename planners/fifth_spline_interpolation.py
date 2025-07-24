from collections.abc import Callable

import numpy as np
from numpy import linalg as la
from numpy.typing import ArrayLike, NDArray


def get_trajectory_interpolated_with_fifth_spline(
    displacement: ArrayLike,  # [m, m, m, rad, rad, rad]
    pos_offset: ArrayLike,  # [m, m, m, rad, rad, rad]
    timestep: float,  # [s]
    n_steps: int,  # [steps]
    init_step: int = 0,
) -> Callable[[int], NDArray]:
    # Set the time window
    t_s = init_step
    t_e = t_s + n_steps

    # Define normalized boundaries of a spline
    normalized_bounds = np.array(
        [
            0,  # start pos
            1,  # end pos
            0,  # start vel
            0,  # end vel
            0,  # start acc
            0,  # end acc
        ]
    )

    # define the parameter matrix for a differentiated fifth-order polynomial
    spline_matrix = np.array(
        [
            [t_s**5, t_s**4, t_s**3, t_s**2, t_s, 1],
            [t_e**5, t_e**4, t_e**3, t_e**2, t_e, 1],
            [5 * t_s**4, 4 * t_s**3, 3 * t_s**2, 2 * t_s**1, 1, 0],
            [5 * t_e**4, 4 * t_e**3, 3 * t_e**2, 2 * t_e**1, 1, 0],
            [20 * t_s**3, 12 * t_s**2, 6 * t_s**1, 2, 0, 0],
            [20 * t_e**3, 12 * t_e**2, 6 * t_e**1, 2, 0, 0],
        ],
        dtype=float,
    )

    # compute the constants of the fifth-order spline
    coeffs = la.solve(spline_matrix, normalized_bounds).squeeze()

    displacement = np.array(displacement)
    pos_offset = np.array(pos_offset)

    def plan(step: int):
        fifth = np.array([step**i for i in range(5, -1, -1)])
        fourth = np.array([step**i * (i + 1) for i in range(4, -1, -1)])
        third = np.array([step**i * (i + 1) * (i + 2) for i in range(3, -1, -1)])

        pos = displacement * np.dot(coeffs[:], fifth) + pos_offset
        vel = displacement * np.dot(coeffs[:-1], fourth) / timestep
        acc = displacement * np.dot(coeffs[:-2], third) / timestep**2

        return np.array([pos, vel, acc])

    return plan
