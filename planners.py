import numpy as np
from numpy import linalg as la


def traj_5th_spline(
        start: np.ndarray,
        goal: np.ndarray,
        timestep: int,
        n_steps: int,
        init_step: int = 0) -> np.ndarray:

    # Set the time window
    t_s = init_step
    t_e = t_s + n_steps

    # Define normalized boundaries of a spline
    normd_bounds = np.array([
        0, 1,  # start/end pos
        0, 0,  # start/end vel 0 to make the motion smoooth
        0, 0])  # set start/end acc at 0 ↑

    # define the parameter matrix for a differentiated fifth-order polynomial
    spline_matrix = np.array([
        [t_s**5, t_s**4, t_s**3, t_s**2, t_s, 1],
        [t_e**5, t_e**4, t_e**3, t_e**2, t_e, 1],
        [5 * t_s**4, 4 * t_s**3, 3 * t_s**2, 2 * t_s**1, 1, 0],
        [5 * t_e**4, 4 * t_e**3, 3 * t_e**2, 2 * t_e**1, 1, 0],
        [20 * t_s**3, 12 * t_s**2, 6 * t_s**1, 2, 0, 0],
        [20 * t_e**3, 12 * t_e**2, 6 * t_e**1, 2, 0, 0]],
        dtype=float)

    # compute the constants of the fifth-order spline
    coeffs = la.solve(spline_matrix, normd_bounds).squeeze()

    def planner(step: int):
        fifth = np.array([step**i for i in range(5, -1, -1)])
        fourth = np.array([step**i * (i + 1) for i in range(4, -1, -1)])
        third = np.array([
            step**i * (i + 1) * (i + 2)for i in range(3, -1, -1)])

        pos = (goal - start) * np.dot(coeffs[:], fifth) + start
        vel = (goal - start) * np.dot(coeffs[:-1], fourth) / timestep
        acc = (goal - start) * np.dot(coeffs[:-2], third) / timestep**2

        return np.array([pos, vel, acc])

    return planner
