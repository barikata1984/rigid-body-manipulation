import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from trajectories.spline_interpolation import generate_spline_trajectory


def plot_jerk_trajectory(t, qjerk, title, save_path, n_dof):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_dof):
        ax.plot(t, qjerk[i, :], label=f"Joint {i + 1}")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Jerk (rad/s^3)")
    ax.legend()
    ax.grid(True)
    fig.savefig(save_path)
    plt.close(fig)


class TestSeventhOrderSpline(unittest.TestCase):
    def setUp(self):
        self.duration = 1.0
        self.fps = 100
        self.n_dof = 6
        self.time_points = np.linspace(0, self.duration, int(self.duration * self.fps))

    def test_seventh_order_spline_scenario_1_start_jerk_zero(self):
        """Scenario 1: Start jerk is constrained to zero, end jerk is arbitrary."""
        print("\n--- Scenario 1: Start Jerk Zero (7th Order Spline) ---")
        start_conditions = {
            "qpos": [0.0] * self.n_dof,
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [0.0] * self.n_dof,  # Constrain start jerk to zero
        }
        end_conditions = {
            "qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [10.0] * self.n_dof,  # Arbitrary end jerk
        }

        trajectory = generate_spline_trajectory(
            self.duration, self.fps, start_conditions, end_conditions,
            trajectory_type="seventh",
        )

        qjerk = trajectory[:, 3, :].T  # Transpose to (n_dof, n_frames)

        # Assert that the start jerk is close to zero
        np.testing.assert_allclose(
            qjerk[:, 0], start_conditions["qjerk"], atol=1e-6,
            err_msg="Scenario 1: Start jerk should be zero."
        )
        # Assert that the end jerk is NOT necessarily zero (it's unconstrained in this scenario)
        self.assertFalse(
            np.allclose(qjerk[:, -1], [0.0]*self.n_dof, atol=1e-6),
            msg="Scenario 1: End jerk should NOT be zero when only start is constrained."
        )

        print(f"  Actual Start Jerk: {qjerk[:, 0].tolist()}")
        print(f"  Actual End Jerk: {qjerk[:, -1].tolist()}")

        plot_jerk_trajectory(
            self.time_points, qjerk,
            "Scenario 1: Start Jerk Zero (7th Order Spline)",
            "debug-figs/seventh_order_spline_scenario_1.png",
            self.n_dof
        )

    def test_seventh_order_spline_scenario_2_end_jerk_zero(self):
        """Scenario 2: End jerk is constrained to zero, start jerk is arbitrary."""
        print("\n--- Scenario 2: End Jerk Zero (7th Order Spline) ---")
        start_conditions = {
            "qpos": [0.0] * self.n_dof,
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [10.0] * self.n_dof,  # Arbitrary start jerk
        }
        end_conditions = {
            "qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [0.0] * self.n_dof,  # Constrain end jerk to zero
        }

        trajectory = generate_spline_trajectory(
            self.duration, self.fps, start_conditions, end_conditions,
            trajectory_type="seventh",
        )

        qjerk = trajectory[:, 3, :].T

        # Assert that the end jerk is close to zero
        np.testing.assert_allclose(
            qjerk[:, -1], end_conditions["qjerk"], atol=1e-6,
            err_msg="Scenario 2: End jerk should be zero."
        )
        # Assert that the start jerk is NOT necessarily zero (it's unconstrained in this scenario)
        self.assertFalse(
            np.allclose(qjerk[:, 0], [0.0]*self.n_dof, atol=1e-6),
            msg="Scenario 2: Start jerk should NOT be zero when only end is constrained."
        )

        print(f"  Actual Start Jerk: {qjerk[:, 0].tolist()}")
        print(f"  Actual End Jerk: {qjerk[:, -1].tolist()}")

        plot_jerk_trajectory(
            self.time_points, qjerk,
            "Scenario 2: End Jerk Zero (7th Order Spline)",
            "debug-figs/seventh_order_spline_scenario_2.png",
            self.n_dof
        )

    def test_seventh_order_spline_scenario_3_both_jerks_zero(self):
        """Scenario 3: Both start and end jerks are constrained to zero."""
        print("\n--- Scenario 3: Both Jerks Zero (7th Order Spline) ---")
        start_conditions = {
            "qpos": [0.0] * self.n_dof,
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [0.0] * self.n_dof,  # Constrain start jerk to zero
        }
        end_conditions = {
            "qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [0.0] * self.n_dof,  # Constrain end jerk to zero
        }

        trajectory = generate_spline_trajectory(
            self.duration, self.fps, start_conditions, end_conditions,
            trajectory_type="seventh",
        )

        qjerk = trajectory[:, 3, :].T

        # Assert that both start and end jerks are close to zero
        np.testing.assert_allclose(
            qjerk[:, 0], start_conditions["qjerk"], atol=1e-6,
            err_msg="Scenario 3: Start jerk should be zero."
        )
        np.testing.assert_allclose(
            qjerk[:, -1], end_conditions["qjerk"], atol=1e-6,
            err_msg="Scenario 3: End jerk should be zero."
        )

        print(f"  Actual Start Jerk: {qjerk[:, 0].tolist()}")
        print(f"  Actual End Jerk: {qjerk[:, -1].tolist()}")

        plot_jerk_trajectory(
            self.time_points, qjerk,
            "Scenario 3: Both Jerks Zero (7th Order Spline)",
            "debug-figs/seventh_order_spline_scenario_3.png",
            self.n_dof
        )

if __name__ == '__main__':
    unittest.main()