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


class TestSixthOrderSpline(unittest.TestCase):
    def setUp(self):
        self.duration = 1.0
        self.fps = 100
        self.n_dof = 6
        self.time_points = np.linspace(0, self.duration, int(self.duration * self.fps))

    def test_sixth_order_spline_scenario_1_start_jerk_zero(self):
        """Scenario 1: Only start jerk is constrained to zero."""
        print("\n--- Scenario 1: Start Jerk Zero ---")
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
            "qjerk": [10.0] * self.n_dof,  # Unconstrained end jerk (arbitrary non-zero value)
        }

        trajectory = generate_spline_trajectory(
            self.duration,
            self.fps,
            start_conditions,
            end_conditions,
            trajectory_type="sixth",
            constrain_end_jerk=False,  # Constrain start jerk
        )

        qjerk = trajectory[:, 3, :].T  # Transpose to (n_dof, n_frames)

        # Assert start jerk is close to zero
        np.testing.assert_allclose(
            qjerk[:, 0], start_conditions["qjerk"], atol=1e-3,
            err_msg="Scenario 1: Start jerk should be zero."
        )
        # Assert end jerk is NOT zero (it's unconstrained)
        self.assertFalse(
            np.allclose(qjerk[:, -1], [0.0] * self.n_dof, atol=1e-3),
            msg="Scenario 1: End jerk should NOT be zero."
        )

        print(f"  Actual Start Jerk: {qjerk[:, 0].tolist()}")
        print(f"  Actual End Jerk: {qjerk[:, -1].tolist()}")

        plot_jerk_trajectory(
            self.time_points, qjerk,
            "Scenario 1: Start Jerk Zero (6th Order Spline)",
            "debug-figs/sixth_order_spline_scenario_1.png",
            self.n_dof
        )

    def test_sixth_order_spline_scenario_2_end_jerk_zero(self):
        """Scenario 2: Only end jerk is constrained to zero."""
        print("\n--- Scenario 2: End Jerk Zero ---")
        start_conditions = {
            "qpos": [0.0] * self.n_dof,
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [10.0] * self.n_dof,  # Unconstrained start jerk (arbitrary non-zero value)
        }
        end_conditions = {
            "qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [0.0] * self.n_dof,  # Constrain end jerk to zero
        }

        trajectory = generate_spline_trajectory(
            self.duration,
            self.fps,
            start_conditions,
            end_conditions,
            trajectory_type="sixth",
            constrain_end_jerk=True,  # Constrain end jerk
        )

        qjerk = trajectory[:, 3, :].T

        # Assert end jerk is close to zero
        np.testing.assert_allclose(
            qjerk[:, -1], end_conditions["qjerk"], atol=1e-3,
            err_msg="Scenario 2: End jerk should be zero."
        )
        # Assert start jerk is NOT zero (it's unconstrained)
        self.assertFalse(
            np.allclose(qjerk[:, 0], [0.0] * self.n_dof, atol=1e-3),
            msg="Scenario 2: Start jerk should NOT be zero."
        )

        print(f"  Actual Start Jerk: {qjerk[:, 0].tolist()}")
        print(f"  Actual End Jerk: {qjerk[:, -1].tolist()}")

        plot_jerk_trajectory(
            self.time_points, qjerk,
            "Scenario 2: End Jerk Zero (6th Order Spline)",
            "debug-figs/sixth_order_spline_scenario_2.png",
            self.n_dof
        )

    def test_sixth_order_spline_scenario_3_both_jerks_zero(self):
        """Scenario 3: Attempt to constrain both start and end jerks to zero."""
        print("\n--- Scenario 3: Both Jerks Zero (Expected Failure) ---")
        start_conditions = {
            "qpos": [0.0] * self.n_dof,
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [0.0] * self.n_dof,  # Attempt to constrain start jerk to zero
        }
        end_conditions = {
            "qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "qvel": [0.0] * self.n_dof,
            "qacc": [0.0] * self.n_dof,
            "qjerk": [0.0] * self.n_dof,  # Attempt to constrain end jerk to zero
        }

        # Try constraining start jerk (constrain_end_jerk=False)
        trajectory_start_constrained = generate_spline_trajectory(
            self.duration,
            self.fps,
            start_conditions,
            end_conditions,
            trajectory_type="sixth",
            constrain_end_jerk=False,
        )
        qjerk_start_constrained = trajectory_start_constrained[:, 3, :].T

        print(f"  (Constraining Start Jerk) Actual Start Jerk: {qjerk_start_constrained[:, 0].tolist()}")
        print(f"  (Constraining Start Jerk) Actual End Jerk: {qjerk_start_constrained[:, -1].tolist()}")

        # Assert start jerk is close to zero
        np.testing.assert_allclose(
            qjerk_start_constrained[:, 0], start_conditions["qjerk"], atol=1e-3,
            err_msg="Scenario 3: Start jerk should be zero when constrained."
        )
        # Assert end jerk is NOT zero (it's unconstrained in this call)
        self.assertFalse(
            np.allclose(qjerk_start_constrained[:, -1], [0.0] * self.n_dof, atol=1e-3),
            msg="Scenario 3: End jerk should NOT be zero when start is constrained."
        )

        plot_jerk_trajectory(
            self.time_points, qjerk_start_constrained,
            "Scenario 3a: Both Jerks Zero (Constraining Start) - 6th Order Spline",
            "debug-figs/sixth_order_spline_scenario_3a.png",
            self.n_dof
        )

        # Try constraining end jerk (constrain_end_jerk=True)
        trajectory_end_constrained = generate_spline_trajectory(
            self.duration,
            self.fps,
            start_conditions,
            end_conditions,
            trajectory_type="sixth",
            constrain_end_jerk=True,
        )
        qjerk_end_constrained = trajectory_end_constrained[:, 3, :].T

        print(f"  (Constraining End Jerk) Actual Start Jerk: {qjerk_end_constrained[:, 0].tolist()}")
        print(f"  (Constraining End Jerk) Actual End Jerk: {qjerk_end_constrained[:, -1].tolist()}")

        # Assert end jerk is close to zero
        np.testing.assert_allclose(
            qjerk_end_constrained[:, -1], end_conditions["qjerk"], atol=1e-3,
            err_msg="Scenario 3: End jerk should be zero when constrained."
        )
        # Assert start jerk is NOT zero (it's unconstrained in this call)
        self.assertFalse(
            np.allclose(qjerk_end_constrained[:, 0], [0.0] * self.n_dof, atol=1e-3),
            msg="Scenario 3: Start jerk should NOT be zero when end is constrained."
        )

        plot_jerk_trajectory(
            self.time_points, qjerk_end_constrained,
            "Scenario 3b: Both Jerks Zero (Constraining End) - 6th Order Spline",
            "debug-figs/sixth_order_spline_scenario_3b.png",
            self.n_dof
        )


if __name__ == '__main__':
    unittest.main()