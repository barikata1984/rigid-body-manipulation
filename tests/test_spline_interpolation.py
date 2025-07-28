import unittest
import numpy as np
import matplotlib.pyplot as plt

from trajectories.spline_interpolation import generate_spline_trajectory

class TestSplineInterpolation(unittest.TestCase):

    def test_generate_spline_trajectory(self):
        # Test case for a 6-DOF manipulator
        duration = 5.0  # seconds
        fps = 100  # Hz
        n_dof = 6

        start_conditions = {
            "qpos": [0.0] * n_dof,
            "qvel": [0.0] * n_dof,
            "qacc": [0.0] * n_dof,
        }
        end_conditions = {
            "qpos": [np.pi / 2, np.pi / 4, np.pi / 3, np.pi, np.pi / 2, np.pi],
            "qvel": [0.0] * n_dof,
            "qacc": [0.0] * n_dof,
        }

        # Generate the trajectory
        trajectory = generate_spline_trajectory(
            duration, fps, start_conditions, end_conditions, trajectory_type="fifth"
        )

        # Check the shape of the output
        self.assertEqual(trajectory.shape, (int(duration * fps), 3, n_dof))

        # Check boundary conditions
        qpos_start = trajectory[0, 0, :]
        qvel_start = trajectory[0, 1, :]
        qacc_start = trajectory[0, 2, :]
        qpos_end = trajectory[-1, 0, :]
        qvel_end = trajectory[-1, 1, :]
        qacc_end = trajectory[-1, 2, :]

        np.testing.assert_allclose(qpos_start, start_conditions["qpos"], atol=1e-6)
        np.testing.assert_allclose(qvel_start, start_conditions["qvel"], atol=1e-6)
        np.testing.assert_allclose(qacc_start, start_conditions["qacc"], atol=1e-6)
        np.testing.assert_allclose(qpos_end, end_conditions["qpos"], atol=1e-6)
        np.testing.assert_allclose(qvel_end, end_conditions["qvel"], atol=1e-6)
        np.testing.assert_allclose(qacc_end, end_conditions["qacc"], atol=1e-6)

        # Plot the results for visual inspection
        time = np.linspace(0, duration, int(duration * fps))
        qpos = trajectory[:, 0, :]
        qvel = trajectory[:, 1, :]
        qacc = trajectory[:, 2, :]

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        fig.suptitle("6-DOF Spline Trajectory Verification")

        for i in range(n_dof):
            axs[0].plot(time, qpos[:, i], label=f"Joint {i+1}")
            axs[1].plot(time, qvel[:, i], label=f"Joint {i+1}")
            axs[2].plot(time, qacc[:, i], label=f"Joint {i+1}")

        axs[0].set_ylabel("Position (rad)")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].set_ylabel("Velocity (rad/s)")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].set_ylabel("Acceleration (rad/s^2)")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        # plt.show() # Comment out to prevent blocking in non-interactive environments
        fig.savefig("debug-figs/spline_interpolation_test.png")
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()
