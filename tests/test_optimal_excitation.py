import os

# As the function is in a sibling directory, we adjust the path to import it.
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trajectories.optimal_excitation import generate_sinusoidal_trajectory


class TestGenerateSinusoidalTrajectory(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for tests."""
        self.n_joints = 6
        self.n_harmonics = 5
        self.duration = 2.0  # seconds
        self.fps = 100
        self.base_frequency = 1.0

        # Generate random but deterministic coefficients and offsets for consistency
        np.random.seed(42)
        self.harmonic_coeffs = np.random.rand(self.n_joints, self.n_harmonics, 2)
        self.jointpos_offset = np.random.rand(self.n_joints)

    def test_output_shapes(self):
        """1. Test if the output arrays have the correct shapes."""
        t_vec, qpos, qvel, qacc = generate_sinusoidal_trajectory(
            harmonic_coeffs=self.harmonic_coeffs,
            jointpos_offset=self.jointpos_offset,
            base_frequency=self.base_frequency,
            duration=self.duration,
            fps=self.fps,
        )

        n_timesteps = int(self.duration * self.fps)

        self.assertEqual(t_vec.shape, (n_timesteps,))
        self.assertEqual(qpos.shape, (self.n_joints, n_timesteps))
        self.assertEqual(qvel.shape, (self.n_joints, n_timesteps))
        self.assertEqual(qacc.shape, (self.n_joints, n_timesteps))

    def test_simple_case_values(self):
        """2. Test the output values for a simple, predictable case."""
        n_j = 1
        n_h = 1
        duration = 1.0
        fps = 1000
        base_freq = 1.0

        # Let q(t) = 1.0 * sin(2*pi*t)
        coeffs = np.array([[[1.0, 0.0]]])  # p_11=1, d_11=0
        offset = np.array([0.0])

        t_vec, qpos, qvel, qacc = generate_sinusoidal_trajectory(
            harmonic_coeffs=coeffs, jointpos_offset=offset, base_frequency=base_freq, duration=duration, fps=fps
        )

        # Check values at t = 0.25s
        # q(0.25) = sin(pi/2) = 1.0
        # q_dot(0.25) = 2*pi*cos(pi/2) = 0.0
        # q_ddot(0.25) = -(2*pi)^2*sin(pi/2) = -4*pi^2
        idx = int(0.25 * fps)

        self.assertAlmostEqual(qpos[0, idx], 1.0, places=5)
        self.assertAlmostEqual(qvel[0, idx], 0.0, places=5)
        self.assertAlmostEqual(qacc[0, idx], -((2 * np.pi) ** 2), places=5)

    def test_derivative_relationships(self):
        """3. Test if numerical derivatives of outputs match for a simple case."""
        # Use a very simple, low-frequency case for this test
        n_j = 1
        n_h = 1
        duration = 2.0
        fps = 1000  # High FPS for better numerical derivative accuracy
        base_freq = 0.1  # Very low frequency

        # Let q(t) = 1.0 * sin(2*pi*0.1*t)
        coeffs = np.array([[[1.0, 0.0]]])
        offset = np.array([0.0])

        t_vec, qpos, qvel, qacc = generate_sinusoidal_trajectory(
            harmonic_coeffs=coeffs, jointpos_offset=offset, base_frequency=base_freq, duration=duration, fps=fps
        )

        joint_idx = 0

        # Numerical derivative of position should be close to velocity
        numerical_qvel = np.gradient(qpos[joint_idx], t_vec)
        self.assertTrue(
            np.allclose(numerical_qvel, qvel[joint_idx], atol=1e-2)
        )  # Looser tolerance for numerical derivative

        # Numerical derivative of velocity should be close to acceleration
        numerical_qacc = np.gradient(qvel[joint_idx], t_vec)
        self.assertTrue(
            np.allclose(numerical_qacc, qacc[joint_idx], atol=1e-1)
        )  # Looser tolerance for numerical derivative

    def test_input_validation(self):
        """4. Test for ValueError with mismatched input shapes."""
        bad_offset = np.random.rand(self.n_joints - 1)  # Mismatched number of joints

        with self.assertRaises(ValueError):
            generate_sinusoidal_trajectory(
                harmonic_coeffs=self.harmonic_coeffs,
                jointpos_offset=bad_offset,
                base_frequency=self.base_frequency,
                duration=self.duration,
                fps=self.fps,
            )


if __name__ == "__main__":
    unittest.main()
