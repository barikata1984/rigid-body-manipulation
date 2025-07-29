import sys
import unittest
import numpy as np
import mujoco
import tyro
from dataclasses import dataclass
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import os

from simulator import SimulatorConfig, generate_model_data
from trajectories.optimal_excitation import objective_function, generate_optimal_excitation_trajectory, generate_full_trajectory
from dynamics.dynamics import calculate_condition_number

# Define TestConfig for tyro
@dataclass
class TestConfig:
    """Configuration for the optimal excitation test."""
    object_path: str = "xml_models/targets/stanford-bunny"
    manipulator_path: str = "xml_models/manipulators/sequential"
    n_harmonics: int = 5 # Number of harmonics for the sinusoidal trajectory

# Global variable to store parsed config
_test_config: TestConfig = None

class TestOptimalExcitation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Access the global config parsed by tyro
        global _test_config
        cfg = _test_config

        print(f"--- Running Optimal Excitation Test (unittest) ---")
        print(f"Loading manipulator from: {cfg.manipulator_path}")
        print(f"Loading object from: {cfg.object_path}")

        # Mimic the configuration setup from main.py
        sim_config = SimulatorConfig(
            manipulator=cfg.manipulator_path,
            object=cfg.object_path,
            duration=1.0, # Dummy value
            fps=50, # Dummy value
            displacements=[0,0,0,0,0,0] # Dummy value
        )
        cli_config = OmegaConf.create(sim_config)

        print("Generating combined model data...")
        try:
            cls.m, cls.d, _ = generate_model_data(cli_config)
            print("Model generated successfully.")
        except Exception as e:
            print(f"Error generating model: {e}")
            raise

        cls.n_dof = cls.m.njnt
        if cls.n_dof != 6:
            raise ValueError(f"Expected 6 DoF manipulator, but model has {cls.n_dof} DoF.")

        cls.duration = 1.0
        cls.fps = 50
        cls.jointpos_offset = np.zeros(cls.n_dof)
        cls.base_frequency = 1.0
        cls.ee_body_name = "link6"
        cls.n_harmonics = cfg.n_harmonics

    def test_objective_function(self):
        # Generate dummy coeffs
        # Shape: (n_joints, n_harmonics, 2)
        coeffs_shape = (self.n_dof, self.n_harmonics, 2)
        coeffs_random = np.random.rand(*coeffs_shape) * 0.1 # Small random values
        coeffs_zeros = np.zeros(coeffs_shape)

        # Test with random coeffs
        print("\nCalculating objective function with random coeffs...")
        cond_num_random = objective_function(
            coeffs=coeffs_random,
            m=self.m,
            d=self.d,
            duration=self.duration,
            fps=self.fps,
            jointpos_offset=self.jointpos_offset,
            base_frequency=self.base_frequency,
            ee_body_name=self.ee_body_name,
        )

        self.assertIsInstance(cond_num_random, float, "Condition number should be a float.")
        self.assertGreater(cond_num_random, 0, "Condition number must be positive.")
        print(f"  Condition Number (random coeffs): {cond_num_random:.4e}")

        # Test with zero coeffs (should result in a very high condition number or error if no motion)
        print("Calculating objective function with zero coeffs...")
        cond_num_zeros = objective_function(
            coeffs=coeffs_zeros,
            m=self.m,
            d=self.d,
            duration=self.duration,
            fps=self.fps,
            jointpos_offset=self.jointpos_offset,
            base_frequency=self.base_frequency,
            ee_body_name=self.ee_body_name,
        )
        self.assertIsInstance(cond_num_zeros, float, "Condition number should be a float.")
        # For zero coeffs, the condition number should be very large (ideally inf) or positive
        self.assertGreater(cond_num_zeros, 0, "Condition number for zero coeffs must be positive.")
        print(f"  Condition Number (zero coeffs): {cond_num_zeros:.4e}")

    def test_generate_optimal_excitation_trajectory(self):
        print("\nTesting generate_optimal_excitation_trajectory (optimization loop)...")
        # Initial condition number (from random coeffs) for comparison
        coeffs_shape = (self.n_dof, self.n_harmonics, 2)
        initial_coeffs = np.random.rand(*coeffs_shape) * 0.01
        initial_cond_num = objective_function(
            coeffs=initial_coeffs,
            m=self.m,
            d=self.d,
            duration=self.duration,
            fps=self.fps,
            jointpos_offset=self.jointpos_offset,
            base_frequency=self.base_frequency,
            ee_body_name=self.ee_body_name,
        )
        print(f"  Initial Condition Number: {initial_cond_num:.4e}")

        # Run the optimization
        t_vec, qpos, qvel, qacc, _ = generate_optimal_excitation_trajectory(
            duration=self.duration,
            fps=self.fps,
            n_harmonics=self.n_harmonics,
            m=self.m,
            d=self.d,
            base_frequency=self.base_frequency,
            jointpos_offset=self.jointpos_offset,
            ee_body_name=self.ee_body_name,
        )

        # Calculate the condition number of the optimized trajectory
        optimized_trajectory = np.stack([qpos.T, qvel.T, qacc.T], axis=1)
        optimized_cond_num = calculate_condition_number(
            m=self.m,
            d=self.d,
            joint_trajectory=optimized_trajectory,
            ee_body_name=self.ee_body_name,
        )
        print(f"  Optimized Condition Number: {optimized_cond_num:.4e}")

        # Assert that the optimized condition number is less than the initial one
        # We use a tolerance because optimization might not always find a strictly better solution
        # especially with simple methods or limited iterations, but it should generally improve.
        self.assertLess(optimized_cond_num, initial_cond_num, 
                        "Optimized condition number should be less than initial.")
        self.assertIsInstance(t_vec, np.ndarray)
        self.assertIsInstance(qpos, np.ndarray)
        self.assertIsInstance(qvel, np.ndarray)
        self.assertIsInstance(qacc, np.ndarray)

    def test_generate_full_trajectory(self):
        print("\nTesting generate_full_trajectory (with transitions)...")
        main_duration = 1.0
        transition_duration = 0.5
        start_qpos = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]) # Non-zero start

        t_vec, qpos, qvel, qacc = generate_full_trajectory(
            main_duration=main_duration,
            transition_duration=transition_duration,
            fps=self.fps,
            n_harmonics=self.n_harmonics,
            m=self.m,
            d=self.d,
            base_frequency=self.base_frequency,
            start_qpos=start_qpos,
            ee_body_name=self.ee_body_name,
        )

        # --- Assertions ---
        # Check shape and type
        self.assertIsInstance(t_vec, np.ndarray)
        self.assertEqual(t_vec.ndim, 1)
        self.assertEqual(qpos.shape, (self.n_dof, t_vec.shape[0]))
        self.assertEqual(qvel.shape, (self.n_dof, t_vec.shape[0]))
        self.assertEqual(qacc.shape, (self.n_dof, t_vec.shape[0]))

        # Check boundary conditions
        np.testing.assert_allclose(qpos[:, 0], start_qpos, atol=1e-6, err_msg="Trajectory must start at the specified start_qpos.")
        np.testing.assert_allclose(qvel[:, 0], 0, atol=1e-6, err_msg="Trajectory must start with zero velocity.")
        np.testing.assert_allclose(qacc[:, 0], 0, atol=1e-6, err_msg="Trajectory must start with zero acceleration.")
        np.testing.assert_allclose(qpos[:, -1], start_qpos, atol=1e-6, err_msg="Trajectory must end at the specified start_qpos.")
        np.testing.assert_allclose(qvel[:, -1], 0, atol=1e-6, err_msg="Trajectory must end with zero velocity.")
        np.testing.assert_allclose(qacc[:, -1], 0, atol=1e-6, err_msg="Trajectory must end with zero acceleration.")

        # Check continuity at transition points
        n_trans_frames = int(transition_duration * self.fps)
        n_main_frames = int(main_duration * self.fps)

        # Transition 1 -> Main Trajectory
        t1_end_idx = n_trans_frames - 1
        main_start_idx = n_trans_frames
        np.testing.assert_allclose(qpos[:, t1_end_idx], qpos[:, main_start_idx], atol=1e-5, err_msg="Position not continuous between T1 and Main")
        np.testing.assert_allclose(qvel[:, t1_end_idx], qvel[:, main_start_idx], atol=1e-5, err_msg="Velocity not continuous between T1 and Main")
        np.testing.assert_allclose(qacc[:, t1_end_idx], qacc[:, main_start_idx], atol=1e-5, err_msg="Acceleration not continuous between T1 and Main")

        # Main Trajectory -> Transition 2
        main_end_idx = n_trans_frames + n_main_frames - 1
        t2_start_idx = n_trans_frames + n_main_frames
        np.testing.assert_allclose(qpos[:, main_end_idx], qpos[:, t2_start_idx], atol=1e-5, err_msg="Position not continuous between Main and T2")
        np.testing.assert_allclose(qvel[:, main_end_idx], qvel[:, t2_start_idx], atol=1e-5, err_msg="Velocity not continuous between Main and T2")
        np.testing.assert_allclose(qacc[:, main_end_idx], qacc[:, t2_start_idx], atol=1e-5, err_msg="Acceleration not continuous between Main and T2")

        print("  Full trajectory boundary and continuity checks passed.")

        # --- Visualization ---
        plot_trajectory(t_vec, qpos, qvel, qacc, "debug-figs/combined_optimal_excitation_trajectory.png")


def plot_trajectory(t, qpos, qvel, qacc, save_path):
    """Plots the trajectory and saves it to a file."""
    print(f"  Plotting trajectory to {save_path}...")
    n_dof = qpos.shape[0]
    fig, axes = plt.subplots(n_dof, 3, figsize=(18, 3 * n_dof), sharex=True)
    fig.suptitle('Combined Optimal Excitation Trajectory', fontsize=16)

    for i in range(n_dof):
        # Plot Position
        axes[i, 0].plot(t, qpos[i, :], label=f'q{i}')
        axes[i, 0].set_ylabel('Position (rad)')
        axes[i, 0].grid(True)
        if i == 0:
            axes[i, 0].set_title('Joint Positions')

        # Plot Velocity
        axes[i, 1].plot(t, qvel[i, :], label=f'qd{i}')
        axes[i, 1].set_ylabel('Velocity (rad/s)')
        axes[i, 1].grid(True)
        if i == 0:
            axes[i, 1].set_title('Joint Velocities')

        # Plot Acceleration
        axes[i, 2].plot(t, qacc[i, :], label=f'qdd{i}')
        axes[i, 2].set_ylabel('Acceleration (rad/s^2)')
        axes[i, 2].grid(True)
        if i == 0:
            axes[i, 2].set_title('Joint Accelerations')

    # Common X-axis label
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Plot saved successfully.")


if __name__ == '__main__':
    # Parse tyro arguments first.
    all_args = sys.argv[1:]

    tyro_args = []
    unittest_args = []
    i = 0
    while i < len(all_args):
        arg = all_args[i]
        if arg.startswith('--'):
            tyro_args.append(arg)
            if i + 1 < len(all_args) and not all_args[i+1].startswith('--'):
                tyro_args.append(all_args[i+1])
                i += 1
        else:
            unittest_args.append(arg)
        i += 1

    _test_config = tyro.cli(TestConfig, args=tyro_args)

    unittest_main_argv = [sys.argv[0]] + unittest_args

    unittest.main(argv=unittest_main_argv)