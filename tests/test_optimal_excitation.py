import os
import unittest
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from dynamics.dynamics import calculate_condition_number
from simulator import SimulatorConfig, generate_model_data
from trajectories.optimal_excitation import (
    generate_full_trajectory,
    generate_optimal_excitation_trajectory,
    objective_function,
)


@dataclass
class AppConfig:
    """Configuration for the optimal excitation test."""

    object_path: str = "xml_models/targets/stanford-bunny" # Default value, can be overridden
    manipulator_path: str = "xml_models/manipulators/sequential" # Default value, can be overridden
    n_harmonics: int = 5


class TestOptimalExcitation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = AppConfig()

        print("--- Running Optimal Excitation Test (unittest) ---")
        print(f"Loading manipulator from: {cfg.manipulator_path}")
        print(f"Loading object from: {cfg.object_path}")

        sim_config = SimulatorConfig(
            manipulator=cfg.manipulator_path,
            object=cfg.object_path,
            duration=1.0,
            fps=50,
            displacements=[0, 0, 0, 0, 0, 0],
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
        cls.fps = 100
        cls.jointpos_offset = np.zeros(cls.n_dof)
        cls.base_frequency = 1.0
        cls.ee_body_name = "link6"
        cls.n_harmonics = cfg.n_harmonics

    def test_objective_function(self):
        coeffs_shape = (self.n_dof, self.n_harmonics, 2)
        coeffs_random = np.random.rand(*coeffs_shape) * 0.1
        coeffs_zeros = np.zeros(coeffs_shape)

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

        self.assertIsInstance(cond_num_random, float)
        self.assertGreater(cond_num_random, 0)
        print(f"  Condition Number (random coeffs): {cond_num_random:.4e}")

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
        self.assertIsInstance(cond_num_zeros, float)
        self.assertGreater(cond_num_zeros, 0)
        print(f"  Condition Number (zero coeffs): {cond_num_zeros:.4e}")

    def test_generate_optimal_excitation_trajectory(self):
        print("\nTesting generate_optimal_excitation_trajectory (optimization loop)...")
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

        t_vec, qpos, qvel, qacc, qjerk, _ = generate_optimal_excitation_trajectory(
            duration=self.duration,
            fps=self.fps,
            n_harmonics=self.n_harmonics,
            m=self.m,
            d=self.d,
            base_frequency=self.base_frequency,
            jointpos_offset=self.jointpos_offset,
            ee_body_name=self.ee_body_name,
        )

        optimized_trajectory = np.stack([qpos.T, qvel.T, qacc.T], axis=1)
        optimized_cond_num = calculate_condition_number(
            m=self.m,
            d=self.d,
            joint_trajectory=optimized_trajectory,
            ee_body_name=self.ee_body_name,
        )
        print(f"  Optimized Condition Number: {optimized_cond_num:.4e}")

        self.assertLess(optimized_cond_num, initial_cond_num)
        self.assertIsInstance(t_vec, np.ndarray)
        self.assertIsInstance(qpos, np.ndarray)
        self.assertIsInstance(qvel, np.ndarray)
        self.assertIsInstance(qacc, np.ndarray)

    def test_generate_full_trajectory(self):
        print("\nTesting generate_full_trajectory (with transitions)...")
        main_duration = 1.0
        transition_duration = 0.5
        start_qpos = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])

        (
            full_t_vec, full_qpos, full_qvel, full_qacc, full_qjerk, # Added full_qjerk
            t1_qpos, t1_qvel, t1_qacc, t1_qjerk, # Added t1_qjerk
            main_qpos, main_qvel, main_qacc, main_qjerk, # Added main_qjerk
            t2_qpos, t2_qvel, t2_qacc, t2_qjerk, # Added t2_qjerk
        ) = generate_full_trajectory(
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

        # 1. Check shapes of the full trajectory
        self.assertIsInstance(full_t_vec, np.ndarray)
        self.assertEqual(full_t_vec.ndim, 1)
        self.assertEqual(full_qpos.shape, (self.n_dof, full_t_vec.shape[0]))
        self.assertEqual(full_qvel.shape, (self.n_dof, full_t_vec.shape[0]))
        self.assertEqual(full_qacc.shape, (self.n_dof, full_t_vec.shape[0]))
        self.assertEqual(full_qjerk.shape, (self.n_dof, full_t_vec.shape[0])) # Added qjerk shape check
        print("  Full trajectory shapes checked.")

        # 2. Check boundary conditions of the full trajectory
        np.testing.assert_allclose(full_qpos[:, 0], start_qpos, atol=1e-6)
        np.testing.assert_allclose(full_qvel[:, 0], 0, atol=1e-6)
        np.testing.assert_allclose(full_qacc[:, 0], 0, atol=1e-6)
        np.testing.assert_allclose(full_qjerk[:, 0], 0, atol=1e-6) # Added qjerk boundary check
        np.testing.assert_allclose(full_qpos[:, -1], start_qpos, atol=1e-6)
        np.testing.assert_allclose(full_qvel[:, -1], 0, atol=1e-6)
        np.testing.assert_allclose(full_qacc[:, -1], 0, atol=1e-6)
        np.testing.assert_allclose(full_qjerk[:, -1], 0, atol=1e-6) # Added qjerk boundary check
        print("  Full trajectory start/end boundary conditions checked.")

        # 3. Check continuity at segment boundaries (before concatenation)
        # Transition 1 (t1) end should match Main trajectory (main) start
        np.testing.assert_allclose(t1_qpos[:, -1], main_qpos[:, 0], atol=1e-6, err_msg="t1_qpos end does not match main_qpos start")
        np.testing.assert_allclose(t1_qvel[:, -1], main_qvel[:, 0], atol=1e-6, err_msg="t1_qvel end does not match main_qvel start")
        np.testing.assert_allclose(t1_qacc[:, -1], main_qacc[:, 0], atol=1e-6, err_msg="t1_qacc end does not match main_qacc start")
        np.testing.assert_allclose(t1_qjerk[:, -1], main_qjerk[:, 0], atol=1e-6, err_msg="t1_qjerk end does not match main_qjerk start") # Added qjerk continuity check
        print("  Transition 1 to Main trajectory continuity checked.")

        # Main trajectory (main) end should match Transition 2 (t2) start
        np.testing.assert_allclose(main_qpos[:, -1], t2_qpos[:, 0], atol=1e-6, err_msg="main_qpos end does not match t2_qpos start")
        np.testing.assert_allclose(main_qvel[:, -1], t2_qvel[:, 0], atol=1e-6, err_msg="main_qvel end does not match t2_qvel start")
        np.testing.assert_allclose(main_qacc[:, -1], t2_qacc[:, 0], atol=1e-6, err_msg="main_qacc end does not match t2_qacc start")
        np.testing.assert_allclose(main_qjerk[:, -1], t2_qjerk[:, 0], atol=1e-3, err_msg="main_qjerk end does not match t2_qjerk start") # Added qjerk continuity check
        print("  Main trajectory to Transition 2 continuity checked.")

        print("  All trajectory checks passed.")

        plot_trajectory(full_t_vec, full_qpos, full_qvel, full_qacc, full_qjerk, # Added full_qjerk
                        "debug-figs/combined_optimal_excitation_trajectory.png",
                        transition_duration, main_duration)


def plot_trajectory(t, qpos, qvel, qacc, qjerk, save_path, transition_duration, main_duration):
    print(f"  Plotting trajectory to {save_path}...")
    n_dof = qpos.shape[0]

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True) # Changed to 4 rows for pos, vel, acc, jerk
    fig.suptitle("Combined Optimal Excitation Trajectory", fontsize=16)

    # Calculate main trajectory start and end times
    main_start_time = transition_duration
    main_end_time = transition_duration + main_duration

    # Plot Joint Positions
    for i in range(n_dof):
        axes[0].plot(t, qpos[i, :], label=f"q{i}")
    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title("Joint Positions")
    axes[0].grid(True)
    axes[0].axvline(main_start_time, color='r', linestyle='--', label='Main Trajectory Start')
    axes[0].axvline(main_end_time, color='g', linestyle='--', label='Main Trajectory End')
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.2, 1)) # Adjust legend position

    # Plot Joint Velocities
    for i in range(n_dof):
        axes[1].plot(t, qvel[i, :], label=f"qd{i}")
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].set_title("Joint Velocities")
    axes[1].grid(True)
    axes[1].axvline(main_start_time, color='r', linestyle='--')
    axes[1].axvline(main_end_time, color='g', linestyle='--')
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.2, 1)) # Adjust legend position

    # Plot Joint Accelerations
    for i in range(n_dof):
        axes[2].plot(t, qacc[i, :], label=f"qdd{i}")
    axes[2].set_ylabel("Acceleration (rad/s^2)")
    axes[2].set_title("Joint Accelerations")
    axes[2].grid(True)
    axes[2].axvline(main_start_time, color='r', linestyle='--')
    axes[2].axvline(main_end_time, color='g', linestyle='--')
    axes[2].legend(loc='upper right', bbox_to_anchor=(1.2, 1)) # Adjust legend position

    # Plot Joint Jerks # Added Jerk plot
    for i in range(n_dof):
        axes[3].plot(t, qjerk[i, :], label=f"qddd{i}")
    axes[3].set_ylabel("Jerk (rad/s^3)")
    axes[3].set_title("Joint Jerks")
    axes[3].grid(True)
    axes[3].axvline(main_start_time, color='r', linestyle='--')
    axes[3].axvline(main_end_time, color='g', linestyle='--')
    axes[3].legend(loc='upper right', bbox_to_anchor=(1.2, 1)) # Adjust legend position

    for ax in axes: # All subplots share x-axis, so only last one needs xlabel
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print("  Plot saved successfully.")


if __name__ == "__main__":
    unittest.main()
