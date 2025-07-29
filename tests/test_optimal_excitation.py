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

    object_path: str = "xml_models/targets/stanford-bunny"
    manipulator_path: str = "xml_models/manipulators/sequential"
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
        cls.fps = 50
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

        self.assertIsInstance(t_vec, np.ndarray)
        self.assertEqual(t_vec.ndim, 1)
        self.assertEqual(qpos.shape, (self.n_dof, t_vec.shape[0]))
        self.assertEqual(qvel.shape, (self.n_dof, t_vec.shape[0]))
        self.assertEqual(qacc.shape, (self.n_dof, t_vec.shape[0]))

        np.testing.assert_allclose(qpos[:, 0], start_qpos, atol=1e-6)
        np.testing.assert_allclose(qvel[:, 0], 0, atol=1e-6)
        np.testing.assert_allclose(qacc[:, 0], 0, atol=1e-6)
        np.testing.assert_allclose(qpos[:, -1], start_qpos, atol=1e-6)
        np.testing.assert_allclose(qvel[:, -1], 0, atol=1e-6)
        np.testing.assert_allclose(qacc[:, -1], 0, atol=1e-6)

        n_trans_frames = int(transition_duration * self.fps)
        n_main_frames = int(main_duration * self.fps)

        t1_end_idx = n_trans_frames - 1
        main_start_idx = n_trans_frames
        np.testing.assert_allclose(qpos[:, t1_end_idx], qpos[:, main_start_idx], atol=1e-5)
        np.testing.assert_allclose(qvel[:, t1_end_idx], qvel[:, main_start_idx], atol=1e-5)
        np.testing.assert_allclose(qacc[:, t1_end_idx], qacc[:, main_start_idx], atol=1e-5)

        main_end_idx = n_trans_frames + n_main_frames - 1
        t2_start_idx = n_trans_frames + n_main_frames
        np.testing.assert_allclose(qpos[:, main_end_idx], qpos[:, t2_start_idx], atol=1e-5)
        np.testing.assert_allclose(qvel[:, main_end_idx], qvel[:, t2_start_idx], atol=1e-5)
        np.testing.assert_allclose(qacc[:, main_end_idx], qacc[:, t2_start_idx], atol=1e-5)

        print("  Full trajectory boundary and continuity checks passed.")

        plot_trajectory(t_vec, qpos, qvel, qacc, "debug-figs/combined_optimal_excitation_trajectory.png")


def plot_trajectory(t, qpos, qvel, qacc, save_path):
    print(f"  Plotting trajectory to {save_path}...")
    n_dof = qpos.shape[0]
    fig, axes = plt.subplots(n_dof, 3, figsize=(18, 3 * n_dof), sharex=True)
    fig.suptitle("Combined Optimal Excitation Trajectory", fontsize=16)

    for i in range(n_dof):
        axes[i, 0].plot(t, qpos[i, :], label=f"q{i}")
        axes[i, 0].set_ylabel("Position (rad)")
        axes[i, 0].grid(True)
        if i == 0:
            axes[i, 0].set_title("Joint Positions")

        axes[i, 1].plot(t, qvel[i, :], label=f"qd{i}")
        axes[i, 1].set_ylabel("Velocity (rad/s)")
        axes[i, 1].grid(True)
        if i == 0:
            axes[i, 1].set_title("Joint Velocities")

        axes[i, 2].plot(t, qacc[i, :], label=f"qdd{i}")
        axes[i, 2].set_ylabel("Acceleration (rad/s^2)")
        axes[i, 2].grid(True)
        if i == 0:
            axes[i, 2].set_title("Joint Accelerations")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print("  Plot saved successfully.")


if __name__ == "__main__":
    unittest.main()
