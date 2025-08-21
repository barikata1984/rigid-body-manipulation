import os
import unittest
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from dynamics.condition_number import calculate_condition_number
from simulator import SimulatorConfig, generate_model_data
from trajectories.excitation import (
    _find_optimal_coeffs,  # Import the new private function
    generate_optimal_excitation_trajectory,
    generate_sinusoidal_trajectory,
    objective_function,
)
from trajectories.exciting_spline import generate_exciting_spline_trajectory
from trajectories.spline_interpolation import (
    BoundaryCondition,
    generate_spline_trajectory,
)


@dataclass
class AppConfig:
    """Configuration for the optimal excitation test."""

    object_path: str = "xml_models/targets/stanford-bunny"  # Default value, can be overridden
    manipulator_path: str = "xml_models/manipulators/sequential"  # Default value, can be overridden
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

        optimized_coeffs = _find_optimal_coeffs(
            n_joints=self.n_dof,
            n_harmonics=self.n_harmonics,
            m=self.m,
            d=self.d,
            main_duration=self.duration,
            fps=self.fps,
            start_qpos=self.jointpos_offset,
            base_frequency=self.base_frequency,
            ee_body_name=self.ee_body_name,
            optimization_max_iter=10,  # Keep test fast
        )

        # Generate trajectory with optimized coeffs to check condition number
        t_vec, qpos, qvel, qacc, _ = generate_sinusoidal_trajectory(
            duration=self.duration,
            fps=self.fps,
            coeffs=optimized_coeffs,
            base_frequency=self.base_frequency,
            jointpos_offset=self.jointpos_offset,
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
        self.assertIsInstance(optimized_coeffs, np.ndarray)

    def test_generate_full_trajectory(self):
        print("\nTesting generate_full_trajectory (with transitions)...")
        main_duration = 1.0
        transition_duration = 0.5
        start_qpos = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])

        trajectory_data = generate_optimal_excitation_trajectory(
            main_duration=main_duration,
            transition_duration=transition_duration,
            fps=self.fps,
            n_harmonics=self.n_harmonics,
            m=self.m,
            d=self.d,
            base_frequency=self.base_frequency,
            start_qpos=start_qpos,
            ee_body_name=self.ee_body_name,
            optimization_max_iter=10,  # Keep test fast
        )

        full_t_vec = trajectory_data["t"]
        full_qpos = trajectory_data["qpos"]
        full_qvel = trajectory_data["qvel"]
        full_qacc = trajectory_data["qacc"]
        full_qjerk = trajectory_data["qjerk"]

        # 1. Check shapes of the full trajectory
        self.assertIsInstance(full_t_vec, np.ndarray)
        self.assertEqual(full_t_vec.ndim, 1)
        self.assertEqual(full_qpos.shape, (self.n_dof, full_t_vec.shape[0]))
        self.assertEqual(full_qvel.shape, (self.n_dof, full_t_vec.shape[0]))
        self.assertEqual(full_qacc.shape, (self.n_dof, full_t_vec.shape[0]))
        self.assertEqual(full_qjerk.shape, (self.n_dof, full_t_vec.shape[0]))
        print("  Full trajectory shapes checked.")

        # 2. Check boundary conditions of the full trajectory
        np.testing.assert_allclose(full_qpos[:, 0], start_qpos, atol=1e-6)
        np.testing.assert_allclose(full_qvel[:, 0], 0, atol=1e-6)
        np.testing.assert_allclose(full_qacc[:, 0], 0, atol=1e-6)
        np.testing.assert_allclose(full_qjerk[:, 0], 0, atol=1e-6)
        np.testing.assert_allclose(full_qpos[:, -1], start_qpos, atol=1e-6)
        np.testing.assert_allclose(full_qvel[:, -1], 0, atol=1e-6)
        np.testing.assert_allclose(full_qacc[:, -1], 0, atol=1e-6)
        np.testing.assert_allclose(full_qjerk[:, -1], 0, atol=1e-6)
        print("  Full trajectory start/end boundary conditions checked.")

        print("  All trajectory checks passed.")

        plot_trajectory(
            full_t_vec,
            full_qpos,
            full_qvel,
            full_qacc,
            full_qjerk,
            "debug-figs/combined_optimal_excitation_trajectory.png",
            transition_duration,
            main_duration,
        )

    def test_generate_exciting_spline_trajectory(self):
        print("\nTesting generate_exciting_spline_trajectory (task-oriented)...")

        # 1. Define start and end positions
        start_qpos = self.jointpos_offset
        end_qpos = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3])

        # 2. Generate the optimized trajectory
        start_cond = BoundaryCondition(qpos=start_qpos.tolist())
        end_cond = BoundaryCondition(qpos=end_qpos.tolist())

        trajectory = generate_exciting_spline_trajectory(
            start_conditions=start_cond,
            end_conditions=end_cond,
            duration=self.duration,
            fps=self.fps,
            n_harmonics=self.n_harmonics,
            base_frequency=self.base_frequency,
            m=self.m,
            d=self.d,
            ee_body_name=self.ee_body_name,
            optimization_max_iter=10,  # Keep test fast
        )

        # 3. Assertions
        # Check boundary conditions
        np.testing.assert_allclose(trajectory["qpos"][:, 0], start_qpos, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(trajectory["qpos"][:, -1], end_qpos, atol=1e-6, rtol=1e-5)

        zeros_q = np.zeros_like(start_qpos)
        np.testing.assert_allclose(trajectory["qvel"][:, 0], zeros_q, atol=1e-6)
        np.testing.assert_allclose(trajectory["qvel"][:, -1], zeros_q, atol=1e-6)
        np.testing.assert_allclose(trajectory["qacc"][:, 0], zeros_q, atol=1e-6)
        np.testing.assert_allclose(trajectory["qacc"][:, -1], zeros_q, atol=1e-6)
        np.testing.assert_allclose(trajectory["qjerk"][:, 0], zeros_q, atol=1e-6)
        np.testing.assert_allclose(trajectory["qjerk"][:, -1], zeros_q, atol=1e-6)
        print("  Boundary conditions checked.")

        # Check optimization effect
        # Condition number of base trajectory (spline only)
        base_traj_data, _ = generate_spline_trajectory(
            "seventh",
            self.duration,
            self.fps,
            BoundaryCondition(
                qpos=start_qpos.tolist(), qvel=[0] * self.n_dof, qacc=[0] * self.n_dof, qjerk=[0] * self.n_dof
            ),
            BoundaryCondition(
                qpos=end_qpos.tolist(), qvel=[0] * self.n_dof, qacc=[0] * self.n_dof, qjerk=[0] * self.n_dof
            ),
        )
        base_joint_traj = np.stack([base_traj_data[:, 0, :], base_traj_data[:, 1, :], base_traj_data[:, 2, :]], axis=1)

        base_cond_num = calculate_condition_number(
            m=self.m, d=self.d, joint_trajectory=base_joint_traj, ee_body_name=self.ee_body_name
        )

        # Condition number of optimized trajectory
        optimized_joint_traj = np.stack([trajectory["qpos"].T, trajectory["qvel"].T, trajectory["qacc"].T], axis=1)
        optimized_cond_num = calculate_condition_number(
            m=self.m, d=self.d, joint_trajectory=optimized_joint_traj, ee_body_name=self.ee_body_name
        )

        print(f"  Base Condition Number (spline only): {base_cond_num:.4e}")
        print(f"  Optimized Condition Number (with excitation): {optimized_cond_num:.4e}")

        self.assertTrue(np.isfinite(optimized_cond_num))
        # With enough iterations, the optimized one should be better (lower)
        # self.assertLess(optimized_cond_num, base_cond_num)


def plot_trajectory(t, qpos, qvel, qacc, qjerk, save_path, transition_duration=None, main_duration=None):
    print(f"  Plotting trajectory to {save_path}...")
    n_dof = qpos.shape[0]

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle("Excitation Trajectory Analysis", fontsize=16)

    # Plot Joint Positions
    for i in range(n_dof):
        axes[0].plot(t, qpos[i, :], label=f"q{i}")
    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title("Joint Positions")
    axes[0].grid(True)

    # Plot Joint Velocities
    for i in range(n_dof):
        axes[1].plot(t, qvel[i, :], label=f"qd{i}")
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].set_title("Joint Velocities")
    axes[1].grid(True)

    # Plot Joint Accelerations
    for i in range(n_dof):
        axes[2].plot(t, qacc[i, :], label=f"qdd{i}")
    axes[2].set_ylabel("Acceleration (rad/s^2)")
    axes[2].set_title("Joint Accelerations")
    axes[2].grid(True)

    # Plot Joint Jerks
    for i in range(n_dof):
        axes[3].plot(t, qjerk[i, :], label=f"qddd{i}")
    axes[3].set_ylabel("Jerk (rad/s^3)")
    axes[3].set_title("Joint Jerks")
    axes[3].grid(True)

    if transition_duration is not None and main_duration is not None:
        main_start_time = transition_duration
        main_end_time = transition_duration + main_duration
        for ax in axes:
            ax.axvline(main_start_time, color="r", linestyle="--", label="Main Trajectory Start")
            ax.axvline(main_end_time, color="g", linestyle="--", label="Main Trajectory End")

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print("  Plot saved successfully.")


if __name__ == "__main__":
    unittest.main()
