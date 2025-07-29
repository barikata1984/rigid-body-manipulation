import sys
import unittest
import numpy as np
import mujoco
import tyro
from dataclasses import dataclass
from omegaconf import OmegaConf

from simulator import SimulatorConfig, generate_model_data
from trajectories.optimal_excitation import objective_function, generate_optimal_excitation_trajectory
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
        t_vec, qpos, qvel, qacc = generate_optimal_excitation_trajectory(
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