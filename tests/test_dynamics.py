import sys
import unittest
import numpy as np
import mujoco
import tyro
from dataclasses import dataclass
from omegaconf import OmegaConf

from simulator import SimulatorConfig, generate_model_data
from dynamics.dynamics import calculate_condition_number
from trajectories.spline_interpolation import generate_spline_trajectory

# Define TestConfig for tyro
@dataclass
class TestConfig:
    """Configuration for the dynamics test."""
    object_path: str = "xml_models/targets/stanford-bunny"
    manipulator_path: str = "xml_models/manipulators/sequential"

# Global variable to store parsed config
_test_config: TestConfig = None

class TestDynamics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Access the global config parsed by tyro
        global _test_config
        cfg = _test_config

        print(f"--- Running Dynamics Test (unittest) ---")
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

    def test_calculate_condition_number(self):
        m = self.__class__.m
        d = self.__class__.d

        # Generate a 6-DOF trajectory suitable for the combined model
        duration = 1.0
        fps = 50
        n_dof = m.njnt

        if n_dof != 6:
            self.fail(f"Expected 6 DoF manipulator, but model has {n_dof} DoF.")

        print(f"Generating a test trajectory for {n_dof}-DOF manipulator...")
        start_conditions = {
            "qpos": [0.0] * n_dof,
            "qvel": [0.0] * n_dof,
            "qacc": [0.0] * n_dof,
        }
        end_conditions = {
            "qpos": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2], # A simple, non-trivial trajectory
            "qvel": [0.0] * n_dof,
            "qacc": [0.0] * n_dof,
        }

        trajectory = generate_spline_trajectory(
            duration, fps, start_conditions, end_conditions
        )

        print("Calculating condition number...")
        condition_number = calculate_condition_number(m, d, trajectory)

        # Assertions
        self.assertIsInstance(condition_number, float, "Condition number should be a float.")
        self.assertGreater(condition_number, 0, "Condition number must be positive.")

        print(f"\nTest Passed!")
        print(f"  - Calculated Condition Number: {condition_number:.4e}")


if __name__ == '__main__':
    # Separate arguments for tyro and unittest
    # tyro arguments typically start with '--'

    all_args = sys.argv[1:]

    tyro_args = []
    unittest_args = []
    i = 0
    while i < len(all_args):
        arg = all_args[i]
        if arg.startswith('--'):
            tyro_args.append(arg)
            # If it's a flag with a value, append the value too
            if i + 1 < len(all_args) and not all_args[i+1].startswith('--'):
                tyro_args.append(all_args[i+1])
                i += 1 # Skip the next argument as it's a value
        else:
            unittest_args.append(arg)
        i += 1

    # Parse tyro arguments. tyro.cli(args=...) does not modify sys.argv.
    _test_config = tyro.cli(TestConfig, args=tyro_args)

    # Prepare argv for unittest.main(). It expects argv[0] to be the program name.
    unittest_main_argv = [sys.argv[0]] + unittest_args

    # Run unittest
    unittest.main(argv=unittest_main_argv)