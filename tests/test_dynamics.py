import unittest
from dataclasses import dataclass

from omegaconf import OmegaConf

from dynamics.dynamics import calculate_condition_number
from simulator import SimulatorConfig, generate_model_data
from trajectories.spline_interpolation import BoundaryCondition, generate_spline_trajectory


@dataclass
class AppConfig:
    """Configuration for the dynamics test."""

    object_path: str = "xml_models/targets/stanford-bunny"
    manipulator_path: str = "xml_models/manipulators/sequential"


class TestDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = AppConfig()

        print("--- Running Dynamics Test (unittest) ---")
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

    def test_calculate_condition_number(self):
        m = self.__class__.m
        d = self.__class__.d

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
            "qpos": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
            "qvel": [0.0] * n_dof,
            "qacc": [0.0] * n_dof,
        }

        start_conditions_obj = BoundaryCondition(
            qpos=start_conditions["qpos"],
            qvel=start_conditions["qvel"],
            qacc=start_conditions["qacc"],
        )
        end_conditions_obj = BoundaryCondition(
            qpos=end_conditions["qpos"],
            qvel=end_conditions["qvel"],
            qacc=end_conditions["qacc"],
        )

        trajectory = generate_spline_trajectory(
            trajectory_type="fifth",
            duration=duration,
            fps=fps,
            start_conditions=start_conditions_obj,
            end_conditions=end_conditions_obj,
        )

        print("Calculating condition number...")
        condition_number = calculate_condition_number(m, d, trajectory)

        self.assertIsInstance(condition_number, float, "Condition number should be a float.")
        self.assertGreater(condition_number, 0, "Condition number must be positive.")

        print("\nTest Passed!")
        print(f"  - Calculated Condition Number: {condition_number:.4e}")


if __name__ == "__main__":
    unittest.main()
