from pathlib import Path
from typing import Annotated

import tyro
from omegaconf import OmegaConf

from factory import instantiate
from trajectories.excited import ExcitedTrajectoryConfig
from trajectories.fourier import FourierTrajectoryConfig

# Import from package
from trajectories.spline import SplineTrajectoryConfig

# Import from package
from trajectories.window import WindowTrajectoryConfig
from trajectories.windowed_fourier import WindowedFourierTrajectoryConfig

# Top-level union for the CLI
TrajectoryConfig = (
    Annotated[SplineTrajectoryConfig, tyro.conf.subcommand(name="spline")]
    | Annotated[FourierTrajectoryConfig, tyro.conf.subcommand(name="fourier")]
    | Annotated[ExcitedTrajectoryConfig, tyro.conf.subcommand(name="excited")]
    | Annotated[WindowTrajectoryConfig, tyro.conf.subcommand(name="window")]
    | Annotated[WindowedFourierTrajectoryConfig, tyro.conf.subcommand(name="windowed-fourier")]
)


def main():
    # Parse CLI arguments using tyro
    cli_config = tyro.cli(TrajectoryConfig)

    # Check if --config was specified (some configs may not have this field)
    config_path = getattr(cli_config, "config", None)

    if config_path is not None:
        # Load YAML config
        yaml_config = OmegaConf.load(config_path)

        # Create base config of the same type
        base_config = type(cli_config)()

        # Merge order: base < yaml < cli (CLI has highest priority)
        # Fields with MISSING in cli_config will be filled from yaml_config
        merged = OmegaConf.merge(base_config, yaml_config, cli_config)

        config = OmegaConf.to_object(merged)
    else:
        config = cli_config

    # Build kinematics_func for ExcitedTrajectory if model paths are provided
    kwargs = {}
    if isinstance(config, ExcitedTrajectoryConfig):
        kinematics_func = None

        # Check if model paths are specified (not None, not MISSING string "???")
        if config.manipulator and config.manipulator != "???" and config.object and config.object != "???":
            import mujoco
            import numpy as np
            from dm_control import mjcf
            from mujoco._structs import MjData, MjModel

            from dynamics import make_kinematics_func, setup_robot_dynamics_parameters
            from simulators.setup import spawn_target_object

            # Load manipulator and object directly
            manipulator_dir = Path(config.manipulator)
            manipulator_path = manipulator_dir / "manipulator.xml"
            target_dir = Path(config.object)
            target_object_path = target_dir / "object.xml"
            target_object_cad_gt_path = target_dir / "object_cad_gt.csv"

            # Spawn target object
            target_object, assets, _ = spawn_target_object(
                target_object_path, target_object_cad_gt_path, compare_cad_mujoco=False
            )

            # Load manipulator and attach object
            manipulator = mjcf.from_path(str(manipulator_path))
            attachment_site = manipulator.find("site", "attachment")
            attachment_site.attach(target_object)

            # Create model and data
            m = MjModel.from_xml_string(manipulator.to_xml_string(filename_with_hash=False), assets=assets)
            d = MjData(m)
            mujoco.mj_forward(m, d)

            # Setup dynamics parameters
            params = setup_robot_dynamics_parameters(m, d, ee_body_name=config.ee_body_name)
            pose_x_sen = params.poses.get_x_("site", "target/ft_sensor")
            kinematics_func = make_kinematics_func(params, pose_x_sen)

        if kinematics_func is not None:
            kwargs["kinematics_func"] = kinematics_func
            print("Loaded MuJoCo model and built kinematics_func for optimization.")

    # Dispatch based on type
    traj = instantiate(config, **kwargs)

    print(f"Generating {type(traj).__name__}...")

    # Use explicit paths from config
    plot_path = str(config.plot_path) if config.plot_path else None
    json_path = str(config.json_path) if config.json_path else None

    # generate() accepts kwargs which will be passed to plot etc
    traj.generate(
        show_plot=config.show_plot,
        plot_path=plot_path,
        json_path=json_path,
    )

    if plot_path:
        print(f"Plot saved to: {plot_path}")
    if json_path:
        print(f"Data saved to: {json_path}")


if __name__ == "__main__":
    main()
