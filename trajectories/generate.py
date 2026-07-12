import dataclasses
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import tyro
from omegaconf import OmegaConf

from factory import instantiate
from trajectories.catalog import append_catalog_entry
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

# Reverse lookup: config type -> CLI subcommand name
SUBCOMMAND_BY_TYPE = {
    SplineTrajectoryConfig: "spline",
    FourierTrajectoryConfig: "fourier",
    ExcitedTrajectoryConfig: "excited",
    WindowTrajectoryConfig: "window",
    WindowedFourierTrajectoryConfig: "windowed-fourier",
}

# Default root for auto-generated trajectory outputs (git-tracked, shared across machines)
DEFAULT_OUTPUT_ROOT = Path("configurations/trajectories")


def _explicitly_set_cli_keys(cli_config) -> set[str]:
    """Return field names whose CLI flag appears in sys.argv.

    Uses tyro's default kebab-case flag naming (``max_iter`` -> ``--max-iter``),
    treating both the positive flag and the ``--no-`` boolean negation as
    explicit. Does not handle nested subcommand prefixes or custom aliases;
    sufficient for the flat TrajectoryConfig dataclasses here.
    """
    argv_tokens = set(sys.argv[1:])
    explicit = set()
    for key in vars(cli_config):
        flag = f"--{key.replace('_', '-')}"
        neg_flag = f"--no-{key.replace('_', '-')}"
        if flag in argv_tokens or neg_flag in argv_tokens:
            explicit.add(key)
    return explicit


def main():
    # Force line-buffered stdout so per-iteration optimizer prints are visible
    # when the output is redirected to a file (Python defaults to block buffering
    # for non-tty stdout, which delays visibility of the SLSQP callback progress).
    sys.stdout.reconfigure(line_buffering=True)

    # Parse CLI arguments using tyro
    cli_config = tyro.cli(TrajectoryConfig)

    # Check if --config was specified (some configs may not have this field)
    config_path = getattr(cli_config, "config", None)

    if config_path is not None:
        # Load YAML config
        yaml_config = OmegaConf.load(config_path)

        # Create base config of the same type
        base_config = type(cli_config)()

        # Build CLI overrides from fields explicitly passed on the command line,
        # so that an explicit CLI value always wins even if it equals the default.
        explicit_keys = _explicitly_set_cli_keys(cli_config)
        cli_overrides = {key: getattr(cli_config, key) for key in explicit_keys}

        # Merge order: base < yaml < cli_overrides
        merged = OmegaConf.merge(base_config, yaml_config, cli_overrides)

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

    subcommand = SUBCOMMAND_BY_TYPE[type(config)]
    now = datetime.now()

    # If neither output path is set, auto-generate a timestamped output directory.
    # An explicit CLI/YAML path always wins and is left untouched.
    if plot_path is None and json_path is None:
        output_dir = DEFAULT_OUTPUT_ROOT / f"{subcommand}_{now.strftime('%Y%m%d_%H%M%S')}"
        json_path = str(output_dir / "trajectory.json")
        plot_path = str(output_dir / "trajectory.png")
    else:
        # Record the directory containing whichever path was explicitly set.
        output_dir = Path(json_path or plot_path).parent

    # generate() accepts kwargs which will be passed to plot etc
    traj.generate(
        show_plot=config.show_plot,
        plot_path=plot_path,
        json_path=json_path,
        metadata={"subcommand": subcommand, "generation_config": dataclasses.asdict(config)},
    )

    if plot_path:
        print(f"Plot saved to: {plot_path}")
    if json_path:
        print(f"Data saved to: {json_path}")

    append_catalog_entry(
        timestamp=now.isoformat(timespec="seconds"),
        subcommand=subcommand,
        output_dir=str(output_dir),
        config=config,
        condition_number=getattr(traj, "final_condition_number", None),
    )


if __name__ == "__main__":
    main()
