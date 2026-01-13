from pathlib import Path
from typing import Annotated

import tyro
from omegaconf import OmegaConf

from factory import instantiate
from trajectories.excited import ExcitedTrajectoryConfig
from trajectories.fourier import FourierTrajectoryConfig

# Import from package
from trajectories.spline import QuinticSplineTrajectoryConfig

# Top-level union for the CLI
TrajectoryConfig = (
    Annotated[QuinticSplineTrajectoryConfig, tyro.conf.subcommand(name="spline")]
    | Annotated[FourierTrajectoryConfig, tyro.conf.subcommand(name="fourier")]
    | Annotated[ExcitedTrajectoryConfig, tyro.conf.subcommand(name="excited")]
)


def main():
    # Parse CLI arguments using tyro
    cli_config = tyro.cli(TrajectoryConfig)

    # Check if --config was specified
    config_path = cli_config.config

    if config_path is not None:
        # Load YAML config
        yaml_config = OmegaConf.load(config_path)

        # Create base config of the same type
        base_config = type(cli_config)()

        # Merge order: base < yaml < cli (CLI has highest priority)
        # Fields with MISSING in cli_config will be filled from yaml_config
        merged = OmegaConf.merge(base_config, yaml_config, cli_config)

        # Convert back to dataclass
        config = OmegaConf.to_object(merged)
    else:
        config = cli_config

    # Dispatch based on type
    traj = instantiate(config)

    print(f"Generating {type(traj).__name__}...")

    plot_path = None
    if not config.no_plot:
        if config.output:
            plot_path = config.output.with_suffix(".png")
        else:
            plot_path = Path("output.png")

    json_path = config.output if config.output else None

    # generate() accepts kwargs which will be passed to plot etc
    traj.generate(
        show_plot=False,
        plot_path=str(plot_path) if plot_path else None,
        json_path=str(json_path) if json_path else None,
    )

    if plot_path:
        print(f"Plot saved to: {plot_path}")
    if json_path:
        print(f"Data saved to: {json_path}")


if __name__ == "__main__":
    main()
