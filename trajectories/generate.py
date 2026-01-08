import dataclasses
from pathlib import Path
from typing import Annotated, Optional, Union

import tyro

from factory import instantiate

# Import from package
from trajectories.spline import QuinticSplineTrajectoryConfig
from trajectories.fourier import FourierTrajectoryConfig
from trajectories.excited import ExcitedTrajectoryConfig


# Top-level union for the CLI
# Since BaseTrajectoryConfig now has output/no_plot, these configs have them too.
TrajectoryConfig = Union[
    Annotated[QuinticSplineTrajectoryConfig, tyro.conf.subcommand(name="spline")],
    Annotated[FourierTrajectoryConfig, tyro.conf.subcommand(name="fourier")],
    Annotated[ExcitedTrajectoryConfig, tyro.conf.subcommand(name="excited")],
]


def main():
    # Parse CLI arguments using tyro
    # tyro automatically creates subcommands based on TrajectoryConfig Union
    config = tyro.cli(TrajectoryConfig)

    # Dispatch based on type
    # instantiate works because library Configs are passed directly
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
        json_path=str(json_path) if json_path else None
    )

    if plot_path:
        print(f"Plot saved to: {plot_path}")
    if json_path:
        print(f"Data saved to: {json_path}")


if __name__ == "__main__":
    main()
