import dataclasses
import sys
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import tyro
import yaml
import numpy as np

from .base_trajectory import BaseTrajectory
from .spline import QuinticSplineTrajectory
from .fourier import FourierTrajectory
from .excited import ExcitedTrajectory


@dataclasses.dataclass
class SharedArgs:
    output: Optional[Path] = None
    no_plot: bool = False
    config_file: Annotated[Optional[Path], tyro.conf.arg(name="config")] = None


@dataclasses.dataclass
class SplineConfig(SharedArgs):
    duration: float = 5.0
    fps: float = 60.0
    start_pos: list[float] = dataclasses.field(default_factory=list)
    end_pos: list[float] = dataclasses.field(default_factory=list)
    start_vel: Optional[list[float]] = None
    end_vel: Optional[list[float]] = None
    start_acc: Optional[list[float]] = None
    end_acc: Optional[list[float]] = None

    def create_trajectory(self) -> QuinticSplineTrajectory:
        return QuinticSplineTrajectory(
            duration=self.duration,
            fps=self.fps,
            start_pos=self.start_pos,
            end_pos=self.end_pos,
            start_vel=self.start_vel,
            end_vel=self.end_vel,
            start_acc=self.start_acc,
            end_acc=self.end_acc,
        )


@dataclasses.dataclass
class FourierConfig(SharedArgs):
    duration: float = 5.0
    fps: int = 60
    num_joints: int = 1
    num_harmonics: int = 5
    base_freq: float = 0.1
    coefficients: Optional[dict] = None
    q0: Optional[list[float]] = None

    def create_trajectory(self) -> FourierTrajectory:
        return FourierTrajectory(
            duration=self.duration,
            fps=self.fps,
            num_joints=self.num_joints,
            num_harmonics=self.num_harmonics,
            base_freq=self.base_freq,
            coefficients=self.coefficients,
            q0=self.q0,
        )


@dataclasses.dataclass
class ExcitedConfig(SharedArgs):
    # For nested config, we need to be careful. 
    # If main_trajectory is a Union, tyro handles it.
    main_trajectory: Union[
        Annotated[SplineConfig, tyro.conf.subcommand(name="spline")],
        Annotated[FourierConfig, tyro.conf.subcommand(name="fourier")]
    ] = dataclasses.field(default_factory=lambda: SplineConfig(duration=5.0, fps=60.0))
    
    num_harmonics: int = 5
    base_freq: float = 0.1

    def create_trajectory(self) -> ExcitedTrajectory:
        main_traj = self.main_trajectory.create_trajectory()
        return ExcitedTrajectory(
            main_trajectory=main_traj,
            num_harmonics=self.num_harmonics,
            base_freq=self.base_freq,
            kinematics_func=None,
        )


# Top-level union for the CLI
TrajectoryConfig = Union[
    Annotated[SplineConfig, tyro.conf.subcommand(name="spline")],
    Annotated[FourierConfig, tyro.conf.subcommand(name="fourier")],
    Annotated[ExcitedConfig, tyro.conf.subcommand(name="excited")],
]


def main():
    # Pre-process sys.argv to inject arguments from config file if present
    argv = sys.argv[1:]
    
    # Handle --type <subcommand> by moving it to the front as a positional arg
    type_arg_idx = -1
    for i, arg in enumerate(argv):
        if arg == "--type":
            if i + 1 < len(argv):
                type_val = argv[i+1]
                # Remove --type and val
                del argv[i:i+2]
                # Insert at front
                argv.insert(0, type_val)
                type_arg_idx = i # Just to mark we did something
                break
        elif arg.startswith("--type="):
            type_val = arg.split("=", 1)[1]
            del argv[i]
            argv.insert(0, type_val)
            break

    # Check for --config in arguments
    config_path = None
    config_arg_idx = -1
    
    for i, arg in enumerate(argv):
        if arg == "--config":
            if i + 1 < len(argv):
                config_path = argv[i + 1]
                config_arg_idx = i
                break
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            # We don't strictly need index if we aren't removing it, but we are injecting
            break

    # Look for subcommand (now it should be at index 0 if it exists)
    subcommands = {"spline", "fourier", "excited"}
    subcommand_idx = -1
    for i, arg in enumerate(argv):
        if arg in subcommands:
            subcommand_idx = i
            break
            
    if config_path and subcommand_idx != -1:
        # Load YAML
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}", file=sys.stderr)
            sys.exit(1)
            
        if config_data:
            extra_args = []
            for key, value in config_data.items():
                # Skip config key itself to avoid recursion if it was in yaml (unlikely but safe)
                if key == "config": 
                    continue
                    
                # Convert key to flag
                flag = "--" + key.replace("_", "-")
                
                # Handle boolean flag quirks? Tyro usually expects --flag or --no-flag. 
                # For simplicity assume explicit values or lists.
                if isinstance(value, bool):
                     if value:
                         extra_args.append(flag)
                     else:
                         # Tyro style for False is --no-flag usually, or --flag False?
                         # Let's try --flag False if it's a bool?
                         # Tyro typically handles boolean flags as toggles.
                         # Safest for tyro is often --no-flagName for False.
                         # But tyro supports --flagName values too if configured.
                         # Let's assume standard tyro bool behavior: --flag for True. nothing for False?
                         # Actually, if default is False, --flag sets True.
                         # If default is True, --no-flag sets False.
                         # Since we don't know the default here easily, passing explicit True/False is risky with flags.
                         # Tyro supports --flagName=True/False. Let's use that.
                         extra_args.append(f"{flag}={str(value)}")
                elif isinstance(value, list) or isinstance(value, tuple):
                    extra_args.append(flag)
                    extra_args.extend(str(v) for v in value)
                else:
                    extra_args.append(flag)
                    extra_args.append(str(value))
            
            # Inject args AFTER subcommand
            # Logic: [before_subcommand] [subcommand] [EXTRA_ARGS] [rest_of_args]
            # This ensures overrides in [rest_of_args] take precedence (last arg wins in argparse/tyro usually)
            
            new_argv = argv[:subcommand_idx+1] + extra_args + argv[subcommand_idx+1:]
            
            # Debug print
            # print(f"Injected args: {new_argv}")
            
            config = tyro.cli(TrajectoryConfig, args=new_argv)
    else:
        config = tyro.cli(TrajectoryConfig)

    # Dispatch based on type
    if isinstance(config, SplineConfig):
        traj = config.create_trajectory()
    elif isinstance(config, FourierConfig):
        traj = config.create_trajectory()
    elif isinstance(config, ExcitedConfig):
        traj = config.create_trajectory()
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

    print(f"Generating {type(traj).__name__}...")

    plot_path = None
    if not config.no_plot:
        if config.output:
            plot_path = config.output.with_suffix(".png")
        else:
            plot_path = Path("output.png")
            
    json_path = config.output if config.output else None

    traj.generate(show_plot=False, plot_path=str(plot_path) if plot_path else None, json_path=str(json_path) if json_path else None)

    if plot_path:
        print(f"Plot saved to: {plot_path}")
    if json_path:
        print(f"Data saved to: {json_path}")


if __name__ == "__main__":
    main()

