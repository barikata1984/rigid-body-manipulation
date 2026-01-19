"""
Trajectory Generation Script using Hydra.

Usage:
    # Default spline trajectory
    generate-trajectory

    # Select trajectory type
    generate-trajectory trajectory=fourier

    # Override parameters (flat access)
    generate-trajectory trajectory=spline duration=10.0 fps=120

    # Trajectory-specific parameters
    generate-trajectory trajectory=fourier num_harmonics=5

    # Save outputs
    generate-trajectory show_plot=true json_path=output.json plot_path=output.png
"""
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Import utilities to register OmegaConf resolvers (pi, eval)
import utilities  # noqa: F401 - import for side effect

from factory import instantiate as factory_instantiate
from trajectories.spline import SplineTrajectoryConfig
from trajectories.fourier import FourierTrajectoryConfig
from trajectories.window import WindowTrajectoryConfig
from trajectories.windowed_fourier import WindowedFourierTrajectoryConfig
from trajectories.excited import ExcitedTrajectoryConfig


# Map target class names to config classes
CONFIG_CLASS_MAP = {
    "trajectories.spline.SplineTrajectory": SplineTrajectoryConfig,
    "SplineTrajectory": SplineTrajectoryConfig,
    "trajectories.fourier.FourierTrajectory": FourierTrajectoryConfig,
    "FourierTrajectory": FourierTrajectoryConfig,
    "trajectories.window.WindowTrajectory": WindowTrajectoryConfig,
    "WindowTrajectory": WindowTrajectoryConfig,
    "trajectories.windowed_fourier.WindowedFourierTrajectory": WindowedFourierTrajectoryConfig,
    "WindowedFourierTrajectory": WindowedFourierTrajectoryConfig,
    "trajectories.excited.ExcitedTrajectory": ExcitedTrajectoryConfig,
    "ExcitedTrajectory": ExcitedTrajectoryConfig,
}

# Get absolute path to config directory
_CONF_DIR = Path(__file__).parent.parent / "configurations" / "trajectory_generation"


def load_trajectory_config(trajectory_name: str) -> DictConfig:
    """Load trajectory-specific YAML file."""
    traj_file = _CONF_DIR / f"{trajectory_name}.yaml"
    if not traj_file.exists():
        available = [f.stem for f in _CONF_DIR.glob("*.yaml") if f.stem != "config"]
        raise ValueError(f"Unknown trajectory: {trajectory_name}. Available: {available}")
    return OmegaConf.load(traj_file)


def create_config_from_dict(cfg_dict: dict):
    """Convert merged config dict to proper trajectory config dataclass."""
    target = cfg_dict.get("_target_", "SplineTrajectory")
    
    config_class = CONFIG_CLASS_MAP.get(target)
    if config_class is None:
        raise ValueError(f"Unknown target class: {target}")
    
    # Remove Hydra-specific keys
    cfg_dict.pop("_target_", None)
    cfg_dict.pop("_recursive_", None)
    cfg_dict.pop("_convert_", None)
    
    return config_class(**cfg_dict)


@hydra.main(version_base=None, config_path=str(_CONF_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate trajectory based on Hydra config."""
    # Load trajectory-specific config based on the 'trajectory' string
    trajectory_name = cfg.trajectory
    traj_specific_cfg = load_trajectory_config(trajectory_name)
    
    # Merge: trajectory-specific < top-level overrides
    # This allows `duration=10.0` to override the default
    merged = OmegaConf.to_container(traj_specific_cfg, resolve=True)
    
    # Override with top-level config values (only if not None)
    # This allows CLI params like `duration=10.0` or `num_harmonics=5` to work
    for key in cfg:
        if key == "trajectory":
            continue
        value = cfg[key]
        # Only override if the value is not None (null in YAML)
        if value is not None:
            merged[key] = value
    
    # Print the actual merged config (what will be used)
    print(f"trajectory: {trajectory_name}")
    print(OmegaConf.to_yaml(OmegaConf.create(merged)))
    
    # Create config dataclass
    config = create_config_from_dict(merged)
    
    # Instantiate trajectory
    traj = factory_instantiate(config)

    
    print(f"Generating {type(traj).__name__}...")
    
    # Get output parameters
    show_plot = cfg.get("show_plot", False)
    plot_path = cfg.get("plot_path", None)
    json_path = cfg.get("json_path", None)
    
    plot_path = str(plot_path) if plot_path else None
    json_path = str(json_path) if json_path else None
    
    # Generate
    traj.generate(
        show_plot=show_plot,
        plot_path=plot_path,
        json_path=json_path,
    )
    
    if plot_path:
        print(f"Plot saved to: {plot_path}")
    if json_path:
        print(f"Data saved to: {json_path}")


if __name__ == "__main__":
    main()
