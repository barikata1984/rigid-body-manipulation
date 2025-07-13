import tyro
from dataclasses import dataclass, field, asdict
from typing import Optional, Union
from math import pi
from pathlib import Path
from omegaconf import OmegaConf # Still needed for YAML loading and merging
import sys # Import sys to access sys.argv

# Assuming these dataclasses are defined elsewhere,
# they are replicated here for the test.

@dataclass
class StateSpaceConfig:
    target_class: str = "StateSpace"
    epsilon: float = 1e-8
    centered: bool = True

@dataclass
class LinearQuadraticRegulatorConfig:
    target_class: str = "LinearQuadraticRegulator"
    state_space: StateSpaceConfig = field(default_factory=StateSpaceConfig)
    input_gain: Optional[list[float]] = None

@dataclass
class JointPositionPlannerConfig:
    target_class: str = "JointPositionPlanner"
    duration: Optional[float] = None
    timestep: float = -1
    pos_offset: Optional[list[float]] = None
    displacements: list[Union[float, str]] = field(default_factory=lambda: [0.2, 0.4, 0.6, 1.0 * pi, 0.3 * pi, 1.5 * pi])

@dataclass
class BasicRecorderConfig:
    target_class: str = "Logger"
    track_cam_name: str = "tracking"
    fig_height: int = 800
    fig_width: int = 800
    fps: int = 60
    videoname: str = "output.mp4"
    videcodec: str = "mp4v"
    dataset_dir: Optional[str] = None
    aabb_scale: Optional[float] = None

@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    manipulator_name: str = "sequential"
    target_name: Optional[str] = None
    reset_keyframe: str = "initial_state"
    recorder: BasicRecorderConfig = field(default_factory=BasicRecorderConfig)
    planner: JointPositionPlannerConfig = field(default_factory=JointPositionPlannerConfig)
    controller: LinearQuadraticRegulatorConfig = field(default_factory=LinearQuadraticRegulatorConfig)
    config: Optional[str] = "configurations/base.yaml" # Path to the default YAML config
    config_export_path: Optional[str] = None

def main():
    # --- Step 1: First pass with tyro to get the config file path ---
    # We define a minimal dataclass to parse only the '--config' argument.
    # We explicitly pass only the relevant arguments to tyro.cli.
    @dataclass
    class ConfigPath:
        config: Optional[str] = "configurations/base.yaml"

    # Filter sys.argv to only include --config and its value
    config_args = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--config":
            config_args.append(sys.argv[i])
            if i + 1 < len(sys.argv):
                config_args.append(sys.argv[i+1])
                i += 1 # Skip the next argument as it's the value
        i += 1

    # Parse only the config path. tyro.cli does not modify sys.argv.
    config_path_args = tyro.cli(ConfigPath, args=config_args, description="Preliminary parse for config file path.")
    yaml_file_path = config_path_args.config

    # --- Step 2: Build the base configuration (YAML over Dataclass Defaults) ---
    # Start with the standard dataclass defaults (Lowest Priority)
    base_cfg_dataclass = SimulationConfig()
    base_omegaconf = OmegaConf.structured(base_cfg_dataclass)

    # Load the YAML file if specified and exists (Middle Priority)
    if yaml_file_path and Path(yaml_file_path).is_file():
        print(f"Loading configuration from: {yaml_file_path}")
        yaml_loaded_omegaconf = OmegaConf.load(yaml_file_path)
        
        # Merge YAML config onto the dataclass defaults
        # OmegaConf.merge places later arguments on top of earlier ones.
        merged_yaml_and_defaults = OmegaConf.merge(base_omegaconf, yaml_loaded_omegaconf)
    else:
        print("No YAML config file found or specified, using dataclass defaults as base.")
        merged_yaml_and_defaults = base_omegaconf

    # Convert the merged OmegaConf object back to a dataclass instance
    # This will be used as the 'default' for the final tyro.cli call.
    base_cfg_for_tyro = OmegaConf.to_object(merged_yaml_and_defaults)

    # --- Step 3: Final pass with tyro (CLI over YAML over Dataclass Defaults) ---
    # tyro.cli will parse sys.argv again. Any arguments present in sys.argv
    # will override the values in 'base_cfg_for_tyro'.
    final_cfg = tyro.cli(
        SimulationConfig,
        default=base_cfg_for_tyro, # This provides the YAML+Defaults base
        description="Main configuration parser. Priority: CLI > YAML > Dataclass Defaults."
    )

    print("\n--- Final Configuration ---")
    print(final_cfg)
    print("\n--- Explanation ---")
    print("The final configuration above was created with the following priority:")
    print("1. Command-line arguments (highest)")
    print("2. YAML file values (loaded dynamically)")
    print("3. Dataclass default values (lowest)")


if __name__ == "__main__":
    main()



