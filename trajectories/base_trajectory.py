from dataclasses import dataclass, field

import numpy as np
import tyro

from .spline_interpolation import generate_spline_trajectory


@dataclass
class TrajectoryConfig:
    trajectory_type: str
    duration: float
    fps: float
    displacement: list[float]
    jointpos_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    init_step: int = 0


def generate_trajectory():
    cfg = tyro.cli(TrajectoryConfig)

    if cfg.trajectory_type == "spline":
        # Convert displacement and pos_offset to numpy arrays, handling 'pi' conversion
        # displacement_converted = [pi_converter(val) for val in displacement]
        # pos_offset_converted = [pi_converter(val) for val in jointpos_offset]

        generate_spline_trajectory(
            duration=cfg.duration,
            fps=cfg.fps,
            displacement=np.array(cfg.displacement),
            jointpos_offset=np.array(cfg.jointpos_offset),
            init_step=cfg.init_step,
        )
    else:
        raise ValueError(f"Unknown trajectory type: {cfg.trajectory_type}")
