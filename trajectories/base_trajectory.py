from dataclasses import dataclass, field

import tyro

from .optimal_excitation import generate_optimal_excitation_trajectory
from .spline_interpolation import generate_spline_trajectory


@dataclass
class TrajectoryConfig:
    trajectory_type: str
    duration: float
    fps: int
    jointpos_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    displacement: list[float] | None = None
    coeffs: list[float] | None = None
    base_frequency: float = 1.0


def generate_trajectory():
    cfg = tyro.cli(TrajectoryConfig)

    if "spline" in cfg.trajectory_type:
        # Convert displacement and pos_offset to numpy arrays, handling 'pi' conversion
        # displacement_converted = [pi_converter(val) for val in displacement]
        # pos_offset_converted = [pi_converter(val) for val in jointpos_offset]

        generate_spline_trajectory(
            trajectory_type=cfg.trajectory_type,
            duration=cfg.duration,
            fps=cfg.fps,
            jointpos_offset=cfg.jointpos_offset,
            displacement=cfg.displacement,
        )
    elif "optimal_excitation" == cfg.trajectory_type:
        tmp = generate_optimal_excitation_trajectory(
            duration=cfg.duration,
            fps=cfg.fps,
            jointpos_offset=cfg.jointpos_offset,
            coeffs=cfg.coeffs,
            base_frequency=cfg.base_frequency,
        )

        import pdb

        pdb.set_trace()
    else:
        raise ValueError(f"Unknown trajectory type: {cfg.trajectory_type}")
