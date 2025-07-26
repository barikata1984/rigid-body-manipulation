from dataclasses import dataclass, field

import tyro

# from .spline_interpolation import get_spline_trajectory


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

    import pdb

    pdb.set_trace()


#    if cfg.trajectory_type == "spline-interpolation":
#        # Convert displacement and pos_offset to numpy arrays, handling 'pi' conversion
#        # displacement_converted = [pi_converter(val) for val in displacement]
#        # pos_offset_converted = [pi_converter(val) for val in jointpos_offset]
#
#        return get_spline_trajectory(
#            duration=duration,
#            fps=fps,
#            pos_offset=np.array(jointpos_offset),
#            displacement=np.array(displacement),
#            init_step=init_step,
#        )
#    else:
#        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
#
