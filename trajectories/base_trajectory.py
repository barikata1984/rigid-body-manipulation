from typing import Callable

import numpy as np
from numpy.typing import NDArray

from trajectories.fifth_order_spline_interpolation import get_trajectory_interpolated_with_fifth_order_spline
from omegaconf_custom_resolvers.pi_converter import pi_converter


def get_trajectory(
    trajectory_type: str,
    duration: float,
    fps: float,
    displacement: list[float],
    pos_offset: list[float],
    init_step: int = 0,
) -> Callable[[int], NDArray]:
    if trajectory_type == "spline-interpolation":
        # Convert displacement and pos_offset to numpy arrays, handling 'pi' conversion
        displacement_converted = [pi_converter(val) for val in displacement]
        pos_offset_converted = [pi_converter(val) for val in pos_offset]

        return get_trajectory_interpolated_with_fifth_order_spline(
            duration=duration,
            fps=fps,
            pos_offset=np.array(pos_offset_converted),
            displacement=np.array(displacement_converted),
            init_step=init_step,
        )
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
