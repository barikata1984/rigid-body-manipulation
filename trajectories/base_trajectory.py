from dataclasses import dataclass, field

import tyro
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .optimal_excitation import generate_optimal_excitation_trajectory
from .spline_interpolation import BoundaryCondition, generate_spline_trajectory

def visualize_spline_trajectory(
    time_points: NDArray,
    qposs: NDArray,
    qvels: NDArray,
    qaccs: NDArray,
    qjerks: NDArray,
    n_dof: int,
):
    """
    Visualizes the generated spline trajectory (position, velocity, acceleration, jerk).
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle("Spline Trajectory Visualization", fontsize=16)

    labels = ["Position", "Velocity", "Acceleration", "Jerk"]
    data_arrays = [qposs, qvels, qaccs, qjerks] # Each element is (n_frames, n_dof)

    joint_groups = {
        0: range(n_dof // 2), # First half of joints (e.g., 0, 1, 2)
        1: range(n_dof // 2, n_dof) # Second half of joints (e.g., 3, 4, 5)
    }

    for row_idx, label_type in enumerate(labels): # Iterate over data types (pos, vel, acc, jerk)
        for col_idx in range(2): # Iterate over columns (left/right joint groups)
            for joint_idx in joint_groups[col_idx]: # Iterate over joints in the current group
                axes[row_idx, col_idx].plot(time_points, data_arrays[row_idx][:, joint_idx], label=f"Joint {joint_idx+1}")

            axes[row_idx, col_idx].set_ylabel(label_type)
            if row_idx == 3: # Only for the bottom-most row (Jerk)
                axes[row_idx, col_idx].set_xlabel("Time (s)")
            
            axes[row_idx, col_idx].grid(True)
            axes[row_idx, col_idx].legend() # Add legend for multiple lines on the same subplot

            # Hide x-axis tick labels for all but the bottom-most row
            if row_idx < 3:
                axes[row_idx, col_idx].tick_params(labelbottom=False)

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


@dataclass
class TrajectoryConfig:
    trajectory_type: str
    duration: float
    fps: int
    start_conditions: BoundaryCondition = field(default_factory=BoundaryCondition)
    end_conditions: BoundaryCondition = field(default_factory=BoundaryCondition)
    jointpos_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    coeffs: list[float] | None = None
    base_frequency: float = 1.0


# def generate_trajectory():
#    cfg = tyro.cli(TrajectoryConfig)
#
#    if "spline" in cfg.trajectory_type:
#        # Convert displacement and pos_offset to numpy arrays, handling 'pi' conversion
#        # displacement_converted = [pi_converter(val) for val in displacement]
#        # pos_offset_converted = [pi_converter(val) for val in jointpos_offset]
#
#        trajectory = generate_spline_trajectory(
#            trajectory_type=cfg.trajectory_type,
#            duration=cfg.duration,
#            fps=cfg.fps,
#            jointpos_offset=cfg.jointpos_offset,
#            displacement=cfg.displacement,
#        )
#    elif "optimal_excitation" == cfg.trajectory_type:
#        trajectory = generate_optimal_excitation_trajectory(
#            duration=cfg.duration,
#            fps=cfg.fps,
#            jointpos_offset=cfg.jointpos_offset,
#            coeffs=cfg.coeffs,
#            base_frequency=cfg.base_frequency,
#        )
#    else:
#        raise ValueError(f"Unknown trajectory type: {cfg.trajectory_type}")
#
#    import pdb
#
#    pdb.set_trace()


def generate_trajectory():
    cfg = tyro.cli(TrajectoryConfig)

    if "spline" in cfg.trajectory_type:
        # Convert displacement and pos_offset to numpy arrays, handling 'pi' conversion
        # displacement_converted = [pi_converter(val) for val in displacement]
        # pos_offset_converted = [pi_converter(val) for val in jointpos_offset]

        trajectory = generate_spline_trajectory(
            trajectory_type=cfg.trajectory_type,
            duration=cfg.duration,
            fps=cfg.fps,
            start_conditions=cfg.start_conditions,
            end_conditions=cfg.end_conditions,
        )

        # Visualize the generated trajectory
        n_frames = int(cfg.duration * cfg.fps)
        time_points = np.linspace(0, cfg.duration, n_frames)
        n_dof = len(cfg.start_conditions.qpos) # Assuming all qpos, qvel, qacc, qjerk have same DOF
        visualize_spline_trajectory(
            time_points,
            trajectory[:, 0, :],
            trajectory[:, 1, :],
            trajectory[:, 2, :],
            trajectory[:, 3, :],
            n_dof,
        )

    elif "optimal_excitation" == cfg.trajectory_type:
        trajectory = generate_optimal_excitation_trajectory(
            duration=cfg.duration,
            fps=cfg.fps,
            jointpos_offset=cfg.jointpos_offset,
            coeffs=cfg.coeffs,
            base_frequency=cfg.base_frequency,
        )
    else:
        raise ValueError(f"Unknown trajectory type: {cfg.trajectory_type}")

    
