import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import tyro
from numpy.typing import NDArray
from omegaconf import DictConfig

from simulator.env_builder import generate_model_data

from .optimal_excitation import generate_optimal_excitation_trajectory
from .spline_interpolation import BoundaryCondition, generate_spline_trajectory


def save_trajectory_to_json(
    qposs: NDArray,
    qvels: NDArray,
    qaccs: NDArray,
    qjerks: NDArray,
    time_points: NDArray,
    trajectory_type: str,
    duration: float,
):
    output_dir = "configurations/trajectories"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{trajectory_type}.json")

    with open(output_filename, "w") as f:
        json.dump(
            {
                "time_points": time_points.tolist(),
                "qpos": qposs.tolist(),
                "qvel": qvels.tolist(),
                "qacc": qaccs.tolist(),
                "qjerk": qjerks.tolist(),
            },
            f,
            indent=4,
        )
    print(f"Trajectory saved to {output_filename}")


@dataclass
class TrajectoryConfig:
    trajectory_type: str
    duration: float
    fps: int
    start_conditions: BoundaryCondition = field(default_factory=BoundaryCondition)
    end_conditions: BoundaryCondition = field(default_factory=BoundaryCondition)
    jointpos_offset: list[float] = field(default_factory=lambda: [0.0] * 6)
    coeffs: list[float] | None = None
    base_frequency: float = 1.0
    # New fields for optimal_excitation
    n_harmonics: int = 5
    transition_duration: float = 0.5
    manipulator_path: str = "xml_models/manipulators/sequential"
    object_path: str = "xml_models/targets/stanford-bunny"
    ee_body_name: str = "link6"


def visualize_trajectory(
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
    data_arrays = [qposs, qvels, qaccs, qjerks]  # Each element is (n_frames, n_dof)

    joint_groups = {
        0: range(n_dof // 2),  # First half of joints (e.g., 0, 1, 2)
        1: range(n_dof // 2, n_dof),  # Second half of joints (e.g., 3, 4, 5)
    }

    for row_idx, label_type in enumerate(labels):  # Iterate over data types (pos, vel, acc, jerk)
        for col_idx in range(2):  # Iterate over columns (left/right joint groups)
            for joint_idx in joint_groups[col_idx]:  # Iterate over joints in the current group
                axes[row_idx, col_idx].plot(
                    time_points, data_arrays[row_idx][:, joint_idx], label=f"Joint {joint_idx + 1}"
                )

            axes[row_idx, col_idx].set_ylabel(label_type)
            if row_idx == 3:  # Only for the bottom-most row (Jerk)
                axes[row_idx, col_idx].set_xlabel("Time (s)")

            axes[row_idx, col_idx].grid(True)
            axes[row_idx, col_idx].legend()  # Add legend for multiple lines on the same subplot

            # Hide x-axis tick labels for all but the bottom-most row
            if row_idx < 3:
                axes[row_idx, col_idx].tick_params(labelbottom=False)

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


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


def generate_trajectory(cfg: TrajectoryConfig):
    if "spline" in cfg.trajectory_type:
        trajectory_data = generate_spline_trajectory(
            trajectory_type=cfg.trajectory_type,
            duration=cfg.duration,
            fps=cfg.fps,
            start_conditions=cfg.start_conditions,
            end_conditions=cfg.end_conditions,
        )
        qposs = trajectory_data[:, 0, :]
        qvels = trajectory_data[:, 1, :]
        qaccs = trajectory_data[:, 2, :]
        qjerks = trajectory_data[:, 3, :]
        time_points = np.linspace(0, cfg.duration, int(cfg.duration * cfg.fps))
        n_dof = len(cfg.start_conditions.qpos)

    elif "optimal_excitation" == cfg.trajectory_type:
        # generate_model_data に渡すための設定オブジェクトを構築
        m, d = None, None
        if cfg.object_path:
            model_cfg = DictConfig(
                {
                    "manipulator": cfg.manipulator_path.replace(".xml", ""),  # .xml 拡張子を削除
                    "object": cfg.object_path.replace(".xml", ""),  # .xml 拡張子を削除
                    "recorder": {"track_cam_name": "tracking"},  # generate_model_data が必要とするダミー値
                    "reset_keyframe": None,  # generate_model_data が必要とするダミー値
                }
            )
            m, d, _ = generate_model_data(model_cfg)  # _ は ground_truth

        # 拡張された generate_optimal_excitation_trajectory を呼び出す
        full_t_vec, full_qpos, full_qvel, full_qacc, full_qjerk = generate_optimal_excitation_trajectory(
            main_duration=cfg.duration,
            transition_duration=cfg.transition_duration,
            fps=cfg.fps,
            n_harmonics=cfg.n_harmonics,
            m=m,
            d=d,
            base_frequency=cfg.base_frequency,
            start_qpos=np.array(cfg.jointpos_offset),
            ee_body_name=cfg.ee_body_name,
            manipulator_path=cfg.manipulator_path,  # Pass the manipulator path
        )
        qposs = full_qpos.T
        qvels = full_qvel.T
        qaccs = full_qacc.T
        qjerks = full_qjerk.T
        time_points = full_t_vec
        n_dof = full_qpos.shape[0]

    else:
        raise ValueError(f"Unknown trajectory type: {cfg.trajectory_type}")

    # 統一された可視化処理
    if qposs is not None and qvels is not None and qaccs is not None and qjerks is not None:
        visualize_trajectory(
            time_points,
            qposs,
            qvels,
            qaccs,
            qjerks,
            n_dof,
        )
        save_trajectory_to_json(
            qposs,
            qvels,
            qaccs,
            qjerks,
            time_points,
            cfg.trajectory_type,
            cfg.duration,
        )


if __name__ == "__main__":
    tyro.cli(generate_trajectory)
