import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import tyro
from numpy.typing import NDArray
from omegaconf import MISSING, DictConfig, OmegaConf

from omegaconf_custom_resolvers import pi_converter
from simulator.env_builder import generate_model_data

from .optimal_excitation import generate_optimal_excitation_trajectory
from .spline_interpolation import BoundaryCondition, generate_spline_trajectory

OmegaConf.register_new_resolver("pi", pi_converter)


@dataclass
class TrajectoryConfig:
    trajectory_type: str
    duration: float
    fps: int
    trajectory_config: str | None = None
    start_conditions: BoundaryCondition = MISSING
    end_conditions: BoundaryCondition = MISSING
    jointpos_offset: list[float] = field(default_factory=lambda: [0.0] * 6)
    coeffs: list[float] | None = None
    base_frequency: float = 1.0
    # New fields for optimal_excitation
    n_harmonics: int = 5
    transition_duration: float = 0.5
    manipulator_path: str = "xml_models/manipulators/sequential"
    object_path: str = "xml_models/targets/stanford-bunny"
    ee_body_name: str = "link6"
    optimization_max_iter: int = 10


def save_trajectory_to_json(
    trajectory_dict: dict,
    trajectory_type: str,
):
    output_dir = "configurations/trajectories"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{trajectory_type}.json")

    time_points = trajectory_dict["t"]
    qposs = trajectory_dict["qpos"].T  # Transpose to (n_frames, n_dof)
    qvels = trajectory_dict["qvel"].T
    qaccs = trajectory_dict["qacc"].T
    qjerks = trajectory_dict["qjerk"].T

    jointvars = []
    for i in range(len(time_points)):
        jointvars.append(
            {
                "time_point": time_points[i].item(),
                "qpos": qposs[i, :].tolist(),
                "qvel": qvels[i, :].tolist(),
                "qacc": qaccs[i, :].tolist(),
                "qjerk": qjerks[i, :].tolist(),
            }
        )

    output_data = {
        "duration": time_points[-1],
        "fps": int(1.0 / (time_points[1] - time_points[0])),
        "jointvars": jointvars,
    }

    if "excitation" in trajectory_dict:
        output_data["excitation"] = trajectory_dict["excitation"]

    with open(output_filename, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Trajectory saved to {output_filename}")


def visualize_trajectory(
    time_points: NDArray,
    qposs: NDArray,
    qvels: NDArray,
    qaccs: NDArray,
    qjerks: NDArray,
    n_dof: int,
    transition_duration: float = 0.0,
    main_duration: float = 0.0,
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

            # Add vertical lines for optimal excitation trajectory
            if transition_duration > 0 or main_duration > 0:
                axes[row_idx, col_idx].axvline(transition_duration, color='r', linestyle='--', label='Main Start')
                axes[row_idx, col_idx].axvline(transition_duration + main_duration, color='g', linestyle='--', label='Main End')
                # Update legend to include new labels
                handles, labels = axes[row_idx, col_idx].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes[row_idx, col_idx].legend(by_label.values(), by_label.keys())

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def generate_trajectory():
    cfg = tyro.cli(TrajectoryConfig)
    if cfg.trajectory_config is not None:
        yaml_cfg = OmegaConf.load(cfg.trajectory_config)
        cfg = OmegaConf.merge(yaml_cfg, cfg)  # priority: yaml > cli

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

        # Create dictionary for saving
        trajectory_dict = {
            "t": time_points,
            "qpos": qposs.T,
            "qvel": qvels.T,
            "qacc": qaccs.T,
            "qjerk": qjerks.T,
        }

    elif cfg.trajectory_type.strip() == "optimal-excitation":
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
        trajectory_dict = generate_optimal_excitation_trajectory(
            main_duration=cfg.duration,
            transition_duration=cfg.transition_duration,
            fps=cfg.fps,
            n_harmonics=cfg.n_harmonics,
            m=m,
            d=d,
            base_frequency=cfg.base_frequency,
            start_qpos=np.array(cfg.jointpos_offset),
            ee_body_name=cfg.ee_body_name,
            manipulator_path=cfg.manipulator_path,
            optimization_max_iter=cfg.optimization_max_iter,
        )
        full_qpos = trajectory_dict["qpos"]
        full_qvel = trajectory_dict["qvel"]
        full_qacc = trajectory_dict["qacc"]
        full_qjerk = trajectory_dict["qjerk"]
        full_t_vec = trajectory_dict["t"]
        qposs = full_qpos.T
        qvels = full_qvel.T
        qaccs = full_qacc.T
        qjerks = full_qjerk.T
        time_points = full_t_vec
        n_dof = full_qpos.shape[0]

    else:
        raise ValueError(f"Unknown trajectory type: {cfg.trajectory_type}")

    # 統一された可視化と保存処理
    if trajectory_dict:
        qposs = trajectory_dict["qpos"].T
        qvels = trajectory_dict["qvel"].T
        qaccs = trajectory_dict["qacc"].T
        qjerks = trajectory_dict["qjerk"].T
        time_points = trajectory_dict["t"]
        n_dof = qposs.shape[1]

        visualize_trajectory(
            time_points,
            qposs,
            qvels,
            qaccs,
            qjerks,
            n_dof,
            transition_duration=cfg.transition_duration if cfg.trajectory_type == "optimal-excitation" else 0.0,
            main_duration=cfg.duration,
        )
        save_trajectory_to_json(
            trajectory_dict, cfg.trajectory_type
        )
