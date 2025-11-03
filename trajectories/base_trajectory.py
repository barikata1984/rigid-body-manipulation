import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import tyro
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from omegaconf_custom_resolvers import pi_converter
from simulator.env_builder import generate_model_data

from .excitation import generate_optimal_excitation_trajectory
from .exciting_spline import generate_exciting_spline_trajectory
from .spline_interpolation import BoundaryCondition, generate_spline_trajectory

OmegaConf.register_new_resolver("pi", pi_converter)


@dataclass
class TrajectoryConfig:
    trajectory_type: str
    duration: float
    fps: int
    trajectory_config: str | None = None
    start_conditions: BoundaryCondition = field(default_factory=lambda: BoundaryCondition())
    end_conditions: BoundaryCondition = field(default_factory=lambda: BoundaryCondition())
    coeffs: list[float] | None = None
    base_frequency: float = 1.0
    n_harmonics: int = 5
    transition_duration: float = 0.5
    optimization_max_iter: int = 10
    manipulator_path: str = "xml_models/manipulators/sequential"
    object_path: str = "xml_models/targets/stanford-bunny"
    ee_body_name: str = "link6"
    seed: int | None = None


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

    if "condition_number" in trajectory_dict:
        output_data["condition_number"] = trajectory_dict["condition_number"]

    if "base_condition_number" in trajectory_dict:
        output_data["base_condition_number"] = trajectory_dict["base_condition_number"]

    if "seed" in trajectory_dict:
        output_data["seed"] = trajectory_dict["seed"]

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
    data_arrays = [qposs, qvels, qaccs, qjerks]

    joint_groups = {
        0: range(n_dof // 2),
        1: range(n_dof // 2, n_dof),
    }

    for row_idx, label_type in enumerate(labels):
        for col_idx in range(2):
            for joint_idx in joint_groups[col_idx]:
                axes[row_idx, col_idx].plot(
                    time_points, data_arrays[row_idx][:, joint_idx], label=f"Joint {joint_idx + 1}"
                )

            axes[row_idx, col_idx].set_ylabel(label_type)
            if row_idx == 3:
                axes[row_idx, col_idx].set_xlabel("Time (s)")

            axes[row_idx, col_idx].grid(True)
            axes[row_idx, col_idx].legend()

            if row_idx < 3:
                axes[row_idx, col_idx].tick_params(labelbottom=False)

            if transition_duration > 0 or main_duration > 0:
                axes[row_idx, col_idx].axvline(transition_duration, color="r", linestyle="--", label="Main Start")
                axes[row_idx, col_idx].axvline(
                    transition_duration + main_duration, color="g", linestyle="--", label="Main End"
                )
                handles, labels_legend = axes[row_idx, col_idx].get_legend_handles_labels()
                by_label = dict(zip(labels_legend, handles, strict=False))
                axes[row_idx, col_idx].legend(by_label.values(), by_label.keys())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def generate_trajectory():
    cfg = tyro.cli(TrajectoryConfig)
    if cfg.trajectory_config is not None:
        yaml_cfg = OmegaConf.load(cfg.trajectory_config)
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    trajectory_dict = None

    if cfg.trajectory_type == "exciting-spline":
        if cfg.start_conditions is None or cfg.end_conditions is None:
            raise ValueError(
                "For 'exciting-spline' trajectory, 'start_conditions' and 'end_conditions' must be specified."
            )

        m, d = None, None
        if cfg.object_path:
            model_cfg = DictConfig(
                {
                    "manipulator": cfg.manipulator_path.replace(".xml", ""),
                    "object": cfg.object_path.replace(".xml", ""),
                    "recorder": {"track_cam_name": "tracking"},
                    "reset_keyframe": None,
                }
            )
            m, d, _ = generate_model_data(model_cfg)

        seed = cfg.seed
        if seed is None:
            seed = np.random.randint(0, 1e6)

        trajectory_dict = generate_exciting_spline_trajectory(
            start_conditions=cfg.start_conditions,
            end_conditions=cfg.end_conditions,
            duration=cfg.duration,
            fps=cfg.fps,
            n_harmonics=cfg.n_harmonics,
            base_frequency=cfg.base_frequency,
            m=m,
            d=d,
            ee_body_name=cfg.ee_body_name,
            optimization_max_iter=cfg.optimization_max_iter,
            seed=seed,
        )

    elif cfg.trajectory_type == "optimal-excitation":
        m, d = None, None
        if cfg.object_path:
            model_cfg = DictConfig(
                {
                    "manipulator": cfg.manipulator_path.replace(".xml", ""),
                    "object": cfg.object_path.replace(".xml", ""),
                    "recorder": {"track_cam_name": "tracking"},
                    "reset_keyframe": None,
                }
            )
            m, d, _ = generate_model_data(model_cfg)

        trajectory_dict = generate_optimal_excitation_trajectory(
            main_duration=cfg.duration,
            transition_duration=cfg.transition_duration,
            fps=cfg.fps,
            n_harmonics=cfg.n_harmonics,
            m=m,
            d=d,
            base_frequency=cfg.base_frequency,
            start_qpos=np.array(cfg.start_conditions.qpos),
            ee_body_name=cfg.ee_body_name,
            manipulator_path=cfg.manipulator_path,
            optimization_max_iter=cfg.optimization_max_iter,
        )

    elif "spline" in cfg.trajectory_type:
        # 1. Load model data for condition number calculation
        m, d = None, None
        if cfg.object_path:
            model_cfg = DictConfig(
                {
                    "manipulator": cfg.manipulator_path.replace(".xml", ""),
                    "object": cfg.object_path.replace(".xml", ""),
                    "recorder": {"track_cam_name": "tracking"},
                    "reset_keyframe": None,
                }
            )
            m, d, _ = generate_model_data(model_cfg)

        # 2. Generate the spline trajectory and calculate condition number
        trajectory_data, condition_number = generate_spline_trajectory(
            trajectory_type=cfg.trajectory_type,
            duration=cfg.duration,
            fps=cfg.fps,
            start_conditions=cfg.start_conditions,
            end_conditions=cfg.end_conditions,
            m=m,
            d=d,
            ee_body_name=cfg.ee_body_name,
        )

        # 4. Create the final dictionary for saving and visualization
        time_points = np.linspace(0, cfg.duration, int(cfg.duration * cfg.fps))
        trajectory_dict = {
            "t": time_points,
            "qpos": trajectory_data[:, 0, :].T,
            "qvel": trajectory_data[:, 1, :].T,
            "qacc": trajectory_data[:, 2, :].T,
            "qjerk": trajectory_data[:, 3, :].T,
            "condition_number": condition_number,
        }

    else:
        raise ValueError(f"Unknown trajectory type: {cfg.trajectory_type}")

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

        save_trajectory_to_json(trajectory_dict, cfg.trajectory_type)
