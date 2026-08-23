"""Command-line entry point for reproducible camera-distance calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from .camera_calibration import CameraCalibrationConfig


@dataclass
class CameraCalibrationCliConfig:
    object: str
    manipulator: str = "xml_models/manipulators/sequential"
    reset_keyframe: str | None = "initial_state"
    calibration_path: str = "camera_calibration.json"
    target_trajectory: str | None = None
    target_foreground_ratio: float = 0.10
    min_distance_factor: float = 1.0
    max_distance_factor: float = 32.0
    distance_tolerance: float = 1.0e-3
    ratio_tolerance: float = 5.0e-3
    sample_count: int = 16
    border_margin: int = 2


def main() -> None:
    cli = tyro.cli(CameraCalibrationCliConfig)
    trajectory_frames = None
    if cli.target_trajectory is not None:
        with Path(cli.target_trajectory).open() as file:
            trajectory_frames = json.load(file)["frames"]

    # Imports are local to avoid making the simulator package import its CLI
    # path while the normal simulator config is being constructed.
    from .setup import generate_model_data
    from .simulator import SimulatorConfig

    cfg = SimulatorConfig(
        object=cli.object,
        manipulator=cli.manipulator,
        reset_keyframe=cli.reset_keyframe,
    )
    cfg.camera_calibration = CameraCalibrationConfig(
        mode="calibrate",
        calibration_path=cli.calibration_path,
        target_foreground_ratio=cli.target_foreground_ratio,
        min_distance_factor=cli.min_distance_factor,
        max_distance_factor=cli.max_distance_factor,
        distance_tolerance=cli.distance_tolerance,
        ratio_tolerance=cli.ratio_tolerance,
        sample_count=cli.sample_count,
        border_margin=cli.border_margin,
    )
    generate_model_data(cfg, trajectory_frames=trajectory_frames)
    print(f"Saved camera calibration for {cli.object} to {cli.calibration_path}")


if __name__ == "__main__":
    main()
