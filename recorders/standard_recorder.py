from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from math import atan2, radians, tan
from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
from mujoco._structs import MjData, MjModel
from mujoco.renderer import Renderer
from omegaconf import MISSING

from utilities import get_element_id

from .base_recorder import BaseRecorderConfig


def split_file_name(name_prefix: str, suffix: str, primary_prefix: str) -> str:
    """Name a transforms file so that exactly one keeps the bare ".json" extension.

    Downstream NeRF loaders glob the dataset root for "*.json" and branch on the hit count,
    so every file except the unsplit dump of ``primary_prefix`` gets ".bak" appended: it stays
    on disk but out of the glob. ``primary_prefix`` decides which series is the one shipped.
    """
    file_name = f"{name_prefix}{suffix}.json"
    if suffix or name_prefix != primary_prefix:
        file_name += ".bak"
    return file_name


@dataclass
class StandardRecorderConfig(BaseRecorderConfig):
    target_class: str = "StandardRecorder"
    track_cam_name: str = "tracking"
    track_cam_distance_factor: float = 5.0
    fig_height: int = 800
    fig_width: int = 800
    videoname: str = "output.mp4"
    videcodec: str = "mp4v"
    dataset_dir: str = MISSING
    aabb_scale: float = MISSING  # | None = None
    fps: int = MISSING  # | None = None
    # gt_mass_distr_file_path: str = MISSING


class StandardRecorder:
    def __init__(
        self,
        cfg: StandardRecorderConfig,
        *args,
        model: MjModel,
        data: MjData,
        **kwargs,
    ) -> None:
        m = model
        d = data
        self.cam_name = cfg.track_cam_name
        self.cam_id = get_element_id(m, "camera", self.cam_name)
        self.fig_height = cfg.fig_height
        self.fig_width = cfg.fig_width
        self.cam_cx = 0.5 * self.fig_width  # horizontal center of the figures
        self.cam_cy = 0.5 * self.fig_height  # vertical center of the figures
        self.cam_fovy = radians(m.cam_fovy[self.cam_id])
        self.cam_focus = 0.5 * self.fig_height / tan(0.5 * self.cam_fovy)
        self.cam_fovx = 2 * atan2(0.5 * self.fig_width, self.cam_focus)
        # Which transforms series is shipped, i.e. keeps the bare ".json" name. Overridden to
        # "unperturbed_transforms" by main when a noise-free series is written alongside.
        self.primary_prefix = "transforms"
        self.dataset_dir = Path(cfg.dataset_dir)
        self.complete_image_dir = self.dataset_dir / "complete"
        self.mask_dir = self.dataset_dir / "masks"
        self.renderer = Renderer(m, self.fig_height, self.fig_width)
        self.aabb_scale = cfg.aabb_scale
        self.fps = cfg.fps

        self.complete_image_dir.mkdir(
            parents=True, exist_ok=True
        )  # has to be called before the videowriter instantiated
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        self.videowriter = cv2.VideoWriter(
            str(self.dataset_dir / cfg.videoname),
            cv2.VideoWriter_fourcc(*cfg.videcodec),  # type: ignore
            self.fps,  # type: ignore
            (self.fig_width, self.fig_height),
        )

        self.base_transform = {
            "date_time": datetime.now().strftime("%d/%m/%Y_%H:%M:%S"),
            "camera_angle_x": self.cam_fovx,
            "camera_angle_y": self.cam_fovy,
            "cx": self.cam_cx,
            "cy": self.cam_cy,
            "fl_x": self.cam_focus,
            "fl_y": self.cam_focus,
            "h": self.fig_height,
            "w": self.fig_width,
            "aabb_scale": self.aabb_scale,
        }

        yyyymmddhhmmss = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recursive_eval_data = {"datetime": yyyymmddhhmmss, "frames": {}}

    def render(self, d, file_name, cam_id=None):
        if cam_id is None:
            cam_id = self.cam_id

        self.renderer.update_scene(d, cam_id)
        bgr = self.renderer.render()[:, :, [2, 1, 0]]

        # Segmentation pass over the same scene: [..., 0] holds the object id and the
        # background is -1, so non-negative entries are the foreground
        self.renderer.enable_segmentation_rendering()
        segmentation = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        mask = ((segmentation[..., 0] >= 0) * 255).astype(np.uint8)

        cv2.imwrite(str(self.mask_dir / file_name), mask)  # single-channel 0/255 mask
        alpha = mask[..., np.newaxis]
        cv2.imwrite(str(self.complete_image_dir / file_name), np.append(bgr, alpha, axis=2))  # image (bgr + alpha)
        # Write a video frame
        self.videowriter.write(bgr)

    def _split(self, data, valid_ratio=0.1, test_ratio=0.1, seed=0):
        """
        Splits the indices of a list into training and testing sets.

        Args:
            data_list: The list of dictionaries.
            validation_ratio: The proportion of data to use for validation (default 0.2).
            test_ratio: The proportion of data to use for test (default 0.2).

        Returns:
            A tuple containing two lists: train_indices and test_indices.
        """
        n = len(data)
        num_test = int(n * test_ratio)
        num_valid = int(n * valid_ratio)
        num_train = n - num_test - num_valid

        # Get shuffled indices for the whole dataset
        rng = np.random.default_rng(seed)
        all_indices = list(range(n))
        rng.shuffle(all_indices)

        train = itemgetter(*all_indices[:num_train])(data)
        valid = itemgetter(*all_indices[num_train : num_train + num_valid])(data)
        test = itemgetter(*all_indices[num_train + num_valid :])(data)

        return train, valid, test

    def _process_split(
        self, frames, gt_iparams, ls_iparams, tls_iparams, split=None, name_prefix="transforms", copy_images=True
    ):
        suffix = ""

        if split:
            suffix = f"_{split}"

            if copy_images:
                split_image_dir = self.dataset_dir / split
                split_image_dir.mkdir(parents=True, exist_ok=True)

                for frame in frames:
                    # frame["file_path"] is relative to the dataset root (absolute paths are
                    # left untouched by "/", so legacy transforms keep working)
                    image_path = self.dataset_dir / frame["file_path"]
                    shutil.copy(image_path, split_image_dir / image_path.name)

        # "aabb_scale" is carried by base_transform as a top-level key, so it is deliberately
        # absent here: labels describes the 10 inertial parameters only
        labels = ["total_mass", "mx", "my", "mz", "ixx", "iyy", "izz", "ixy", "iyz", "izx"]

        split_transform = self.base_transform.copy()
        split_transform["frames"] = frames
        split_transform["labels"] = labels
        split_transform["global_gt"] = list(gt_iparams)
        split_transform["ls"] = list(ls_iparams)
        split_transform["tls"] = list(tls_iparams)

        file_name = split_file_name(name_prefix, suffix, self.primary_prefix)

        with open(self.dataset_dir / file_name, "w") as f:
            json.dump(split_transform, f, indent=2)

    def write_split_transforms(
        self, frames, gt_iparams, ls_iparams, tls_iparams, name_prefix="transforms", copy_images=True
    ):
        train_frames, valid_frames, test_frames = self._split(frames)

        kwargs = {"name_prefix": name_prefix, "copy_images": copy_images}
        self._process_split(frames, gt_iparams, ls_iparams, tls_iparams, **kwargs)
        self._process_split(train_frames, gt_iparams, ls_iparams, tls_iparams, split="train", **kwargs)
        self._process_split(valid_frames, gt_iparams, ls_iparams, tls_iparams, split="valid", **kwargs)
        self._process_split(test_frames, gt_iparams, ls_iparams, tls_iparams, split="test", **kwargs)

    def finish(self, frames, gt_iparams, ls_iparams, tls_iparams):
        self.videowriter.release()
        self.write_split_transforms(frames, gt_iparams, ls_iparams, tls_iparams)
