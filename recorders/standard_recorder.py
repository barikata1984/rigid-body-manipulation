from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from math import atan2, radians, tan
from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
from mujoco.renderer import Renderer
from numpy.linalg import lstsq
from omegaconf import MISSING

from regressions import total_lstsq
from utilities import get_element_id

from .base_recorder import BaseRecorderConfig


@dataclass
class StandardRecorderConfig(BaseRecorderConfig):
    target_class: str = "StandardRecorder"  # type: ignore
    track_cam_name: str = "tracking"
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
        **kwargs,
    ) -> None:
        m = kwargs["model"]
        d = kwargs["data"]
        self.cam_name = cfg.track_cam_name
        self.cam_id = get_element_id(m, "camera", self.cam_name)
        self.fig_height = cfg.fig_height
        self.fig_width = cfg.fig_width
        self.cam_cx = 0.5 * self.fig_width  # horizontal center of the figures
        self.cam_cy = 0.5 * self.fig_height  # vertical center of the figures
        self.cam_fovy = radians(m.cam_fovy[self.cam_id])
        self.cam_focus = 0.5 * self.fig_height / tan(0.5 * self.cam_fovy)
        self.cam_fovx = 2 * atan2(0.5 * self.fig_width, self.cam_focus)
        self.dataset_dir = Path(cfg.dataset_dir)
        self.complete_image_dir = self.dataset_dir / "complete"
        self.renderer = Renderer(m, self.fig_height, self.fig_width)
        self.aabb_scale = cfg.aabb_scale
        self.fps = cfg.fps

        os.makedirs(self.complete_image_dir, exist_ok=True)  # has to be called before the videowriter instantiated

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
        # Make an alpha mask to remove the white background
        alpha = np.where(np.all(bgr == 0, axis=-1), 0, 255).astype(np.uint8)[..., np.newaxis]
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
        test = itemgetter(*all_indices[num_train : num_train + num_valid])(data)

        return train, valid, test

    def _process_split(self, frames, regressors, gt_iparams, split=None):
        suffix = ""
        wrenches = []

        if split:
            suffix = f"_{split}"
            split_image_dir = self.dataset_dir / split
            split_image_dir.mkdir(parents=True, exist_ok=True)

            for frame in frames:
                wrenches.append(frame["wrench"])
                image_path = Path(frame["file_path"])
                shutil.copy(image_path, split_image_dir / image_path.name)
        else:
            for frame in frames:
                wrenches.append(frame["wrench"])

        regressors = np.reshape(regressors, (-1, 10))
        wrenches = np.reshape(wrenches, -1)
        ls_iparams = lstsq(regressors, wrenches)[0]
        tls_iparams = total_lstsq(regressors, wrenches)[0]
        labels = ["total_mass", "mx", "my", "mz", "ixx", "iyy", "izz", "ixy", "iyz", "izx", "aabb_scale"]

        split_transform = self.base_transform.copy()
        split_transform["frames"] = frames
        split_transform["labels"] = labels
        split_transform["global_gt"] = [*gt_iparams, self.aabb_scale]
        split_transform["ls"] = [*ls_iparams, np.nan]
        split_transform["tls"] = [*tls_iparams, np.nan]

        with open(self.dataset_dir / f"transforms{suffix}.json", "w") as f:
            json.dump(split_transform, f, indent=2)

    def finish(self, frames, regressors, gt_iparams):
        self.videowriter.release()

        train_frames, valid_frames, test_frames = self._split(frames)
        train_regressors, valid_regressors, test_regressors = self._split(regressors)
        train_regressors, valid_regressors, test_regressors = self._split(regressors)

        self._process_split(frames, regressors, gt_iparams)
        self._process_split(train_frames, train_regressors, gt_iparams, split="train")
        self._process_split(valid_frames, valid_regressors, gt_iparams, split="valid")
        self._process_split(test_frames, test_regressors, gt_iparams, split="test")

#        print("Tracking camera setup =======================================\n"
#             f"    Tracking camera id:         {self.id}\n"
#             f"    Image size (w x h [px]):    {self.width} x {self.height}\n"
#             f"    Focus [px]:                 {self.focus}\in"
#             f"    FoV (h, v [deg]):           {deg(self.fovx)}, {deg(self.fovy)}\n"
#             f"    Output file:                {self.output_file}")
