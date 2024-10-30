import os
import shutil
from datetime import datetime
from dataclasses import dataclass
from math import atan2, radians, tan
from operator import itemgetter
from pathlib import Path

import cv2
import json
import numpy as np
import pandas as pd
from mujoco._structs import MjData, MjModel
from mujoco.renderer import Renderer
from omegaconf import MISSING

#from main import Scorer
from utilities import get_element_id


@dataclass
class LoggerConfig:
    target_class: str = "Logger"
    track_cam_name: str = "tracking"
    fig_height: int = 800
    fig_width: int = 800
    fps: int = 60
    videoname: str = "output.mp4"
    videcodec: str = "mp4v"
    dataset_dir: str = MISSING
    aabb_scale: float = MISSING
    #gt_mass_distr_file_path: str = MISSING


class Logger:
    def __init__(self,
                 cfg: LoggerConfig,
                 m: MjModel,
                 d: MjData,
                 ) -> None:
        self.cam_name = cfg.track_cam_name
        self.cam_id = get_element_id(m, "camera", self.cam_name)
        self.fig_height = cfg.fig_height
        self.fig_width = cfg.fig_width
        self.cam_fovy = radians(m.cam_fovy[self.cam_id])
        self.cam_focus = self.fig_height / tan(.5 * self.cam_fovy)
        self.cam_fovx = 2 * atan2(self.fig_width, self.cam_focus)
        self.fps = cfg.fps
        self.dataset_dir = Path(cfg.dataset_dir)
        self.complete_image_dir = self.dataset_dir / "complete"
        self.renderer = Renderer(m, self.fig_height, self.fig_width)
        self.aabb_scale = cfg.aabb_scale

        os.makedirs(self.complete_image_dir, exist_ok=True)  # not sure but should be called before
                                                    # the videowriter is instantiated

        self.videowriter = cv2.VideoWriter(
            str(self.dataset_dir / cfg.videoname),
            cv2.VideoWriter_fourcc(*cfg.videcodec),
            self.fps,
            (self.fig_width, self.fig_height),
        )

        self.base_transform = dict(
            date_time=datetime.now().strftime("%d/%m/%Y_%H:%M:%S"),
            camera_angle_x=self.cam_fovx,
            aabb_scale=self.aabb_scale,
        )

    def render(self, d, file_name, cam_id=None):
        if cam_id is None:
            cam_id = self.cam_id

        self.renderer.update_scene(d, cam_id)
        bgr = self.renderer.render()[:, :, [2, 1, 0]]
        # Make an alpha mask to remove the white background
        alpha = np.where(np.all(bgr == 0, axis=-1), 0, 255)[..., np.newaxis]
        cv2.imwrite(str(self.complete_image_dir / file_name),
                    np.append(bgr, alpha, axis=2))  # image (bgr + alpha)
        # Write a video frame
        self.videowriter.write(bgr)

    def _split(self, data, valid_ratio=0.15, test_ratio=0.15, seed=0):
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
        valid = itemgetter(*all_indices[num_train:num_train+num_valid])(data)
        test = itemgetter(*all_indices[num_train:num_train+num_valid])(data)

        return train, valid, test

    def _process_split(self, frames, regressors, scorer, split=None):
        suffix = ""
        fts_sen = []

        if split:
            suffix = f"_{split}"
            split_image_dir = self.dataset_dir / split
            split_image_dir.mkdir(parents=True, exist_ok=True)

            for frame in frames:
                fts_sen.append(frame["ft_sen"])
                image_path = Path(frame["file_path"])
                shutil.copy(image_path, split_image_dir / image_path.name)
        else:
            for frame in frames:
                fts_sen.append(frame["ft_sen"])

        regressors = np.reshape(regressors, (-1, 10))
        fts_sen = np.reshape(fts_sen, -1)
        est_iparams, _, _, _ = np.linalg.lstsq(regressors, fts_sen)
        score = scorer.calculate(est_iparams)
        gt_iparams = scorer.gt_iparams

        labels = ["total_mass",
                  "mx", "my", "mz",
                  "ixx", "iyy", "izz", "ixy", "iyz", "izx",
                  "aabb_scale", "score"]
        global_gt = [ *gt_iparams, self.aabb_scale, np.nan]
        lstsq     = [*est_iparams,          np.nan,  score]

        split_transform = self.base_transform.copy()
        split_transform["frames"] = frames
        split_transform["labels"] = labels
        split_transform["global_gt"] = global_gt
        split_transform["lstsq"] = lstsq
        #split_transform["globalinertia"] = comparison.to_json()

        with open(self.dataset_dir / f"transform{suffix}.json", "w") as f:
            json.dump(split_transform, f, indent=2)

    def finish(self,
               frames,
               regressors,
               scorer):
        self.videowriter.release()

        train_frames, valid_frames, test_frames = self._split(frames)
        train_regressors, valid_regressors, test_regressors = self._split(regressors)

        self._process_split(frames, regressors, scorer)
        self._process_split(train_frames, train_regressors, scorer, split="train")
        self._process_split(valid_frames, valid_regressors, scorer, split="valid")
        self._process_split(test_frames, test_regressors, scorer, split="test")

#        with open(self.dataset_dir / "transform.json", "w") as f:
#            json.dump(self.transform, f, indent=2)


#        print("Tracking camera setup =======================================\n"
#             f"    Tracking camera id:         {self.id}\n"
#             f"    Image size (w x h [px]):    {self.width} x {self.height}\n"
#             f"    Focus [px]:                 {self.focus}\in"
#             f"    FoV (h, v [deg]):           {deg(self.fovx)}, {deg(self.fovy)}\n"
#             f"    Output file:                {self.output_file}")
