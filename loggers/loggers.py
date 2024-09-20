import os
from datetime import datetime
from dataclasses import dataclass
from math import atan2, radians, tan
from pathlib import Path

import cv2
import json
import numpy as np
from mujoco._structs import MjData, MjModel
from mujoco.renderer import Renderer
from omegaconf import MISSING

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
        self.image_dir = self.dataset_dir / "images"
        self.renderer = Renderer(m, self.fig_height, self.fig_width)

        os.makedirs(self.image_dir, exist_ok=True)  # not sure but should be called before
                                                    # the videowriter is instantiated

        self.videowriter = cv2.VideoWriter(
            str(self.dataset_dir / cfg.videoname),
            cv2.VideoWriter_fourcc(*cfg.videcodec),
            self.fps,
            (self.fig_width, self.fig_height),
        )

        #_aabb_scale = get_aabb_scale(m)

        #if _aabb_scale != cfg.aabb_scale:
        #    raise ValueError(f"aabb_scale ({_aabb_scale}) retrieved from an MjModel "
        #                     f"instance differs from aabb_scale ({cfg.aabb_scale}) "
        #                      "stored in 'ground_truth.csv'. Review your code.")

        self.transform = dict(
            date_time=datetime.now().strftime("%d/%m/%Y_%H:%M:%S"),
            camera_angle_x=self.cam_fovx,
            aabb_scale=cfg.aabb_scale,
            #cfg.target_object_aabb_scale,
            #gt_mass_distr_file_path=cfg.gt_mass_distr_file_path,
            frames=[],  # list(),
            lstsq=None
        )

    def render(self, d, file_name, cam_id=None):
        if cam_id is None:
            cam_id = self.cam_id

        self.renderer.update_scene(d, cam_id)
        bgr = self.renderer.render()[:, :, [2, 1, 0]]
        # Make an alpha mask to remove the white background
        alpha = np.where(np.all(bgr == 0, axis=-1), 0, 255)[..., np.newaxis]
        cv2.imwrite(str(self.image_dir / file_name),
                    np.append(bgr, alpha, axis=2))  # image (bgr + alpha)
        # Write a video frame
        self.videowriter.write(bgr)


    def finish(self):
        self.videowriter.release()
        with open(self.dataset_dir / "transform.json", "w") as f:
            json.dump(self.transform, f, indent=2)


#        print("Tracking camera setup =======================================\n"
#             f"    Tracking camera id:         {self.id}\n"
#             f"    Image size (w x h [px]):    {self.width} x {self.height}\n"
#             f"    Focus [px]:                 {self.focus}\in"
#             f"    FoV (h, v [deg]):           {deg(self.fovx)}, {deg(self.fovy)}\n"
#             f"    Output file:                {self.output_file}")
