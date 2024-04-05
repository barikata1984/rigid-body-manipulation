import os
from dataclasses import dataclass
from math import atan2, radians, tan
from pathlib import Path
from typing import Any

import cv2
from mujoco._enums import mjtObj
from mujoco._functions import mj_name2id
from mujoco._structs import MjData, MjModel
from mujoco.renderer import Renderer
from omegaconf import MISSING


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


class Logger:
    def __init__(self,
                cfg: LoggerConfig,
                m: MjModel,
                d: MjData,
                ) -> None:
        self.cam_name = cfg.track_cam_name
        self.cam_id = mj_name2id(m, mjtObj.mjOBJ_CAMERA, self.cam_name)
        self.fig_height = cfg.fig_height
        self.fig_width = cfg.fig_width
        self.cam_fovy = radians(m.cam_fovy[self.cam_id])
        self.cam_focus = self.fig_height / tan(.5 * self.cam_fovy)
        self.cam_fovx = 2 * atan2(self.fig_width, self.cam_focus)
        self.fps = cfg.fps
        self.dataset_dir = Path(cfg.dataset_dir)
        self.image_dir = self.dataset_dir / "images"

        self.videowriter = cv2.VideoWriter(str(self.dataset_dir / cfg.videoname),
                                           cv2.VideoWriter_fourcc(*cfg.videcodec),
                                           self.fps,
                                           (self.fig_width, self.fig_height),
                                           )

        self.renderer = Renderer(m, self.fig_height, self.fig_width)

        os.makedirs(self.image_dir, exist_ok=True)

#        print("Tracking camera setup =======================================\n"
#             f"    Tracking camera id:         {self.id}\n"
#             f"    Image size (w x h [px]):    {self.width} x {self.height}\n"
#             f"    Focus [px]:                 {self.focus}\n"
#             f"    FoV (h, v [deg]):           {deg(self.fovx)}, {deg(self.fovy)}\n"
#             f"    Output file:                {self.output_file}")
