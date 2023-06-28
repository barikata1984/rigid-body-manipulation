import os
import cv2
import numpy as np
import mujoco as mj
import planners as pln
from dataclasses import dataclass
from math import tan, atan2, pi, radians as rad, degrees as deg


@dataclass
class TimeConfig:
    duration: float  # Simulation time [s]
    fps: int  # Rendering frequency [Hz]
    timestep: float = None
    def __post_init__(self):
        if None is self.timestep:
            self.timestep = mj.MjOption().timestep  # 0.002 [s] (500 [Hz]) by default

        self.n_steps = int(self.duration / self.timestep)


@dataclass
class CameraConfig:
    cam_id: int 
    cam_fovy: float
    height: int
    width: int
    codec_4chr: str
    output_file_name: str
    
    def __post_init__(self):
        self.focus = self.height / tan(self.cam_fovy / 2)  # [pixel]
        self.cam_fovx = 2 * atan2(self.width, self.focus)  # [rad]
        self.fourcc = cv2.VideoWriter_fourcc(*self.codec_4chr)


def load_configs(xml_file):
    xml_file = "obj_w_links.xml"
    xml_path = os.path.join("./xml_models", xml_file)
    print(f"Loaded xml file: {xml_file}")
    m = mj.MjModel.from_xml_path(xml_path)
    d = mj.MjData(m)

    # Setup simulation time configuration
    t = TimeConfig(5, 60)
    
    # Setup tracking camera configuration
    cam_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, "tracking")
    c = CameraConfig(
            cam_id = cam_id, 
            cam_fovy = rad(m.cam_fovy[cam_id]),  # [rad]
            height = 600,
            width = 960,
            codec_4chr = "mp4v",
            output_file_name = "output.mp4")

    # Prepare variables to srore dynamics data
    print("Number of =================================")
    print(f"    coorindates in joint space (nv): {m.nv:>2}")
    print(f"    degrees of freedom (nu):         {m.nu:>2}")
    print(f"    actuator activations (na):       {m.na:>2}")
    print(f"    sensor outputs (nsensordata):    {m.nsensordata:>2}")
    
    show_time_config(t)
    show_camera_config(c)

    return m, d, t, c


def show_time_config(time_config: TimeConfig):
    print("Simulation time setup =====================")
    print(f"    Simulation time: {time_config.duration} [s]")
    print(f"    Timestep (freq.): {time_config.timestep} [s] ({1/time_config.timestep} [Hz])")
    print(f"    Number of steps: {time_config.n_steps}")
    print(f"    Rendering freq.: {time_config.fps} [Hz]")


def show_camera_config(cam_conf: CameraConfig):
    print("Tracking camera setup =====================")
    print(f"    Tracking camera id:   {cam_conf.cam_id}")
    print(f"    Picture height [px]:  {cam_conf.height}")
    print(f"     ↑      width [px]:   {cam_conf.width}")
    print(f"    Focus [px]:           {deg(cam_conf.focus)}")
    print(f"    FoV horizontal [deg]: {deg(cam_conf.cam_fovx)}")
    print(f"     ↑  vertical [deg]:   {deg(cam_conf.cam_fovy)}")
    print(f"    Output file:          {cam_conf.output_file_name}")


def plan_trajectory(m: mj.MjModel, d: mj.MjData, t: TimeConfig, keyframe_name: str, dqpos: np.ndarray):
    ## Load the initial keyframe
    init_key_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, keyframe_name)
    mj.mj_resetDataKeyframe(m, d, init_key_id)
    ## Derive start and goal qpos 
    start_qpos = d.qpos.copy()
    goal_qpos = start_qpos + dqpos
    
    print("Trajectory setup ==========================")
    print(f"    qpos: {d.qpos}")
    print(f"           ↓")
    print(f"          {d.qpos + dqpos}")

    return pln.traj_5th_spline(start_qpos, goal_qpos, t.timestep, t.n_steps)




