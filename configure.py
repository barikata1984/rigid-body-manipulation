import os
import cv2
import toml
import numpy as np
import mujoco as mj
import planners as pln
import dynamics as dyn
from dataclasses import dataclass
from attrdict import AttrDict
from math import tan, atan2, radians as rad, degrees as deg


@dataclass
class TimeConfig:
    duration: float  # Simulation time [s]
    fps: int  # Rendering frequency [Hz]
    timestep: float = None

    def __post_init__(self):
        if None is self.timestep or self.timestep < 0:
            self.timestep = mj.MjOption().timestep  # 0.002 [s] by default

        self.n_steps = int(self.duration / self.timestep)

        print("Simulation time setup =======================================\n"
             f"    Number of steps:            {self.n_steps}\n"
             f"    Simulation time [s]:        {self.duration}\n"
             f"    Timestep [s]:               {self.timestep}\n"
             f"    Simulation freq. [Hz]:      {1 / self.timestep}\n"
             f"    Rendering freq. [Hz]:       {self.fps}")


@dataclass
class CameraConfig:
    id: int
    fovy: float
    height: int
    width: int
    codec_4chr: str
    output_file: str

    def __post_init__(self):
        self.focus = self.height / tan(self.fovy / 2)
        self.fovx = 2 * atan2(self.width, self.focus)
        self.fourcc = cv2.VideoWriter_fourcc(*self.codec_4chr)

        print("Tracking camera setup =======================================\n"
             f"    Tracking camera id:         {self.id}\n"
             f"    Image size (w x h [px]):    {self.width} x {self.height}\n"
             f"    Focus [px]:                 {self.focus}\n"
             f"    FoV (h, v [deg]):           {deg(self.fovx)}, {deg(self.fovy)}\n"
             f"    Output file:                {self.output_file}")


def generate_trajectory_planner(
        m: mj.MjModel,
        d: mj.MjData,
        t: TimeConfig,
        init_frame: str,
        dqpos: np.ndarray):

    # Load the initial keyframe
    init_key_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, init_frame)
    mj.mj_resetDataKeyframe(m, d, init_key_id)
    # Derive start and goal qpos
    start_qpos = d.qpos.copy()
    goal_qpos = start_qpos + dqpos

    print("Simulation time setup =======================================\n"
         f"    qpos: {d.qpos}\n"
          "            ↓\n"
         f"          {d.qpos + dqpos}")

    return pln.traj_5th_spline(start_qpos, goal_qpos, t.timestep, t.n_steps)


def load_configs(config_file):
    config = AttrDict(toml.load(config_file))

    print("Configure manipulator and object ============================")
    xml_file = config.xml.system_file
    print(f"    Loaded xml file: {xml_file}")
    xml_path = os.path.join("./xml_models", xml_file)
    m = mj.MjModel.from_xml_path(xml_path)
    print("Number of ===================================================\n"
         f"    coorindates in joint space (nv):    {m.nv:>2}\n"
         f"    degrees of freedom (nu):            {m.nu:>2}\n"
         f"    actuator activations (na):          {m.na:>2}\n"
         f"    sensor outputs (nsensordata):       {m.nsensordata:>2}")

    # Generate mjData struct which stores simulation state
    d = mj.MjData(m)
    # Setup simulation time configuration
    timestep = config.simu_time.timestep
    if timestep < 0:
        timestep = mj.MjOption().timestep
    t = TimeConfig(
        duration=config.simu_time.duration,
        fps=config.simu_time.fps,
        timestep=timestep)
    # Setup tracking camera configuration
    cam_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, config.track_cam.name)
    cam = CameraConfig(
        id=cam_id,
        fovy=rad(m.cam_fovy[cam_id]),  # [rad]
        height=config.track_cam.height,
        width=config.track_cam.width,
        codec_4chr=config.track_cam.codec,
        output_file=config.track_cam.output_file)

#    ss = dyn.StateSpace(m, d, config)

    plan = generate_trajectory_planner(
        m=m,
        d=d,
        t=t,
        init_frame=config.trajectory.init_frame,
        dqpos=config.trajectory.dqpos)

    return m, d, t, cam, plan  # ss, plan

