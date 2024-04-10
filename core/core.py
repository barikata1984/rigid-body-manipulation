import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from dm_control import mjcf
from mujoco._functions import mj_name2id, mj_resetDataKeyframe
from mujoco._structs import MjData, MjModel, MjOption
from mujoco._enums import mjtObj
from numpy.typing import NDArray
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

# all the modules of the packages below are imported to enable autoinstantiate()
from controllers import *
from dynamics import *
from loggers import *
from planners import *


@dataclass
class SimulationConfig:
    manipulator_name: str = "sequential"
    target_name: str = "uniform123_128"
    reset_keyframe: str = "initial_state"
    state_space: StateSpaceConfig = StateSpaceConfig()  # from dynamics
    logger: LoggerConfig = LoggerConfig()
    planner: JointPositionPlannerConfig = JointPositionPlannerConfig()
    controller: LinearQuadraticRegulatorConfig = LinearQuadraticRegulatorConfig()
    read_config: str = MISSING
    write_config: str = MISSING


def generate_model_data(cfg: Union[DictConfig, ListConfig],
                        ) -> tuple[MjModel, MjData, NDArray]:
    # Load the .xml of a manipulator
    xml_dir = Path.cwd() / "xml_models"
    manipulator_path = xml_dir / "manipulators" / f"{cfg.manipulator_name}.xml"
    manipulator = mjcf.from_path(manipulator_path)

    # Load the .xml of a target object and its ground truth mass distribution .npy
    target_dir = xml_dir / "targets" / cfg.target_name
    target_object_path = target_dir / "object.xml"
    target_object = mjcf.from_path(target_object_path)

    gt_mass_distr_path = target_dir / "gt_mass_distr.npy"
    gt_mass_distr = np.zeros(0)

    # Attache the object to obtain the complete model tree
    attachement_site = manipulator.find("site", "attachment")
    attachement_site.attach(target_object)

    # Relocate the track cam according to the target object's aabb scale
    target_object_aabb_scale = target_object.find("numeric", "aabb_scale")
    track_cam_pos = [0, 0, 2*target_object_aabb_scale.data[0]]
    track_cam = manipulator.find('camera', cfg.logger.track_cam_name)
    track_cam.pos = track_cam_pos

    # Spawn a mujoco model and a mujoco data
    m = MjModel.from_xml_string(manipulator.to_xml_string())
    d = MjData(m)

    #print(f"{manipulator.to_xml_string()}")

    # Reset the manipulator's configuration if reset_keyframe specified.
    # If reset_keyframe does not match any keyframe inside the .xml files,
    # nothing happens.
    mj_resetDataKeyframe(m, d,
                         mj_name2id(m, mjtObj.mjOBJ_KEY, cfg.reset_keyframe))

    return m, d, gt_mass_distr

    #print("Configure manipulator and object ============================")
    #xml_file = config.xml.system_file
    #print(f"    Loaded xml file: {xml_file}")
    #xml_path = os.path.join("./xml_models", xml_file)
    #m = MjModel.from_xml_path(xml_path)
    #print("Number of ===================================================\n"
    #     f"    coorindates in joint space (nv):    {m.nv:>2}\n"
    #     f"    degrees of freedom (nu):            {m.nu:>2}\n"
    #     f"    actuator activations (na):          {m.na:>2}\n"
    #     f"    sensor outputs (nsensordata):       {m.nsensordata:>2}")


def autoinstantiate(cfg: DictConfig,
                    m: MjModel,
                    d: MjData,
                    *args, **kwargs) -> Any:  # TODO: reasonable but rough

    for name, _class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if cfg.target_class == name:
            return _class(cfg, m, d, *args, **kwargs)

