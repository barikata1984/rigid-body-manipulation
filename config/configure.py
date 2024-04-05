import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dm_control import mjcf
from mujoco._functions import mj_name2id, mj_resetDataKeyframe
from mujoco._structs import MjData, MjModel, MjOption
from mujoco._enums import mjtObj
from numpy.typing import NDArray
from omegaconf.dictconfig import DictConfig

from controllers import *
from dynamics import *
from loggers import *
from planners import *


@dataclass
class CoreConfig:
    manipulator_name: str = "sequential"
    target_name: str = "uniform123_128"
    reset_keyframe: str = "initial_state"


@dataclass
class SimulationConfig:
    core: CoreConfig = CoreConfig()
    state_space: StateSpaceConfig = StateSpaceConfig()
    logger: LoggerConfig = LoggerConfig()
    planner: JointPositionPlannerConfig = JointPositionPlannerConfig()
    controller: LinearQuadraticRegulatorConfig = LinearQuadraticRegulatorConfig()


def generate_model_data(cfg: CoreConfig,
                        ) -> tuple[MjModel, MjData, NDArray]:
    # Load a manipulator's .xml
    xml_dir = Path.cwd() / "xml_models"
    manipulator_path = xml_dir / "manipulators" / f"{cfg.manipulator_name}.xml"
    manipulator = mjcf.from_path(manipulator_path)

    # Load the .xml of a target object and its mass distribution .npy
    target_dir = xml_dir / "targets" / cfg.target_name
    target_object_path = target_dir / "object.xml"
    target_object = mjcf.from_path(target_object_path)

    mass_distr_path = target_dir / "mass_distr.npy"
    gt_mass_distr = np.zeros(0)

    # Attache the object to obtain the complete model tree
    attachement_site = manipulator.find("site", "attachment")
    attachement_site.attach(target_object)
    # Spawn a model and a data
    m = MjModel.from_xml_string(manipulator.to_xml_string())  # I thought that this should print exactly the same as the final generated XML
    d = MjData(m)

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
                    ) -> Any:  # TODO: reasonable but rough, use Protocol or Generic

    for name, _class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if cfg.target_class == name:
            return _class(cfg, m, d)

