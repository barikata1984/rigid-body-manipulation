import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import pandas as pd
from dm_control import mjcf
from mujoco._functions import mj_name2id, mj_resetDataKeyframe
from mujoco._structs import MjData, MjModel
from mujoco._enums import mjtObj
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.errors import ConfigAttributeError, MissingMandatoryValue
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

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


def load_config():
    cfg = OmegaConf.structured(SimulationConfig)
    cli_cfg = OmegaConf.from_cli()

    try:
        yaml_cfg = OmegaConf.load(cli_cfg.read_config)
    except ConfigAttributeError:  # if read_config not provided on cli, cli_cfg
        yaml_cfg = {}             # does not have it as its attribute, so using
                                  # this error rather than MissingMandatoryValue

    cfg = OmegaConf.merge(cfg, yaml_cfg, cli_cfg)

    try:
        OmegaConf.save(cfg, cfg.write_config)
    except MissingMandatoryValue:
        pass

    return cfg


def get_target_object_gt(target_object_aux_path):
    # Recover the object's diaginertia and its orientation manually =======
    target_object_aux = pd.read_csv(target_object_aux_path,
                          nrows=1,  # num data rows after the header
                          ).loc[0]

    aabb_scale = target_object_aux["aabb_scale"]

    # Orientaion of the object's body frame w.r.t its inertia frame
    s_rx, s_ry, s_rz = target_object_aux["rx":"rz"]
    rot_obji_obj= euler2mat(s_rx, s_ry, s_rz, "sxyz")  # "S"tatic "XYZ" euler
    iquat = mat2quat(rot_obji_obj.T)

    # Get the mass and the center of mass of the target object w.r.t its body frame
    mass = target_object_aux["total_mass"]
    com = target_object_aux["cx":"cz"]

    # Object's moments of inertia w.r.t the body frame. <- statement
    ixx, iyy, izz, ixy, iyz, izx = target_object_aux["ixx":"izx"]
    imat_obj_obji = np.array([[ixx, ixy, izx],
                              [ixy, iyy, iyz],
                              [izx, iyz, izz]])

    diaginertia_tensor = rot_obji_obj @ imat_obj_obji @ rot_obji_obj.T
    diaginertia = np.diag(diaginertia_tensor)

    fullinertia = [ixx, iyy, izz, ixy, izx, iyz]

    return dict(aabb_scale=aabb_scale, mass=mass, com=com, iquat=iquat,
                diaginertia=diaginertia, fullinertia=fullinertia)


def show_comparison(m,
                    mkey,
                    target_object_gt):

    print(f"{mkey=}")

    mj_mass = m.body_mass[get_element_id(m, "body", mkey)]
    mj_diaginertia = m.body_inertia[get_element_id(m, "body", mkey)]
    mj_iquat = m.body_iquat[get_element_id(m, "body", mkey)]

    # Show the result =====================================================
    total_mass = target_object_gt["mass"]
    diaginertia = target_object_gt["diaginertia"]
    quat_obj_obji = target_object_gt["iquat"]
    print("Mass and diaginertia")
    print(pd.DataFrame({"cad-gt": [total_mass,
                                   *diaginertia,
                                   *quat_obj_obji,
                                   ],
                        "mujoco": [mj_mass,
                                   *mj_diaginertia,
                                   mj_iquat,
                                   ]
                        },
                        index=["total_mass",
                               "pixx", "piyy", "pizz",
                               "real", "i", "j", "k",
                               ]
                       ).transpose()
          )

    mj_rot_body_obji = quat2mat(mj_iquat)
    mj_diaginertia_tensor = np.diag(mj_diaginertia)
    mj_imat_obj_obji = mj_rot_body_obji @ mj_diaginertia_tensor @ mj_rot_body_obji.T

    rot_obj_obji = quat2mat(quat_obj_obji)
    imat_obj_obji = rot_obj_obji @ np.diag(diaginertia) @ rot_obj_obji.T
    print(f"rot_obj_obji:\n{rot_obj_obji}")
    print(f"imat_obj_obji:\n{imat_obj_obji}")
    print(f"mj_imat_obj_obji:\n{mj_imat_obj_obji}")
    print(f"isclose?\n{np.isclose(imat_obj_obji, mj_imat_obj_obji)}")


def spawn_target_object(target_object_path,
                        target_object_aux_path,
                        set_fullinertia=True,
                        compare_cad_mujoco=True,
                        ):
    # Recover the object's diaginertia and its orientation manually =======

    target_object_gt = get_target_object_gt(target_object_aux_path)

    # Orientaion of the object's body frame w.r.t its inertia frame
    quat_obj_obji = target_object_gt["iquat"]  # mat2quat(rot_obji_obj.T)
    rot_obji_obj= quat2mat(quat_obj_obji)

    # Get the mass and the center of mass of the target object w.r.t its body frame
    target_object_mass = target_object_gt["mass"]
    pos_obj_obji = target_object_gt["com"]

    # Object's moments of inertia w.r.t the body frame. <- statement
    fullinertia = target_object_gt["fullinertia"]
    ixx, iyy, izz, ixy, izx, iyz = fullinertia
    imat_obj_obji = np.array([[ixx, ixy, izx],
                              [ixy, iyy, iyz],
                              [izx, iyz, izz]])

    diaginertia_tensor = rot_obji_obj @ imat_obj_obji @ rot_obji_obj.T
    manual_diaginertia = np.diag(diaginertia_tensor)

    print(f"{target_object_gt['aabb_scale']=}")
    # Get mass and diaginertia computed by mujoco =========================
    target_object = mjcf.from_path(target_object_path)
    target_object.custom.add("numeric",
                             name="aabb_scale",
                             data=str(target_object_gt["aabb_scale"]),
                             )

    # Get asset files
    assets = {}
    meshes = []
    for each_mesh in iter(target_object.asset.mesh):
        file = each_mesh.get_attributes()['file']
        assets[file.prefix+file.extension] = file.contents
        meshes.append(file.prefix)

    for elem_texture in iter(target_object.asset.texture):
        file = elem_texture.get_attributes()['file']
        assets[file.prefix+file.extension] = file.contents

    target_object_body = target_object.find("body", "object")
    if set_fullinertia:
        print("======== Set 'fullinertia' to the target object. ========")
        target_object_body.add("inertial",
                               pos=pos_obj_obji,
                               mass=target_object_mass,
                               fullinertia=fullinertia,
                               )

    else:
        print("==== Set 'density' to the target object's each geom. ====")
        component_mass_densities = pd.read_csv(target_object_aux_path,
                                             skiprows=2,  # after the header and data
                                             )

        assert len(component_mass_densities) == len(meshes), \
               RuntimeError("Number of components does not match with the " \
                            "number of aux data. Review the XML and CSV file")

        for idx, mesh in zip(reversed(component_mass_densities.index),
                             reversed(meshes),
                             ):
            density = component_mass_densities.loc[idx, "mass_density"]
            target_object_body.insert("geom", 0,
                                      type="mesh",
                                      mesh=mesh,
                                      density=density,
                                      )

        # Use only the inserted geom for inertia computation
        inertiagrouprange = " ".join(str(i) for i in [0, len(component_mass_densities)-1])
        target_object.compiler.inertiagrouprange = inertiagrouprange

    if compare_cad_mujoco:
        m = MjModel.from_xml_string(target_object.to_xml_string(), assets=assets)
        show_comparison(m, "object", target_object_gt)

    return target_object, assets


def generate_model_data(
        cfg: Union[DictConfig, ListConfig],
    ) -> tuple[MjModel, MjData, float]:  # , PathLike]:
    # Get the ground truth data output by a CAD application ========================
    xml_dir = Path.cwd() / "xml_models"
    target_dir = xml_dir / "targets" / cfg.target_name
    target_object_path = target_dir / "object.xml"
    target_object_aux_path = target_dir / "object_aux.csv"
    target_object, assets = spawn_target_object(target_object_path,
                                                target_object_aux_path,
                                                compare_cad_mujoco=False,
                                                )

    # Load the .xml of a manipulator and attache the target object
    manipulator_path = xml_dir / "manipulators" / f"{cfg.manipulator_name}.xml"
    manipulator = mjcf.from_path(manipulator_path)
    attachement_site = manipulator.find("site", "attachment")
    attachement_site.attach(target_object)

    # Set camera position
    aabb_scale = manipulator.custom.numeric['target/aabb_scale'].data[0]
    track_cam_pos = [0, 0, 3 * aabb_scale]
    track_cam = manipulator.find('camera', cfg.logger.track_cam_name)
    track_cam.pos = track_cam_pos

    # Spawn a mujoco model and a mujoco data
    m = MjModel.from_xml_string(manipulator.to_xml_string(), assets=assets)
    d = MjData(m)

    print(f"{type(manipulator.to_xml_string())=}")
    with open("test.xml", "w") as my_file:
        my_file.write(manipulator.to_xml_string())

    #show_comparison(m, "target/object", cad_gt)

    mj_resetDataKeyframe(m, d,
                         get_element_id(m, "keyframe", cfg.reset_keyframe))

    return m, d, aabb_scale

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


def get_element_id(m, elem_type, name):
    obj_enum = None

    if "body"== elem_type:
        obj_enum = mjtObj.mjOBJ_BODY
    elif "camera"== elem_type:
        obj_enum = mjtObj.mjOBJ_CAMERA
    elif "joint"== elem_type:
        obj_enum = mjtObj.mjOBJ_JOINT
    elif "sensor"== elem_type:
        obj_enum = mjtObj.mjOBJ_SENSOR
    elif "site"== elem_type:
        obj_enum = mjtObj.mjOBJ_SITE
    elif "keyframe"== elem_type:
        obj_enum = mjtObj.mjOBJ_KEY
    else:
        raise ValueError(f"'{elem_type}' is not supported for now. Use mj_name2id and check the value of an ID instead.")

    id = mj_name2id(m, obj_enum, name)

    if -1 == id:
        raise ValueError(f"ID for '{name}' not found. Check the manipulator .xml or the object .xml")

    return id

