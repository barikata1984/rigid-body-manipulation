import inspect
import logging
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
    target_name: str = MISSING
    reset_keyframe: str = "initial_state"
    state_space: StateSpaceConfig = MISSING  # StateSpaceConfig()  # from dynamics
    logger: LoggerConfig = LoggerConfig()
    planner: JointPositionPlannerConfig = MISSING  # JointPositionPlannerConfig()
    controller: LinearQuadraticRegulatorConfig = MISSING # LinearQuadraticRegulatorConfig()
    read_config: str = "./configurations/base.yaml"
    write_config: str = MISSING


def load_config():
    cfg = OmegaConf.structured(SimulationConfig)
    base_cfg = OmegaConf.load(cfg.read_config)
    OmegaConf.structured(SimulationConfig)

    cli_cfg = OmegaConf.from_cli()

    try:
        yaml_cfg = OmegaConf.load(cli_cfg.read_config)
    except ConfigAttributeError:  # if read_config not provided on cli, cli_cfg
        yaml_cfg = {}             # does not have it as its attribute, so using
                                  # this error rather than MissingMandatoryValue

    cfg = OmegaConf.merge(cfg, base_cfg, yaml_cfg, cli_cfg)

    try:
        OmegaConf.save(cfg, cfg.write_config)
    except MissingMandatoryValue:
        pass

    return cfg


def show_comparison(
        m,
        mkey,
        target_object_gt,
        mode: str = "diaginertia",
    ):
    mj_mass = m.body_mass[get_element_id(m, "body", mkey)]
    mj_com = m.body_ipos[get_element_id(m, "body", mkey)]
    mj_diaginertia = m.body_inertia[get_element_id(m, "body", mkey)]
    mj_iquat = m.body_iquat[get_element_id(m, "body", mkey)]

    # Show the result =====================================================
    total_mass = target_object_gt["mass"]
    com = target_object_gt["com"]

    index = ["total_mass", "mx", "my", "mz"]
    cad_gt = [total_mass, *(total_mass*com)]
    mj = [mj_mass, *(mj_mass*mj_com)]

    if "diaginertia" == mode:
        diaginertia = target_object_gt["diaginertia"]
        quat_obj_obji = target_object_gt["iquat"]

        index += ["pixx", "piyy", "pizz", "real", "i", "j", "k"]
        cad_gt += [*diaginertia, *quat_obj_obji]
        mj += [*mj_diaginertia, *mj_iquat]
    elif "fullinertia" == mode:
        fullinertia = target_object_gt["fullinertia"]
        mj_irot = quat2mat(mj_iquat)
        mj_fi = mj_irot @ np.diag(mj_diaginertia) @ mj_irot.T
        mj_fullinertia = np.array([mj_fi[0, 0], mj_fi[1, 1], mj_fi[2, 2],
                                   mj_fi[0, 1], mj_fi[1, 2], mj_fi[2, 0]])

        index += ["ixx", "iyy", "izz", "ixy", "iyz", "izx"]
        cad_gt += [*fullinertia]
        mj += [*mj_fullinertia]
    else:
        raise ValueError(f"'mode' argument has to be either 'diaginertia' or 'fullinertia'. \
                           '{mode}' is invalid.")

    print(pd.DataFrame({"cad-gt": cad_gt,
                        "mujoco": mj,
                        },
                        index=index,
                       ).transpose()
          )


def get_target_object_gt(target_object_cad_gt_path):
    # Recover the object's diaginertia and its orientation manually =======
    target_object_aux = pd.read_csv(target_object_cad_gt_path,
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
    #print(f"diaginertia_tensor:\n{diaginertia_tensor}")
    diaginertia = np.diag(diaginertia_tensor)

    fullinertia = [ixx, iyy, izz, ixy, iyz, izx]

    return dict(aabb_scale=aabb_scale, mass=mass, com=com, iquat=iquat,
                diaginertia=diaginertia, fullinertia=fullinertia)


def spawn_target_object(target_object_path,
                        target_object_cad_gt_path,
                        inertia_setting="diaginertia",
                        compare_cad_mujoco=True,
                        ):
    # Recover the object's diaginertia and its orientation manually =======
    target_object_gt = get_target_object_gt(target_object_cad_gt_path)
    # Get mass and diaginertia computed by mujoco =========================
    target_object = mjcf.from_path(target_object_path)
    numeric = dict(name="aabb_scale",
                   data=str(target_object_gt["aabb_scale"]),
                   )
    target_object.custom.add("numeric", **numeric)

    headlight = target_object.visual.headlight
    headlight.ambient=".5 .5 .5"
    headlight.diffuse=".4 .4 .4"
    headlight.specular=".5 .5 .5"

    # Get asset files
    assets = {}
    meshes = []
    for each_mesh in iter(target_object.asset.mesh):
        file = each_mesh.get_attributes()['file']
        assets[file.prefix+file.extension] = file.contents
        meshes.append(file.prefix)

    for elem_texture in iter(target_object.asset.texture):
        try:
            file = elem_texture.get_attributes()['file']
            assets[file.prefix+file.extension] = file.contents
        except:
            print(f"'{elem_texture.type}' type texture, '{elem_texture.name}', "
                   "does not have 'file' attribute.")

    target_object.asset.add("texture",
                            name="white_background",
                            type="skybox",
                            builtin="flat",
                            rgb1="1 1 1",
                            rgb2="1 1 1",
                            width="1",
                            )

    if target_object.worldbody.find('site', 'ft_sensor') is None:
        target_object.worldbody.add("site", name="ft_sensor")

    target_object_body = target_object.find("body", "object")

    inertial = dict(pos=target_object_gt["com"],
                    mass=target_object_gt["mass"],
                    )

    if "diaginertia" == inertia_setting:
        logging.info("======== Set 'diaginertia' to the target object. ========")
        inertial["quat"] = target_object_gt["iquat"]
        inertial["diaginertia"] = target_object_gt["diaginertia"]
    elif "fullinertia" == inertia_setting:
        logging.info("======== Set 'fullinertia' to the target object. ========")
        ixx, iyy, izz, ixy, iyz, izx = target_object_gt["fullinertia"]
        inertial["fullinertia"] = [ixx, iyy, izz, ixy, iyz, izx]
    elif False and "density" == inertia_setting:  # NOTE: disabled for now
        print("==== Set 'density' to the target object's each geom. ====")
        component_mass_densities = pd.read_csv(target_object_cad_gt_path,
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

    target_object_body.add("inertial", **inertial)

    if compare_cad_mujoco:
        m = MjModel.from_xml_string(target_object.to_xml_string(), assets=assets)
        show_comparison(m, "object", target_object_gt)

    return target_object, assets, target_object_gt


def generate_model_data(
        cfg: Union[DictConfig, ListConfig],
    ) -> tuple[MjModel, MjData]:  # , PathLike]:
    # Get the ground truth data output by a CAD application ========================
    xml_dir = Path.cwd() / "xml_models"
    target_dir = xml_dir / "targets" / cfg.target_name
    target_object_path = target_dir / "object.xml"
    target_object_cad_gt_path = target_dir / "object_cad_gt.csv"
    target_object, assets, cad_gt = spawn_target_object(target_object_path,
                                                        target_object_cad_gt_path,
                                                        compare_cad_mujoco=False,
                                                        )

    # Load the .xml of a manipulator and attache the target object
    manipulator_path = xml_dir / "manipulators" / f"{cfg.manipulator_name}.xml"
    manipulator = mjcf.from_path(manipulator_path)
    attachement_site = manipulator.find("site", "attachment")
    attachement_site.attach(target_object)

    # Set camera position
    aabb_scale = manipulator.custom.numeric["target/aabb_scale"].data[0]
    track_cam_pos = [0, 0, 4*aabb_scale]
    track_cam = manipulator.find("camera", cfg.logger.track_cam_name)
    track_cam.pos = track_cam_pos

    # Spawn a mujoco model and a mujoco data
    m = MjModel.from_xml_string(manipulator.to_xml_string(), assets=assets)
    d = MjData(m)

    show_comparison(m, "target/object", cad_gt, mode="fullinertia")

    mj_resetDataKeyframe(m, d,
                         get_element_id(m, "keyframe", cfg.reset_keyframe))

    return m, d

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
    elif "numeric"== elem_type:
        obj_enum = mjtObj.mjOBJ_NUMERIC
    else:
        raise ValueError(f"'{elem_type}' is not supported for now. Use mj_name2id and check the value of an ID instead.")

    id = mj_name2id(m, obj_enum, name)

    if -1 == id:
        raise ValueError(f"ID for '{name}' not found. Check the manipulator .xml or the object .xml")

    return id

