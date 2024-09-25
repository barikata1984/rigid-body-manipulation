import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import pandas as pd
from dm_control import mjcf
from numpy.typing import NDArray
from mujoco._functions import mj_resetDataKeyframe
from mujoco._structs import MjData, MjModel
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.errors import ConfigAttributeError, MissingMandatoryValue
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

# all the modules of the packages below are imported to enable autoinstantiate()
import dynamics as dyn
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
        index += ["pixx", "piyy", "pizz",
                  "real", "i", "j", "k",
                  ]

        cad_gt += [*target_object_gt["diaginertia"],
                   *target_object_gt["iquat"],
                   ]

        mj += [*mj_diaginertia,
               *mj_iquat,
               ]

    elif "fullinertia" == mode:
        mj_irot = quat2mat(mj_iquat)
        mj_fi = mj_irot @ np.diag(mj_diaginertia) @ mj_irot.T

        index += ["ixx", "iyy", "izz", "ixy", "iyz", "izx"]
        cad_gt += [*target_object_gt["fullinertia"]]
        mj += [mj_fi[0, 0], mj_fi[1, 1], mj_fi[2, 2], mj_fi[0, 1], mj_fi[1, 2], mj_fi[2, 0]]

    else:
        raise ValueError(f"'mode' argument has to be either 'diaginertia' or 'fullinertia'. \
                           '{mode}' is invalid.")

    print(pd.DataFrame({"cad-gt": cad_gt,
                        "mujoco": mj,
                        },
                        index=index,
                       ).transpose()
          )


def get_target_object_ground_truth(target_object_cad_gt_path):
    # Recover the object's diaginertia and its orientation manually =======
    target_object_aux = pd.read_csv(target_object_cad_gt_path,
                          nrows=1,  # num data rows after the header
                          ).loc[0]

    aabb_scale = target_object_aux["aabb_scale"]

    # Orientaion of the object's body frame w.r.t its inertia frame
    s_rx, s_ry, s_rz = target_object_aux["rx":"rz"].to_numpy()
    rot_obji_obj= euler2mat(s_rx, s_ry, s_rz, "sxyz")  # "S"tatic "XYZ" euler
    iquat = mat2quat(rot_obji_obj.T)

    # Get the mass and the center of mass of the target object w.r.t its body frame
    mass = target_object_aux["total_mass"]
    pos_aabb_obji = target_object_aux["cx":"cz"].to_numpy()  # CoM

    # Get the object's moments of inertia w.r.t the frame whose origin is the 
    # object's com and its orientation is aligned with the object's aabb frame
    ixx, iyy, izz, ixy, iyz, izx = target_object_aux["ixx":"izx"].to_numpy()
    fullinertia = [ixx, iyy, izz, ixy, iyz, izx]  # list[float]
    _fullinertia = np.array([[ixx, ixy, izx],
                             [ixy, iyy, iyz],
                             [izx, iyz, izz]])

    # Get the principal inertia tensor of the object
    _diaginertia = rot_obji_obj @ _fullinertia @ rot_obji_obj.T
    diaginertia = np.diag(_diaginertia).tolist()

    # Get the inertia tensor w.r.t the object's aabb frame
    pose_aabb_obji = SE3(SO3.identity(), pos_aabb_obji)
    _globalinertia = dyn.coordinate_transfer_imat(pose_aabb_obji,
                                                  _fullinertia,
                                                  mass,
                                                  )

    globalinertia = [_globalinertia[0, 0], _globalinertia[1, 1], _globalinertia[2, 2],
                     _globalinertia[0, 1], _globalinertia[1, 2], _globalinertia[0, 1]]

    return dict(aabb_scale=aabb_scale, mass=mass, com=pos_aabb_obji, iquat=iquat,
                diaginertia=diaginertia, fullinertia=fullinertia,
                globalinertia=globalinertia)


def spawn_target_object(target_object_path,
                        target_object_cad_gt_path,
                        inertia_setting="diaginertia",
                        compare_cad_mujoco=True,
                        ):
    # Recover the object's diaginertia and its orientation manually =======
    ground_truth = get_target_object_ground_truth(target_object_cad_gt_path)
    # Get mass and diaginertia computed by mujoco =========================
    target_object = mjcf.from_path(target_object_path)
    target_object.custom.add("numeric", 
                             name="aabb_scale",
                             data=str(ground_truth["aabb_scale"]),
                             )

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
        target_object.worldbody.add("site", name="ft_sensor", euler="0 0 180")
    target_object_body = target_object.find("body", "object")

    inertial = dict(pos=ground_truth["com"],
                    mass=ground_truth["mass"],
                    )

    if "diaginertia" == inertia_setting:
        logging.info("======== Set 'diaginertia' to the target object. ========")
        inertial["quat"] = ground_truth["iquat"]
        inertial["diaginertia"] = ground_truth["diaginertia"]
    elif "fullinertia" == inertia_setting:
        logging.info("======== Set 'fullinertia' to the target object. ========")
        ixx, iyy, izz, ixy, iyz, izx = ground_truth["fullinertia"]
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
        show_comparison(m, "object", ground_truth)

    return target_object, assets, ground_truth


def generate_model_data(
        cfg: Union[DictConfig, ListConfig],
    ) -> tuple[MjModel, MjData, dict[str, Union[float, list[float]]]]:
    # Get the ground truth data output by a CAD application ========================
    xml_dir = Path.cwd() / "xml_models"
    target_dir = xml_dir / "targets" / cfg.target_name
    target_object_path = target_dir / "object.xml"
    target_object_cad_gt_path = target_dir / "object_cad_gt.csv"
    target_object, assets, ground_truth = spawn_target_object(
        target_object_path, target_object_cad_gt_path, compare_cad_mujoco=False)

    # Load the .xml of a manipulator and attach the target object to it
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
    m = MjModel.from_xml_string(manipulator.to_xml_string(filename_with_hash=False), assets=assets)
    d = MjData(m)

    show_comparison(m, "target/object", ground_truth, mode="diaginertia")

    mj_resetDataKeyframe(m, d,
                         get_element_id(m, "keyframe", cfg.reset_keyframe))

    return m, d, ground_truth

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

