from datetime import datetime
from pathlib import Path

import cv2
import json
import matplotlib as mpl
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from mujoco._enums import mjtObj
from mujoco._functions import mj_differentiatePos, mj_name2id, mj_step
from mujoco._structs import MjModel, MjData, MjOption
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError, MissingMandatoryValue
from scipy import linalg
from tqdm import tqdm

import transformations as tf
import visualization as vis
from core import SimulationConfig, generate_model_data, autoinstantiate
from dynamics import compose_spatial_inertia_matrices, transfer_simats, inverse, coordinate_transform_dtwist, compute_linacc


# Naming convention of spatial and dynamics variables:
#
# {descriptor}_{reference}_{described}, where
#
#    descriptor | Definition
# --------------+------------
#       (s)imat | (spatial) inertia matrix
#       (h)pose | (home) pose
#      (u)screw | (unit) screw
#      (d)twist | (first-order time derivative of) twist
#  (lin/ang)vel | (linear/angular) velocity
#  (lin/ang)acc | (linear/angular) acceleration
#         momsi | moments of inertia
#          gacc | graviatational acceleration
#  NOTE: 's' may follow a descriptor to clarify that the variable multiple descriptors.
#
#     reference |
#     /descried | Definition
# --------------+-------------
#             x | world frame
#             b | body itself or its frame (refer to the official documentation)
#            bi | body's principal frame where the body's interia ellipsoid is defined
#          a/ai | body's parent itself or its frame/parent's principal frame
#          c/ci | body's child itself or its frame/child's principal frame
#             q | joint space
#


# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams['axes.xmargin'] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=5, suppress=True)


def simulate(m: MjModel,
             d: MjData,
             gt_mass_distr,  # TODO: annotate late...
             logger, planner, controller,  # TODO: annotate late... make a BaseModule or something and use Protocol or Generic, maybe...
             ):
    _simats_bi_b = compose_spatial_inertia_matrices(m.body_mass, m.body_inertia)
    # Convert sinert_i to sinert_b rel2 the body frame
    poses_b_bi = tf.posquat2SE3s(m.body_ipos, m.body_iquat)
    poses_x_b = tf.posquat2SE3s(d.xpos, d.xquat)
    _simats_b_b = transfer_simats(poses_b_bi, _simats_bi_b)

    # Join the spatial inertia matrices of bodies later than, or fixed relative to,
    # link6 to the matrix of link6 so that dyn.inverse() can consider the bodies'
    # inertia later
    link6_id = mj_name2id(m, mjtObj.mjOBJ_BODY, "link6")
    pose_x_link6 = poses_x_b[link6_id]
    simats_b_b = _simats_b_b[:link6_id+1]

    for p_x_b, _sim_b_b in zip(poses_x_b[link6_id+1:], _simats_b_b[link6_id+1:]):
        p_link6_b = pose_x_link6.inv().dot(p_x_b)
        _sim_link6_b = transfer_simats(p_link6_b, _sim_b_b)
        simats_b_b[link6_id] += _sim_link6_b

    mom_i = np.array([*simats_b_b[-1, 3:, 3:].diagonal(),
                      *simats_b_b[-1, 3 , 4:],
                       simats_b_b[-1, 4 , 5 ]])

#    print("Target object's inertial parameters wr2 its body frame ======\n"
#         f"    Mass:               {m.body_mass[-1]}\n"
#         f"    First moments:      {SO3.vee(simats_b_b[-1, 3:, :3])}\n"
#         f"    Moments of inertia: {mom_i}\n")

    # Configure SE3 of child frame wr2 parent frame (M_{i, i - 1} in MR)
    hposes_a_b = tf.posquat2SE3s(m.body_pos, m.body_quat)

    # Obtain unit screw wr2 each link = body (A_{i} in MR)
    uscrew_b_b = np.zeros((m.body_jntnum.sum(), 6))  # bb = (11, 22, ..., 66)
    for b, (jnt_type, ax) in enumerate(zip(m.jnt_type, m.jnt_axis), 0):
        slicer = 3 * (jnt_type - 2)  # jnt_type: 2 for slide, 3 for hinge
        uscrew_b_b[b, slicer:slicer + 3] = ax / linalg.norm(ax)

    # Set up dynamics related variables =======================================
    # (d)twist vectors for the worldbody to be used for inverse dynamics
    twist_x_x = np.zeros(6)
    gacc_x = np.zeros(6)
    gacc_x[:3] = -1 * MjOption().gravity
    dtwist_x_x = gacc_x.copy()

    # IDs for convenience
    sensor_id = mj_name2id(m, mjtObj.mjOBJ_SITE, "ft_sen")
    object_id = mj_name2id(m, mjtObj.mjOBJ_BODY, "target/object")

    # Dictionary to be converted to a .json file for training
    aabb_scale = 0.3
    transforms = dict(
        date_time=datetime.now().strftime("%d/%m/%Y_%H:%M:%S"),
        camera_angle_x=logger.cam_fovx,
        aabb_scale=aabb_scale,
        gt_mass_distr_file_path=None,
        frames=[],  # list(),
    )

    # Prepare data containers =================================================
    res_qpos = np.empty(m.nu)
    tgt_trajectory = []
    trajectory = []
    linaccs_sen_obj = []
    fts_sen = []
    time = []
    frame_count = 0

    # Main loop ===============================================================
    for step in tqdm(range(planner.n_steps), desc="Progress"):
        # Compute actuator controls and evolute the simulatoin
        tgt_traj = planner.plan(step)
        tgt_ctrl, _, _, _= inverse(
            tgt_traj, hposes_a_b, simats_b_b, uscrew_b_b, twist_x_x, dtwist_x_x)
        # Residual of state
        mj_differentiatePos(  # Use this func to differenciate quat properly
            m,  # MjModel
            res_qpos,  # data container for the residual of qpos
            m.nu,  # idx of a joint up to which res_qpos are calculated
            d.qpos,  # current qpos
            tgt_traj[0])  # target qpos or next qpos to calkculate dqvel
        res_state = np.concatenate((res_qpos, tgt_traj[1] - d.qvel))
        # Compute and set control, or actuator inputs
        d.ctrl = tgt_ctrl - controller.control_gain @ res_state

        mj_step(m, d)  # Evolve the simulation = = = = = = = = = = = = = = =

        # Process sensor reads and compute necessary data
        sensorread = d.sensordata.copy()
        # Scale sampled normalized coordinates ∈ (-1, 1) in wisp to the maximum
        # length of an axis-aligned bounding box of the object.
        # Camera pose rel. to the object
        pose_x_obj = tf.trzs2SE3(d.xpos[object_id], d.xmat[object_id])
        pose_x_cam = tf.trzs2SE3(d.cam_xpos[logger.cam_id],
                                 d.cam_xmat[logger.cam_id])
        pose_obj_cam = pose_x_obj.inv().dot(pose_x_cam)
        # FT sensor pose rel. to the object
        pose_x_sen = tf.trzs2SE3(d.site_xpos[sensor_id], d.site_xmat[sensor_id])
        pose_sen_obj = pose_x_sen.inv().dot(pose_x_obj)

        # Compute (d)twists using dyn.inverse() again to validate the method by
        # comparing derived acceleration and force/torque with their sensor
        # measurements later
        traj = np.stack((d.qpos, d.qvel, d.qacc))
        _, _, twists, dtwists = inverse(
            traj, hposes_a_b, simats_b_b, uscrew_b_b, twist_x_x, dtwist_x_x)

        # First-order time derivative - - - - - - - - - - - - - - - - - - - - -
        twist_obj_obj = twists[-1]
        twist_sen_obj = pose_sen_obj.adjoint() @ twist_obj_obj
#        linvel_sen_obj = compute_linvel(pose_sen_obj, twist_obj_obj, coord_xfer_twist=True)

        # Second-order time derivative - - - - - - - - - - - - - - - - - - - - 
        dtwist_obj_obj = dtwists[-1]
        dtwist_sen_obj = coordinate_transform_dtwist(
            pose_sen_obj, twist_sen_obj, dtwist_obj_obj)  # , coord_xfer_twist=True)
        linacc_sen_obj = compute_linacc(
            pose_sen_obj, twist_sen_obj, dtwist_sen_obj)  # , coord_xfer_twist=True)

#        v_sen_obj, w_sen_obj = np.split(twist_sen_obj, 2)
#        dv_sen_obj, dw_sen_obj = np.split(dtwist_sen_obj, 2)
#        skewed_w_sen_obj = SO3.wedge(w_sen_obj)
#        skewed_dw_sen_obj = SO3.wedge(dw_sen_obj)
#        # 
#        _linacc1_sen_obj = dv_sen_obj + skewed_w_sen_obj @ v_sen_obj
#        _linacc2_sen_obj = skewed_dw_sen_obj @ pose_sen_obj.trans \
#                         + skewed_w_sen_obj @ skewed_w_sen_obj @ pose_sen_obj.trans  # element-wise part
#        _linacc_sen_obj = _linacc1_sen_obj + _linacc2_sen_obj

        # Retrieve force and torque measurements
        ft_sen = sensorread[1*m.nu:2*m.nu]
#        total_mass = ft_sen[:3] / linacc_sen_obj

        # ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 検証用コード追加ゾーン ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 

        if frame_count <= d.time * logger.fps:

        # ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ 検証用コード追加ゾーン ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ 

            # Writing a single frame of a dataset =============================
            logger.renderer.update_scene(d, logger.cam_id)
            bgr = logger.renderer.render()[:, :, [2, 1, 0]]
            # Make an alpha mask to remove the black background
            alpha = np.where(
                np.all(bgr == 0, axis=-1), 0, 255)[..., np.newaxis]
            file_name = f"{frame_count:04}.png"
            cv2.imwrite(str(logger.image_dir / file_name),
                        np.append(bgr, alpha, axis=2))  # image (bgr + alpha)
            # Write a video frame
            logger.videowriter.write(bgr)

            # Log NeMD ingredients ============================================
            frame = dict(
                file_path=str(logger.image_dir / file_name),
#                pose_obj_cam=pose_obj_cam.as_matrix().T.tolist(),
                transform_matrix=pose_obj_cam.as_matrix().tolist(),
                pose_sen_obj=pose_sen_obj.as_matrix().tolist(),
                twist_sen_obj=twist_sen_obj.tolist(),
                dtwist_sen_obj=dtwist_sen_obj.tolist(),
#                obj_linacc_sen=obj_linacc_sen,
                linacc_sen_obj=linacc_sen_obj.tolist(),
                ft_sen=ft_sen.tolist(),
#                aabb_scale=[aabb_scale],
                )

            transforms["frames"].append(frame)

            # Log velocity components relative to the sensor frame
            tgt_trajectory.append(tgt_traj)
            trajectory.append(traj)
            linaccs_sen_obj.append(linacc_sen_obj)
            fts_sen.append(ft_sen)
            time.append(d.time)

            # Sampling for NeMD terminated while "frame_count" incremented
            frame_count += 1

    logger.videowriter.release()

#    qpos_meas, qvel_meas, qfrc_meas, ft_meas_sen, obj_vel_x, obj_acc_x = np.split(
#        sensordata, [1*m.nu, 2*m.nu, 3*m.nu, 4*m.nu, 5*m.nu], axis=1)

    with open(logger.dataset_dir / "transform.json", "w") as f:
        json.dump(transforms, f, indent=2)

    # Convert lists of logged data into ndarrays ==============================
    tgt_trajectory = np.array(tgt_trajectory)
    trajectory = np.array(trajectory)
    linaccs_sen_obj = np.array(linaccs_sen_obj)
    fts_sen = np.array(fts_sen)
    frame_iter = np.arange(frame_count)

    # Visualize data ==========================================================
    # Object linear acceleration and ft sensor measurements rel. to {sensor}
    # Actual and target joint positions
    qpos_fig, qpos_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
    qpos_fig.suptitle("qpos")
    qpos_axes[1].set(xlabel="time [s]")
    yls = ["q0-2 [m]", "q3-5 [rad]"]
    for i in range(len(qpos_axes)):
        slcr = slice(i*3, (i+1)*3)
        vis.ax_plot_lines_w_tgt(
            qpos_axes[i], time, trajectory[:, 0, slcr], tgt_trajectory[:, 0, slcr], yls[i])

    # Object linear acceleration and ft sensor measurements rel. to {sensor}
    acc_ft_fig, acc_ft_axes = plt.subplots(3, 1, tight_layout=True)
    acc_ft_fig.suptitle("linacc vs ft")
    acc_ft_axes[0].set(xlabel="# of frames")
    acc_ft_axes[2].set(xlabel="time [s]")
    vis.ax_plot_lines(acc_ft_axes[0], frame_iter, linaccs_sen_obj, "obj_linacc_sen [m/s/s]")
    vis.ax_plot_lines(acc_ft_axes[1], frame_iter, fts_sen[:, :3], "frc_sen [N]")
    vis.ax_plot_lines(acc_ft_axes[2], frame_iter, fts_sen[:, 3:], "trq_sen [N·m]")

    for ax in acc_ft_axes:
        ax.hlines(0.0, frame_iter[0], frame_iter[-1], ls="dashed", alpha=0.5)

    # Joint forces and torques
#     ctrl_fig, ctrl_axes = plt.subplots(3, 1, sharex="col", tight_layout=True)
#     ctrl_fig.suptitle("act_qfrc VS tgt_ctrls")
#     ctrl_axes[0].set(ylabel="q0-1 [N]")
#     ctrl_axes[1].set(ylabel="q2 [N]")
#     ctrl_axes[2].set(xlabel="time [s]")
#     vis.axes_plot_frc(
#         ctrl_axes[:2], time, sens_qfrc[:, :3], tgt_ctrls[:, :3])
#     vis.ax_plot_lines_w_tgt(
#         ctrl_axes[2], time, sens_qfrc[:, 3:], tgt_ctrls[:, 3:], "q3-5 [N·m]")

    plt.show()


if __name__ == "__main__":
    # Load configuraion
    cfg = OmegaConf.structured(SimulationConfig)
    cli_cfg = OmegaConf.from_cli()

    try:
        yaml_cfg = OmegaConf.load(cli_cfg.read_config)
        cfg = OmegaConf.merge(cfg, yaml_cfg)
    except ConfigAttributeError:  # if read_config not provided, cli_cfg does
        pass                      # not have it as its attribute, so using
                                  # this error rather than MissingMandatoryValue

    cfg = OmegaConf.merge(cfg, cli_cfg)

    try:
        OmegaConf.save(cfg, cfg.write_config)
    except MissingMandatoryValue:
        pass

    # Fill a potentially missing field of a logger configulation
    try:
        cfg.logger.dataset_dir
    except MissingMandatoryValue:
        cfg.logger.dataset_dir = Path.cwd() / "datasets" / cfg.target_name

    # Generate core data structures
    m, d, gt_mass_distr = generate_model_data(cfg)
    # Instantiate necessary classes
    logger = autoinstantiate(cfg.logger, m, d)
    planner = autoinstantiate(cfg.planner, m, d)
    controller = autoinstantiate(cfg.controller, m, d)

    simulate(m, d, gt_mass_distr, logger, planner, controller)
