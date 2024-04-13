from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Union

import cv2
import json
import matplotlib as mpl
import numpy as np
from datetime import datetime
from liegroups import SO3
from matplotlib import pyplot as plt
from mujoco._functions import mj_differentiatePos, mj_step
from mujoco._structs import MjModel, MjData, MjOption
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError, MissingMandatoryValue
from tqdm import tqdm

import dynamics as dyn
import transformations as tf
import utilities as utils
import visualization as vis
from core import SimulationConfig, generate_model_data, autoinstantiate


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
#            bi | body's principal frame
#            bj | frame attached to a body's joint
#       a/ai/aj | body's parent itself or its body/principal/joint frame
#       c/ci/cj | body's child itself or its body/principal/joint frame
#             q | joint space
#


# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams['axes.xmargin'] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=5, suppress=True)


def simulate(m: MjModel,
             d: MjData,
             gt_mass_distr_path: Union[str, PathLike],
             logger, planner, controller,  # TODO: annotate late... make a BaseModule or something and use Protocol or Generic, maybe...
             ):

    # Get ids and indices for the sake of convenience
    lastlink_id = utils.get_element_id(m, "body", "link6")
    sen_site_id = utils.get_element_id(m, "site", "target/ft_sensor")
    obj_id = utils.get_element_id(m, "body", "target/object")

    linacc_x_idx = utils.get_sensor_measurement_idx(m, "sensor", "linacc_x_obj")
    frc_sen_idx = utils.get_sensor_measurement_idx(m, "sensor", "target/force")
    trq_sen_idx = utils.get_sensor_measurement_idx(m, "sensor", "target/torque")
    ft_sen_idx = frc_sen_idx + trq_sen_idx

    # Join the spatial inertia matrices of bodies later than the last link into its spatial inertia matrix so that dyn.inverse() can consider the bodies' inertia later =========================
    _simats_bi_b = dyn.compose_spatial_inertia_matrices(m.body_mass, m.body_inertia)
    # Convert sinert_i to sinert_b rel2 the body frame
    poses_b_bi = tf.posquat2SE3s(m.body_ipos, m.body_iquat)
    poses_x_b = tf.posquat2SE3s(d.xpos, d.xquat)
    _simats_b_b = dyn.transfer_simats(poses_b_bi, _simats_bi_b)
    simats_b_b = _simats_b_b[:lastlink_id+1]

    poses_b_bi = tf.posquat2SE3s(m.body_ipos, m.body_iquat)

    print(f"{SO3.from_quaternion((1, 0, 0, 0)).as_matrix()=}")
    identity_quats = [[1, 0, 0, 0] for _ in range(len(m.jnt_pos))]
    poses_b_bj = tf.posquat2SE3s(m.jnt_pos, identity_quats)
    #print(f"{poses_b_bj=}")

    # Join the spatial inertia matrices of bodies later than, or fixed relative to,
    # link6 to the matrix of link6 so that dyn.inverse() can consider the bodies'
    # inertia later
    #link6_id = mj_name2id(m, mjtObj.mjOBJ_BODY, "link6")
    pose_x_lastlink = poses_x_b[lastlink_id]
    #simats_b_b = _simats_b_b[:link6_id+1]

    for p_x_b, _sim_b_b in zip(poses_x_b[lastlink_id+1:], _simats_b_b[lastlink_id+1:]):
        #p_link6_b = pose_x_link6.inv().dot(p_x_b)
        p_lastlink_b = pose_x_lastlink.inv().dot(p_x_b)
        _sim_lastlink_b = dyn.transfer_simats(p_lastlink_b, _sim_b_b)
        simats_b_b[lastlink_id] += _sim_lastlink_b

    mom_i = np.array([*simats_b_b[-1, 3:, 3:].diagonal(),
                      *simats_b_b[-1, 3 , 4:],
                       simats_b_b[-1, 4 , 5 ]])

#    print("Target object's inertial parameters wr2 its body frame ======\n"
#         f"    Mass:               {m.body_mass[-1]}\n"
#         f"    First moments:      {SO3.vee(simats_b_b[-1, 3:, :3])}\n"
#         f"    Moments of inertia: {mom_i}\n")

    # Configure SE3 of child frame wr2 parent frame (M_{i, i - 1} in MR)
    hposes_a_b = tf.posquat2SE3s(m.body_pos, m.body_quat)

    # ここのユニットスクリューの定義の仕方について疑問が湧いた。
    # Sec. 3.3.2.2 を見るとスクリューの定義にも参照座標系からみた pose が関わって
    # いるんだけれど、下記の処置だと軸を揃えただけ、つまり姿勢は考慮しているが、
    # 並進変位は考慮していないように見える。なので並進変位を考慮したスクリュー軸の#
    # 定義に変更したい。
    # 
    # ではどうすればよいか？
    # スクリュー軸は関節の動作の軸に一致するので、joint frame を定義して、
    # それを body とか principal frame を参照するよう座標変換するのが良さそう。
    # jnt_pos と jnt_axis ってので pos and axis of a joint local to the body は取れるようなので、pos_b_b と rot?_b_bj を作ってそれを pose_b_bj に合成して、それを Sec. 3.3.2.2 に従って正規化してスクリュー軸に定義するってのはできそう

    if True:
        # Obtain unit screw wr2 each link = body (A_{i} in MR)
        # NOTE: m.jnt_axis of shape (m.njnt, 3) express the directions of joints axes wr2 {b},
        # whih mean the axes is considered as the joints' orientational displacements wr2 {b}
        uscrew_b_b = np.zeros((m.body_jntnum.sum(), 6))  # bb = (11, 22, ..., 66)
        for b, (jnt_type, jnt_ax) in enumerate(zip(m.jnt_type, m.jnt_axis), 0):
            # Instances of ligroups SE3 classes assume the first 3 elements of
            # screw axes are for translation and the last 3 elements for rotation
            if 2 == jnt_type:  # transtation axis
                uscrew_b_b[b, :3] += jnt_ax
            elif 3 == jnt_type:  # rotation axis
                uscrew_b_b[b, 3:] += jnt_ax
            else:
              raise TypeError("Only slide or hinge joints, represented as 2 or 3 for m.jnt_type, are supported.")

    # Set up dynamics related variables =======================================
    # (d)twist vectors for the worldbody to be used for inverse dynamics
    twist_x_x = np.zeros(6)
    gacc_x = np.zeros(6)
    gacc_x[:3] = -1 * MjOption().gravity
    dtwist_x_x = gacc_x.copy()

    # Dictionary to be converted to a .json file for training
    aabb_scale = 1.28
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
        tgt_ctrl, _, _, _= dyn.inverse(
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
        measurements= d.sensordata.copy()
        # Scale sampled normalized coordinates ∈ (-1, 1) in wisp to the maximum
        # length of an axis-aligned bounding box of the object.
        # Camera pose rel. to the object
        pose_x_obj = tf.trzs2SE3(d.xpos[obj_id], d.xmat[obj_id])
        pose_x_cam = tf.trzs2SE3(d.cam_xpos[logger.cam_id],
                                 d.cam_xmat[logger.cam_id])
        pose_obj_cam = pose_x_obj.inv().dot(pose_x_cam)
        # FT sensor pose rel. to the object
        pose_x_sen = tf.trzs2SE3(d.site_xpos[sen_site_id], d.site_xmat[sen_site_id])
        pose_sen_obj = pose_x_sen.inv().dot(pose_x_obj)

        # Compute (d)twists using dyn.inverse() again to validate the method by
        # comparing derived acceleration and force/torque with their sensor
        # measurements later
        traj = np.stack((d.qpos, d.qvel, d.qacc))
        _, _, twists, dtwists = dyn.inverse(
            traj, hposes_a_b, simats_b_b, uscrew_b_b, twist_x_x, dtwist_x_x)

        # First-order time derivative - - - - - - - - - - - - - - - - - - - - -
        twist_obj_obj = twists[-1]
        twist_sen_obj = pose_sen_obj.adjoint() @ twist_obj_obj  # Eq. 3.83-84 in Modern Robotics
#        linvel_sen_obj = compute_linvel(pose_sen_obj, twist_obj_obj, coord_xfer_twist=True)

        # Second-order time derivative - - - - - - - - - - - - - - - - - - - - 
        dtwist_obj_obj = dtwists[-1]
        dtwist_sen_obj = dyn.coordinate_transform_dtwist(
            pose_sen_obj, twist_sen_obj, dtwist_obj_obj)  # , coord_xfer_twist=True)
        linacc_sen_obj = dyn.compute_linacc(  # Not \dot{v} but \ddot{p} in Modern Robotics
            pose_sen_obj, twist_sen_obj, dtwist_sen_obj)  # , coord_xfer_twist=True)

        # Retrieve force and torque measurements
        ft_sen = measurements[ft_sen_idx]
        #linacc_x_obj = measurements[linacc_x_idx]
        #linaccl_sen_obj = pose_sen_x.as_matrix() @ linacc_x_obj
        # ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 検証用コード追加ゾーン ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 
#        v_sen_obj, w_sen_obj = np.split(twist_sen_obj, 2)
#        dv_sen_obj, dw_sen_obj = np.split(dtwist_sen_obj, 2)
#        skewed_w_sen_obj = SO3.wedge(w_sen_obj)
#        skewed_dw_sen_obj = SO3.wedge(dw_sen_obj)
#        # 
#        _linacc1_sen_obj = dv_sen_obj + skewed_w_sen_obj @ v_sen_obj
#        _linacc2_sen_obj = skewed_dw_sen_obj @ pose_sen_obj.trans \
#                         + skewed_w_sen_obj @ skewed_w_sen_obj @ pose_sen_obj.trans  # element-wise part
#        _linacc_sen_obj = _linacc1_sen_obj + _linacc2_sen_obj
#
#        # Retrieve force and torque measurements
#        total_mass = ft_sen[:3] / linacc_sen_obj  # linacc_sen_obj would be wrong...
#        _total_mass = ft_sen[:3] / _linacc_sen_obj  # linacc_sen_obj would be wrong...
#

        if frame_count <= d.time * logger.fps:
            # ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 今使ってる検証用セクション ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 
            #print(f"{measurements.shape=}")
            twist_link6_link6 = twists[-1]  # 上だと _obj_obj だとしてるけど、こう考えたほうが正確
            twist_link6plus_link6plus = twists[-1]  # もっというと、上だと _obj_obj だとしてるけど、こう考えたほうが正確

            pose_sen_x = pose_x_sen.inv()
            pose_sen_lastlink = pose_sen_x.dot(pose_x_lastlink)

   #         print(f"{pose_sen_link6.rot=}")


            #twist_sen_link6 = pose_sen_lastlink.adjoint() @ twist_lastlink_lastlink

            # obj と link 
            #twist_sen_sen = twist_link6_link6

            #print(f"{total_mass=}")
            #print(f"{_total_mass=}")
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
    except ConfigAttributeError:  # if read_config not provided on cli, cli_cfg
        yaml_cfg = {}             # does not have it as its attribute, so using
                                  # this error rather than MissingMandatoryValue

    cfg = OmegaConf.merge(cfg, yaml_cfg, cli_cfg)

    try:
        OmegaConf.save(cfg, cfg.write_config)
    except MissingMandatoryValue:
        pass

    # Fill a potentially missing field of a logger configulation
    try:
        cfg.logger.dataset_dir
    except MissingMandatoryValue:
        cfg.logger.dataset_dir = Path.cwd() / "datasets" / cfg.target_name

    # Generate data structures
    m, d, gt_mass_distr_path = generate_model_data(cfg)
    # Instantiate necessary classes
    logger = autoinstantiate(cfg.logger, m, d)
    planner = autoinstantiate(cfg.planner, m, d)
    controller = autoinstantiate(cfg.controller, m, d)

    simulate(m, d, gt_mass_distr_path, logger, planner, controller)
