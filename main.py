from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
from liegroups import SO3, SE3
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
#             b | body itself or its frame (refer to the official documentation)
#            bi | body's principal frame
#            bj | frame attached to a body's joint
#       a/ai/aj | body's parent itself or its body/principal/joint frame
#       c/ci/cj | body's child itself or its body/principal/joint frame
#             l | link itself or its frame (l ∈ b)
#            li | link's principal frame
#            lj | frame attached to a link's joint
#             x | world frame (x ∈ b)
#             q | joint space
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━ "b"ody and its p"a"rent body ━━━━━━━━━━━━━━━━━━━━━━┓
#
#  x, link1 (firstlink), link2, ..., link6 or sth (lastlink), attachment, object
#
#                                   ┗━ lastlink "m"erged with the later ones ━┛
#
# ┗━━━━━━━━ "l"ink and its parent (= prior to 'l', which is "k") body ━━━━━━━━┛
#


# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams['axes.xmargin'] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=5, suppress=True)


def simulate(m: MjModel,
             d: MjData,
             logger, planner, controller,  # TODO: annotate late... make a BaseModule or something and use Protocol or Generic, maybe...
             ):

    # Get ids and indices for the sake of convenience =============================
    fl_id = utils.get_element_id(m, "body", "link1")  # f(irst) l(ink)
    ll_id = utils.get_element_id(m, "body", "link6")  # l(ast) l(ink)
    sen_site_id = utils.get_element_id(m, "site", "target/ft_sensor")
    obj_id = utils.get_element_id(m, "body", "target/object")

    x2ll_idx = slice(0, ll_id + 1)  # 0 for worldboidy
    fl2ll_idx = slice(fl_id, ll_id + 1)
    linacc_x_idx = utils.get_sensor_measurement_idx(m, "sensor", "linacc_x_obj")
    frc_sen_idx = utils.get_sensor_measurement_idx(m, "sensor", "target/force")
    trq_sen_idx = utils.get_sensor_measurement_idx(m, "sensor", "target/torque")
    ft_sen_idx = frc_sen_idx + trq_sen_idx

    # Join the spatial inertia matrices of bodies later than the last link into its spatial inertia matrix so that dyn.inverse() can consider the bodies' inertia =========================
    # この時点で _simats_bi_b には x, link1, ..., link6, attachment, object の simats が入っている
    _simats_bi_b = dyn.get_spatial_inertia_matrix(m.body_mass, m.body_inertia)  # all bodies

    simats_bi_b = _simats_bi_b[:ll_id+1].copy()
    poses_b_bi = tf.compose(m.body_ipos, m.body_iquat)  # len == 9
    pose_b_lli = poses_b_bi[ll_id]  # principal frame of the last link
    for p_b_bi, _sim_bi_b in zip(poses_b_bi[ll_id+1:], _simats_bi_b[ll_id+1:]):
        #p_link6_b = pose_x_link6.inv().dot(p_x_b)
        p_lli_bi = pose_b_lli.inv().dot(p_b_bi)
        _sim_lli_b = dyn.transfer_simat(p_lli_bi, _sim_bi_b)
        simats_bi_b[ll_id] += _sim_lli_b

    # TEST SECTION TO GENERATE THE LASTLINK MERGED WITH THE OBJECT >>>>>>>>>>>>>>>>

    # 最終的に欲しい物はなにか？
    # 1. simat_mi_m
    poses_x_bi = tf.compose(d.xipos, d.ximat)
    pose_x_mi = tf.compose(d.subtree_com, d.ximat)[ll_id]  # subtree_com ≒ subtree_xipos
    simat_mi_m = np.zeros((6, 6))
    for i in range(ll_id, m.nbody):
        simat_bi_b= _simats_bi_b[i]
        pose_x_bi = poses_x_bi[i]
        pose_mi_bi = pose_x_mi.inv().dot(pose_x_bi)
        simat_mi_b = dyn.transfer_simat(pose_mi_bi, simat_bi_b)
        simat_mi_m += simat_mi_b
    print(f"{simat_mi_m=}")  # うまくいったように見える

    # 2. pose_llj_mi
    pose_x_ll = poses_x_bi[ll_id]
    pose_llj_ll = SE3(SO3.identity(), m.jnt_pos[-1])
    pose_x_llj = pose_x_ll.dot(pose_llj_ll.inv())
    pose_llj_mi = pose_x_llj.inv().dot(pose_x_mi)
    print(f"{pose_llj_mi=}")  # うまくいったように見える

    #    mom_i = np.array([*simats_b_b[-1, 3:, 3:].diagonal(),
#                      *simats_b_b[-1, 3 , 4:],
#                       simats_b_b[-1, 4 , 5 ]])

#    print("Target object's inertial parameters wr2 its body frame ======\n"
#         f"    Mass:               {m.body_mass[-1]}\n"
#         f"    First moments:      {SO3.vee(simats_b_b[-1, 3:, :3])}\n"
#         f"    Moments of inertia: {mom_i}\n")

    # <<<<<<<<<<<<<<<< TEST SECTION TO GENERATE THE LASTLINK MERGED WITH THE OBJECT

    # Configure SE3 of child frame wr2 parent frame (M_{i, i - 1} in MR)
    hposes_k_l = tf.compose(m.body_pos, m.body_quat)[x2ll_idx]

    poses_l_li = poses_b_bi[fl2ll_idx]  # len == 6
    id_quat = [1, 0, 0, 0]
    id_quats = [id_quat for _ in range(m.njnt)]
    poses_l_lj = tf.compose(m.jnt_pos, np.vstack(id_quats))

    uscrew_li_lj = []
    for i, (p_l_li, p_l_lj) in enumerate(zip(poses_l_li, poses_l_lj)):
        us_lj_lj = np.zeros(6)
        if 2 == m.jnt_type[i]:  # slider joint
            us_lj_lj[:3] += m.jnt_axis[i]
        elif 3 == m.jnt_type[i]:  # hinge joint
            us_lj_lj[3:] += m.jnt_axis[i]
        else:
            raise TypeError("Only slide or hinge joints, represented as 2 or 3 for an element of m.jnt_type, are supported.")

        p_li_lj = p_l_li.inv().dot(p_l_lj)
        uscrew_li_lj.append(p_li_lj.adjoint() @ us_lj_lj)

    uscrew_li_lj = np.array(uscrew_li_lj)

    # Set up dynamics related variables =======================================
    # (d)twist vectors for the worldbody to be used for inverse dynamics
    twist_x_x = np.zeros(6)
    gacc_x = np.zeros(6)
    gacc_x[:3] = -1 * MjOption().gravity
    dtwist_x_x = gacc_x.copy()

    # Prepare data containers =================================================
    res_qpos = np.empty(m.nu)
    tgt_trajectory = []
    trajectory = []
    linaccs_sen_obj = []
    fts_sen = []
    time = []
    frame_count = 0

    #print(f"{len(hposes_k_l)=}")
    #print(f"{len(simats_bi_b)=}")
    #print(f"{len(uscrew_li_lj)=}")
    #print(f"{len(twist_x_x)=}")
    #print(f"{len(dtwist_x_x)=}")
    # Main loop ===============================================================
    for step in tqdm(range(planner.n_steps), desc="Progress"):
        # Compute actuator controls and evolute the simulatoin
        tgt_traj = planner.plan(step)
        tgt_ctrl, _, _, _= dyn.inverse(
            #tgt_traj, hposes_a_b, simats_b_b, uscrew_b_b, twist_x_x, dtwist_x_x)
            tgt_traj, hposes_k_l, simats_bi_b, np.array(uscrew_li_lj), twist_x_x, dtwist_x_x)
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
        pose_x_obj = tf.compose(d.xpos[obj_id], d.xmat[obj_id])
        pose_x_cam = tf.compose(d.cam_xpos[logger.cam_id], d.cam_xmat[logger.cam_id])
        pose_obj_cam = pose_x_obj.inv().dot(pose_x_cam)
        # FT sensor pose rel. to the object
        pose_x_sen = tf.compose(d.site_xpos[sen_site_id], d.site_xmat[sen_site_id])
        pose_sen_obj = pose_x_sen.inv().dot(pose_x_obj)

        # Compute (d)twists using dyn.inverse() again to validate the method by
        # comparing derived acceleration and force/torque with their sensor
        # measurements later
        traj = np.stack((d.qpos, d.qvel, d.qacc))   # actually the same as sensor measurements
        _, _, twists, dtwists = dyn.inverse(
            #tgt_traj, hposes_a_b, simats_b_b, uscrew_b_b, twist_x_x, dtwist_x_x)
            traj, hposes_k_l, simats_bi_b, np.array(uscrew_li_lj), twist_x_x, dtwist_x_x)

        # should be rewriten later >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        twists_li_l, dtwists_li_l = twists, dtwists
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
        linacc_x_obj = measurements[linacc_x_idx]
#
        #pose_sen_x = pose_x_sen.inv()
        _linaccl_sen_obj = pose_x_sen.inv().as_matrix() @ np.append(linacc_x_obj, 1)
        linacc_sen_obj = _linaccl_sen_obj[:-1]
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

            logger.transform["frames"].append(frame)

            # Log velocity components relative to the sensor frame
            tgt_trajectory.append(tgt_traj)
            trajectory.append(traj)
            linaccs_sen_obj.append(linacc_sen_obj)
            fts_sen.append(ft_sen)
            time.append(d.time)

            # Sampling for NeMD terminated while "frame_count" incremented
            frame_count += 1

    logger.finish()  # video and dataset json generated

#    qpos_meas, qvel_meas, qfrc_meas, ft_meas_sen, obj_vel_x, obj_acc_x = np.split(
#        sensordata, [1*m.nu, 2*m.nu, 3*m.nu, 4*m.nu, 5*m.nu], axis=1)

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
    # Load configuraion >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Generate mujoco data structures and aux data
    m, d, target_object_aabb_scale, gt_mass_distr_file_path = generate_model_data(cfg)

    # Fill (potentially) missing fields of a logger configulation >>>>>>>>>>>>>>>>>
    try:
        cfg.logger.target_object_aabb_scale
    except MissingMandatoryValue:
        cfg.logger.target_object_aabb_scale = float(target_object_aabb_scale)

    try:
        cfg.logger.gt_mass_distr_file_path
    except MissingMandatoryValue:
        cfg.logger.gt_mass_distr_file_path = gt_mass_distr_file_path

    try:
        cfg.logger.dataset_dir
    except MissingMandatoryValue:
        cfg.logger.dataset_dir = Path.cwd() / "datasets" / cfg.target_name
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Instantiate necessary classes
    logger = autoinstantiate(cfg.logger, m, d)
    planner = autoinstantiate(cfg.planner, m, d)
    controller = autoinstantiate(cfg.controller, m, d)

    simulate(m, d, logger, planner, controller)
