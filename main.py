from copy import deepcopy
from functools import partial
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
from liegroups import SE3
from matplotlib import pyplot as plt
from mujoco._functions import mj_differentiatePos, mj_step
from mujoco._structs import MjModel, MjData, MjOption
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError, MissingMandatoryValue
from tqdm import tqdm

import dynamics as dyn
import visualization as vis
from core import SimulationConfig, generate_model_data, autoinstantiate
from transformations import Poses, compose, differentiate_adjoint, homogenize
from utilities import Measurements, get_element_id


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
#
#     reference |
#     /descried | Definition
# --------------+-------------
#             b | body itself or its frame (refer to the official documentation)
#            bi | body's principal frame
#            bj | frame attached to a body's joint
#       a/ai/aj | body's parent itself or its body/principal/joint frame
#       l/li/lj | link itself or its body/principal/joint frame
#       k/ki/kj | link's parent itself or its body/principal/joint frame
#       m/mi/mj | merged lastlink itself or its body/principal/joint frame
#             x | world frame (x ∈ b)
#             q | joint space
#
#  NOTE: 's' follows the descriptor part of a variable's name to clarify that
#        the variable contains multiple descriptors.
#
#        ┏━━━━━━━━━━━━━━━ Body namespace: "b"ody and its p"a"rent body ━━━━━━━━━━━━━━━━┓
#
# Bodies: x, link1 (firstlink), link2, ..., link6 or sth (lastlink), attachment, object
#
#                                          ┗━━ lastlink "m"erged with the later ones ━━┛
#
#        ┗━ Link namespace: "l"ink and its parent body (= prior to 'l', which is "k") ━┛
#


# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams['axes.xmargin'] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=5, suppress=True)


def simulate(m: MjModel,
             d: MjData,
             logger, planner, controller, poses, measurements,  # TODO: annotate late... make a BaseModule or something and use Protocol or Generic, maybe...
             ):

    # Get ids and indices for the sake of convenience =============================
    fl_id = get_element_id(m, "body", "link1")  # f(irst) l(ink)
    ll_id = get_element_id(m, "body", "link6")  # l(ast) l(ink)
    fl2ll_idx = slice(fl_id, ll_id + 1)

    # Join the spatial inertia matrices of bodies later than the last link into the
    # spatial inertia matrix of the link so that dyn.inverse() can consider the
    # bodies' inertia =============================================================
    simats_bi_b = dyn.get_spatial_inertia_matrix(m.body_mass, m.body_inertia)
    pose_x_mi = compose(d.subtree_com, d.ximat)[ll_id]
    pose_x_obj = poses.x_("body", "target/object")
    pose_obj_cam = pose_x_obj.inv().dot(poses.x_cam[logger.cam_id])
    # FT sensor pose rel. to the object
    pose_x_sen = poses.x_("site", "target/ft_sensor")
    pose_sen_obj = pose_x_sen.inv().dot(pose_x_obj)
    pose_sen_mi = pose_x_sen.inv().dot(pose_x_mi)  # confirmed static

    # Get simat_mi_m and simat_li_l ===============================================
    simat_mi_m = np.zeros((6, 6))
    for b in range(ll_id, m.nbody):  # "b" here is for {ll, attachment, object}
        pose_mi_bi = pose_x_mi.inv().dot(poses.x_bi[b])
        simat_mi_m += dyn.transfer_simat(pose_mi_bi, simats_bi_b[b])
    # simat_lli_ll excluded simat_mi_m included
    simats_li_l = np.vstack([simats_bi_b[:ll_id], np.expand_dims(simat_mi_m, 0)])

    #print(f"{simats_li_l[-1]=}")  # looks working fine
    mass = simat_mi_m[0, 0]

    # Get hhposes_ki_li ===========================================================
    hpose_x_ll = compose(d.xpos, d.xmat)[ll_id]
    hposes_l_li = poses.b_bi[:ll_id] + [hpose_x_ll.inv().dot(pose_x_mi)]
    hposes_li_ki = [SE3.identity()]  # for worldbodl
    for k, hpose_k_l in enumerate(poses.a_b[fl2ll_idx]):  # num_iter. == num_links
        hpose_ki_k = hposes_l_li[k].inv()
        hpose_l_li = hposes_l_li[k+1]
        hpose_ki_li = hpose_ki_k.dot(hpose_k_l.dot(hpose_l_li))
        hposes_li_ki.append(hpose_ki_li.inv())

    #print(f"{hposes_li_ki[-1]=}")  # looks working fine

    # Get uscrews_li_lj ===========================================================
    id_quat = [1, 0, 0, 0]
    id_quats = [id_quat for _ in range(m.njnt)]
    hposes_l_lj = compose(m.jnt_pos, np.vstack(id_quats))
    uscrews_li_lj = []
    for i, (p_l_li, p_l_lj) in enumerate(zip(hposes_l_li, hposes_l_lj)):
        us_lj_lj = np.zeros(6)
        if 2 == m.jnt_type[i]:  # slider joint
            us_lj_lj[:3] += m.jnt_axis[i]
        elif 3 == m.jnt_type[i]:  # hinge joint
            us_lj_lj[3:] += m.jnt_axis[i]
        else:
            raise TypeError("Only slide or hinge joints, represented as 2 or 3 for an element of m.jnt_type, are supported.")

        p_li_lj = p_l_li.inv().dot(p_l_lj)
        uscrews_li_lj.append(p_li_lj.adjoint() @ us_lj_lj)

    #print(f"{uscrews_li_lj=}")  # looks working fine

    # Set some arguments of dyn.inverse() which dose not evolve along time ========
    gacc_x = -1 * np.array([*MjOption().gravity, 0, 0, 0])
    inverse = partial(dyn.inverse,
                      hposes_body_parent=hposes_li_ki,
                      simats_bodyi=simats_li_l,
                      uscrews_bodyi=np.array(uscrews_li_lj),
                      twist_0=np.zeros(6),
                      dtwist_0=gacc_x.copy(),
                      )

    # Prepare data containers =================================================
    res_qpos = np.empty(m.nu)
    tgt_trajectory = []
    trajectory = []
    linaccs_sen_obj = []
    fts_sen = []
    time = []
    frame_count = 0

    # For test
    frcs_sen = []

    # Main loop ===============================================================
    for step in tqdm(range(planner.n_steps), desc="Progress"):
        # Compute actuator controls and evolute the simulatoin
        tgt_traj = planner.plan(step)
        tgt_ctrl, _, _, _= inverse(tgt_traj)
        # Residual of state
        mj_differentiatePos(# Use this func to differenciate quat properly
            m,  # MjModel
            res_qpos,  # data container for the residual of qpos
            m.nu,  # idx of a joint up to which res_qpos are calculated
            d.qpos,  # current qpos
            tgt_traj[0],  # target qpos or next qpos to calkculate dqvel
        )

        res_state = np.concatenate((res_qpos, tgt_traj[1] - d.qvel))
        # Compute and set control, or actuator inputs
        d.ctrl = tgt_ctrl - controller.control_gain @ res_state

        mj_step(m, d) # Evolve the simulation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Compute (d)twists using dyn.inverse() again to validate the method by
        # comparing derived acceleration and force/torque with their sensor
        # measurements later
        act_traj = np.stack((d.qpos, d.qvel, d.qacc))   # d.qSTH is the same as 
                                                        # reading joint variable
                                                        # sensor measurements
                                                        # actually
        _, _, twists_li_l, dtwists_li_l = inverse(act_traj)

        # ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 検証用コード追加ゾーン ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 

        twist_mi = twists_li_l[-1]
        twist_sen = pose_sen_mi.adjoint() @ twist_mi

        dtwist_mi = dtwists_li_l[-1]
        pose_sen_mi_dadjoint = differentiate_adjoint(twist_sen,
                                                     pose_sen_mi.adjoint(),
                                                     )

        dtwist_sen = pose_sen_mi_dadjoint @ twist_mi \
                   + pose_sen_mi.adjoint() @ dtwist_mi

        linacc_sen_obj = dyn.get_linear_acceleration(twist_sen,
                                                     dtwist_sen,
                                                     pose_sen_obj,
                                                     )

        if frame_count <= d.time * logger.fps:
            # Attempt to compute linacc_x_obj from linacc_x_sen and something other
            # Step 1. recover trans_x_obj
            _trans_x_obj = pose_x_sen.dot(homogenize(pose_sen_obj.trans))
            #print(f"{np.allclose(pose_x_obj.trans, _trans_x_obj[:3])=}")  # True confirmed

            linacc_x_obj = measurements.get("linacc_x_obj")
            linacc_sen_obj = pose_x_sen.inv().dot(homogenize(linacc_x_obj, 0))[:3]
            linaccs_sen_obj.append(linacc_sen_obj)

        # ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ 検証用コード追加ゾーン ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ 

            # Log velocity components relative to the sensor frame
            tgt_trajectory.append(tgt_traj)
            trajectory.append(act_traj)
            time.append(d.time)
            # force-torque
            frc_sen = measurements.get("target/force")
            trq_sen = measurements.get("target/torque")
            fts_sen.append(np.concatenate((frc_sen, trq_sen)))
            frcs_sen.append(frc_sen)

            # Writing a single frame of a dataset =============================
            logger.renderer.update_scene(d, logger.cam_id)
            bgr = logger.renderer.render()[:, :, [2, 1, 0]]
            # Make an alpha mask to remove the black background
            alpha = np.where(np.all(bgr == 0, axis=-1), 0, 255)[..., np.newaxis]
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
                twist_sen_obj=twist_sen.tolist(),
                dtwist_sen_obj=dtwist_sen.tolist(),
#                obj_linacc_sen=obj_linacc_sen,
                linacc_sen_obj=linaccs_sen_obj[-1].tolist(),
                ft_sen=fts_sen[-1].tolist(),
#                aabb_scale=[aabb_scale],
                )

            logger.transform["frames"].append(frame)

            # Sampling for NeMD terminated while "frame_count" incremented
            frame_count += 1

    logger.finish()  # video and dataset json generated

    # Convert lists of logged data into ndarrays ==============================
    tgt_trajectory = np.array(tgt_trajectory)
    trajectory = np.array(trajectory)
    linaccs_sen_obj = np.array(linaccs_sen_obj)
    #fts_sen = np.array(fts_sen)
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
    fts_sen = np.array(fts_sen)
    vis.ax_plot_lines(acc_ft_axes[0], frame_iter, linaccs_sen_obj, "linacc_sen_obj")
    vis.ax_plot_lines(acc_ft_axes[1], frame_iter, fts_sen[:, :3], "frc_sen")
    vis.ax_plot_lines(acc_ft_axes[2], frame_iter, fts_sen[:, 3:], "trq_sen")
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

#    frc_fig, frc_axes = plt.subplots(3, 1, sharex="col", tight_layout=True)
#    vis.ax_plot_lines(frc_axes[0], time, frc_x_sen, "frc_x_sen")
#    vis.ax_plot_lines(frc_axes[1], time, frc_x_obj, "frc_x_obj")

#    lin_fig, lin_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
#    vis.ax_plot_lines(lin_axes[0], time, __linvel_sen_obj, "__linvel_sen_obj")
#    vis.ax_plot_lines(lin_axes[1], time, _linvel_sen_obj, "_linvel_sen_obj")

    plt.show()


if __name__ == "__main__":
    # Load configuraion ===========================================================
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

    # Generate mujoco data structures and aux data ================================
    m, d, target_object_aabb_scale, gt_mass_distr_file_path = generate_model_data(cfg)

    # Fill (potentially) missing fields of a logger configulation =================
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

    # Instantiate necessary classes ===============================================
    logger = autoinstantiate(cfg.logger, m, d)
    planner = autoinstantiate(cfg.planner, m, d)
    controller = autoinstantiate(cfg.controller, m, d)
    poses = Poses(m, d)
    measurements = Measurements(m, d)

    simulate(m, d, logger, planner, controller, poses, measurements)
