from functools import partial

import matplotlib as mpl
import numpy as np
from liegroups import SE3
from matplotlib import pyplot as plt
from mujoco._functions import mj_differentiatePos, mj_step
from mujoco._structs import MjModel, MjData, MjOption
from tqdm import tqdm

import dynamics as dyn
import visualization as vis
from transformations import Poses
from sensors import Sensors
from utilities import get_element_id


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
#    ll/lli/llj | last link itself or its body/principal/joint frame
#             x | world frame (x ∈ b)
#             q | joint space
#
#  NOTE: 's' follows the descriptor part of a variable's name to clarify that
#        the variable contains multiple descriptors.
#
#        ┏━━━━━━━━━━━━ Body namespace: "b"ody and its p"a"rent body ━━━━━━━━━━━━┓
#
# Bodies: x, link1 (firstlink), ..., link6 or sth (lastlink), attachment, object
#
#                                   ┗━ "l"ast"l"ink merged with the later ones ━┛
#
#        ┗━━ Link namespace: "l"ink and its parent body (= "k", prior to 'l') ━━┛
#


# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams['axes.xmargin'] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=5, suppress=True)


def simulate(m: MjModel,
             d: MjData,
             logger, planner, controller,  # TODO: annotate late... make a BaseModule or something and use Protocol or Generic, maybe...
             ):

    # Instantiate register classes ================================================
    poses = Poses(m, d)
    sensors = Sensors(m, d)

    # Get ids and indices for the sake of convenience =============================
    id_ll = get_element_id(m, "body", "link6")  # l(ast) l(ink)
    id_x2ll = slice(0, id_ll + 1)

    # Join the spatial inertia matrices of bodies later than the last link into the
    # spatial inertia matrix of the link so that dyn.inverse() can consider the
    # bodies' inertia =============================================================
    pose_x_obj = poses.get_x_("body", "target/object")
    pose_obj_obji = poses.get_b_biof("target/object")
    pose_x_obji = pose_x_obj.dot(pose_obj_obji)
    # FT sensor pose rel. to the object
    pose_x_sen = poses.get_x_("site", "target/ft_sensor")
    pose_sen_obj = pose_x_sen.inv().dot(pose_x_obj)
    pose_sen_obji = pose_x_sen.inv().dot(pose_x_obji)
    pose_x_ll = poses.x_b[id_ll]  # dynamic
    pose_ll_llj = poses.l_lj[id_ll]  # static
    # NOTE: Variables below should be declared not here but whenever neccessary.
    # pose_x_llj = pose_x_ll.dot(pose_ll_llj)  # static, should be dynamic tho
    # pose_sen_llj = pose_x_sen.inv().dot(pose_x_llj)  # dynamic, should be static tho

    # Get unit screws wr2 link joints =============================================
    uscrews_lj = []
    for t, ax in zip(m.jnt_type, m.jnt_axis):
        us_lj = np.zeros(6)
        if 2 == t:  # slider joint
            us_lj[:3] += ax
        elif 3 == t:  # hinge joint
            us_lj[3:] += ax
        else:
            raise TypeError("Only slide or hinge joints, represented as 2 or 3 "
                            "for an element of m.jnt_type, are supported.")

        uscrews_lj.append(us_lj)
    uscrews_lj = np.array(uscrews_lj)

    # Transfer the reference frame where each link's spatial inertia matrix is de-
    # fined from the body principal frame to the joint frame ======================
    # 下のメソッドが出力するのはボディの慣性座標系で記述された空間慣性テンソル 
    simats_bi_b = dyn.get_spatial_inertia_matrix(m.body_mass,
                                                 m.body_inertia,
                                                 )

    simats_lj_l = []
    for pose_lj_li, simat_li_l in zip(poses.lj_li, simats_bi_b[id_x2ll]):  # x~last
        simats_lj_l.append(dyn.transfer_simat(pose_lj_li, simat_li_l))

    simats_lj_l = np.array(simats_lj_l)

    # Join the spatial inertia matrices of the bodies later than the last link to
    # its spatial inertia matrix so that dyn.inverse() can consider the bodies'
    # inertia =====================================================================

    simat_sen_obj = np.zeros((6, 6))

    for pose_x_bi, simat_bi_b in zip(poses.x_bi[id_ll+1:], simats_bi_b[id_ll+1:]):
        # "b" here is ∈ {attachment, object}
        pose_x_llj = pose_x_ll.dot(pose_ll_llj)
        pose_bi_llj = pose_x_bi.inv().dot(pose_x_llj)
        simat_llj_b = dyn.transfer_simat(pose_bi_llj.inv(), simat_bi_b)
        simat_sen_obj += simat_llj_b
        simats_lj_l[id_ll] += simat_llj_b

    # Get link joints' home poses wr2 their parents' joint frame ==================
    hposes_lj_kj = [SE3.identity()]  # for worldbody
    for k in range(m.njnt):
        hpose_kj_k = poses.l_lj[k].inv()
        hpose_l_lj = poses.l_lj[k+1]
        hpose_k_l = poses.a_b[k+1]
        hpose_kj_lj = hpose_kj_k.dot(hpose_k_l.dot(hpose_l_lj))
        hposes_lj_kj.append(hpose_kj_lj.inv())

    # Set some arguments of dyn.inverse() which dose not evolve along time ========
    gacc_x = -1 * np.array([*MjOption().gravity, 0, 0, 0])
    inverse = partial(dyn.inverse,
                      hposes_body_parent=hposes_lj_kj,
                      simats_body=simats_lj_l,
                      uscrews_body=np.array(uscrews_lj),
                      twist_0=np.zeros(6),
                      dtwist_0=gacc_x,
                      )


    # Set a random number generator ===========================================
    rng = np.random.default_rng()
    rng.standard_normal(10)  

    # Prepare data containers =================================================
    res_qpos = np.empty(m.nu)
    tgt_trajectory = []
    trajectory = []
    fts_sen = []
    time = []
    linacc_sen_obji = []
    frame_count = 0
    regressors = []

    # Main loop ===============================================================
    for step in tqdm(range(planner.n_steps), desc="Progress"):
        # Compute actuator controls and evolute the simulatoin
        tgt_traj = planner.plan(step)
        tgt_ctrl, _, _, _= inverse(tgt_traj)

        # Get current sensor measurements of joint variables by calling d.q***
        qpos, qvel, qacc = d.qpos, d.qvel, d.qacc

        perturb_joint_variables = False  # True
        if perturb_joint_variables:
            # Perturb the joint variables
            pos_std = 0.0001
            vel_std = 0.001
            acc_std = 0.01
            qpos += pos_std * rng.standard_normal(qpos.shape)
            qvel += vel_std * rng.standard_normal(qvel.shape)
            qacc += acc_std * rng.standard_normal(qacc.shape)

        act_traj = np.stack((qpos, qvel, qacc))
        _, _, twists_lj_l, dtwists_lj_l = inverse(act_traj)

        if frame_count <= d.time * logger.fps:
            time.append(d.time)
            tgt_trajectory.append(tgt_traj)
            trajectory.append(act_traj)

            # Get (d)twist_sen, and linacc_sen_obj for later verification
            pose_sen_llj = pose_x_sen.inv().dot(pose_x_ll.dot(pose_ll_llj))
            twist_llj = twists_lj_l[id_ll]
            twist_sen = pose_sen_llj.adjoint() @ twist_llj
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            dtwist_llj = dtwists_lj_l[id_ll]
            pose_sen_llj_dadjoint = SE3.curlywedge(twist_sen) @ pose_sen_llj.adjoint()
            dtwist_sen = pose_sen_llj_dadjoint @ twist_llj \
                       + pose_sen_llj.adjoint() @ dtwist_llj
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            perturb_sensor_twist = True  # False
            if perturb_sensor_twist:
                v_std = 1.0 * 1e-1
                w_std = 1.0 * 1e-3
                dv_std = 1.0 * 1e-0
                dw_std = 1.0 * 1e-2

                v_noise = v_std * rng.standard_normal(3)
                w_noise = w_std * rng.standard_normal(3)
                dv_noise = dv_std * rng.standard_normal(3)
                dw_noise = dw_std * rng.standard_normal(3)

                twist_sen += np.concatenate((v_noise, w_noise))
                dtwist_sen += np.concatenate((dv_noise, dw_noise))

            linacc_sen_obji.append(
                dyn.extract_linacc_frame_transferred(twist_sen,
                                                     dtwist_sen,
                                                     pose_sen_obji)
            )

            # Get force-torque measurements
            force = sensors.get("force")
            torque = sensors.get("torque")
            wrench = np.concatenate([force, torque], axis=None)

            perturb_wrench = True  # False
            if perturb_wrench:
                f_std = 1.0 * 1e-1
                t_std = 1.0 * 1e-3
                f_noise = f_std * rng.standard_normal(3)
                t_noise = t_std * rng.standard_normal(3)
                wrench += np.concatenate((f_noise, t_noise))

            fts_sen.append(wrench)

            regressor = dyn.get_regressor_matrix(twist_sen, dtwist_sen)
            regressors.append(regressor)

            # Writing a single frame of a dataset =============================
            file_name = f"{frame_count:04}.png"
            logger.render(d, file_name)  # logger.cam_id is selected internally

            # Log NeMD ingredients ============================================
            # Items which need to be computed at every frame recoding
            pose_obj_cam = pose_x_obj.inv().dot(poses.x_cam[logger.cam_id])

            frame = dict(
                file_path=str(logger.image_dir / file_name),
#                pose_obj_cam=pose_obj_cam.as_matrix().T.tolist(),
                transform_matrix=pose_obj_cam.as_matrix().tolist(),
                pose_sen_obj=pose_sen_obj.as_matrix().tolist(),
                pose_sen_obji=pose_sen_obji.as_matrix().tolist(),
                twist_sen=twist_sen.tolist(),
                dtwist_sen=dtwist_sen.tolist(),
                ft_sen=fts_sen[-1].tolist(),
                linacc_sen_obji=linacc_sen_obji[-1].tolist(),
#                aabb_scale=[aabb_scale],
                )

            logger.transform["frames"].append(frame)
            frame_count += 1

        # Get residual of state
        mj_differentiatePos(# Use this func to differenciate quat properly
            m,  # MjModel
            res_qpos,  # data container for the residual of qpos
            m.nu,  # idx of a joint up to which res_qpos are calculated
            qpos,  # current qpos
            tgt_traj[0],                 # target qpos or next qpos to calkculate dqvel
        )

        #res_state = np.concatenate((res_qpos, tgt_traj[1] - d.qvel))
        res_state = np.concatenate((res_qpos, tgt_traj[1] - qvel))
        # Compute and set control, or actuator inputs
        d.ctrl = tgt_ctrl - controller.gain_matrix @ res_state

        mj_step(m, d) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Evolve the simulation

    # Convert lists of logged data into ndarrays ==============================
    tgt_trajectory = np.array(tgt_trajectory)
    trajectory = np.array(trajectory)
    linacc_sen_obji = np.array(linacc_sen_obji)
    frame_iter = np.arange(frame_count)
    fts_sen = np.array(fts_sen)
    regressors = np.array(regressors)

    solution = np.linalg.lstsq(regressors.reshape(-1, 10),
                               fts_sen[:, :].reshape(-1),
                               rcond=None,
                               )

    identified = solution[0]
    residuals = solution[1]

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
    vis.ax_plot_lines(acc_ft_axes[0], frame_iter, linacc_sen_obji, "recovered_linacc_sen_obji [m/s/s]")
    vis.ax_plot_lines(acc_ft_axes[1], frame_iter, fts_sen[:, :3], "frc_sen [N]")
    vis.ax_plot_lines(acc_ft_axes[2], frame_iter, fts_sen[:, 3:], "trq_sen [N*m]")
    for ax in acc_ft_axes:
        ax.hlines(0.0, frame_iter[0], frame_iter[-1], ls="dashed", alpha=0.5)

#    @dataclass
#    class Plot:
#        fig: mpl.figure.Figure
#        ax: mpl.axes.Axes
#
#    mass_plot = Plot(*plt.subplots(1, 1, sharex="col", tight_layout=True))
#    mass_plot.fig.suptitle("regressed mass")
#    mass_plot.ax.plot(solutions[:, 0])
    #mass_plot.ax.plot(np.linalg.norm(fts_sen[:, :3]/twists_sen[:, :3], axis=1))


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

#    lin_fig, lin_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
#    vis.ax_plot_lines(lin_axes[0], time, _linacc_sen_obji, "_linacc_sen_obji")
#    vis.ax_plot_lines(lin_axes[1], time, linacc_sen_obji, "linacc_sen_obji")

    plt.show()

    return dict(lstsq=identified, residuals=residuals)
