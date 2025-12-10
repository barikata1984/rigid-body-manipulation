import matplotlib as mpl
import numpy as np
from liegroups import SE3
from matplotlib import pyplot as plt
from mujoco._functions import mj_differentiatePos, mj_step
from mujoco._structs import MjData, MjModel, MjOption
from numpy import linalg as nla
from tqdm import tqdm

import dynamics as dyn
import visualization_ as vis
from dynamics import setup_robot_dynamics_parameters
from sensors import Sensors


# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams["axes.xmargin"] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=5, suppress=True)


class Simulation:
    m: MjModel
    d: MjData

    def __init__(self, m: MjModel, d: MjData, recorder, planner, controller):
        # Set a random number generator ===========================================
        rng = np.random.default_rng()
        rng.standard_normal(10)

        self.m = m
        self.d = d
        self.recorder = recorder
        self.planner = planner
        self.controller = controller

        self.fps = self.recorder.fps
        self.sensors = Sensors(self.m, self.d, self.fps)

        (
            self.poses,
            self.id_ll,
            self.pose_ll_llj,
            self.uscrews_lj,
            self.simats_lj_l,
            self.hposes_lj_kj,
            self.inverse,
        ) = setup_robot_dynamics_parameters(self.m, self.d)

        self.pose_obj_obji = self.poses.get_b_biof("target/object")
        self.pose_x_obj = self.poses.get_x_("body", "target/object")
        self.pose_x_obji = self.pose_x_obj.dot(self.pose_obj_obji)
        self.pose_x_ll = self.poses.x_b[self.id_ll]  # dynamic
        self.pose_x_sen = self.poses.get_x_("site", "target/ft_sensor")
        self.pose_sen_obj = self.pose_x_sen.inv().dot(self.pose_x_obj)
        self.pose_sen_obji = self.pose_x_sen.inv().dot(self.pose_x_obji)
        # NOTE: Variables below should be declared not here but whenever necessary.
        # self.pose_x_llj = pose_x_ll.dot(pose_ll_llj)  # static, should be dynamic tho
        # self.pose_sen_llj = pose_x_sen.inv().dot(pose_x_llj)  # dynamic, should be static tho

        # Data buffer and storages =============================================================
        self.qpos_err =  np.empty(m.nu)

        self.qpos_errors = []
        self.qvel_errors = []
        self.qacc_errors = []

        self.act_trajectory = []
        self.tgt_trajectory = []

        self.poses_sen_obj = []
        self.poses_sen_obji = []
        self.twists_sen = []
        self.dtwists_sen = []
        self.linaccs_sen_obji = []
        self.fts_sen = []
        self.regressors = []

        self.frame_count = 0
        self.frames = []
        self.file_paths = []
        self.time = []
        self.transform_matrices = []

    def run(self):
        if not self.recorder.videowriter.isOpened():
            print("Error: VideoWriter failed to open, inside simulation.")

        for step in tqdm(range(self.planner.n_steps), desc="Progress"):
            act_traj = np.stack(self.sensors.get("jointvars", perturbed=True))  # hape: (6,), (6,), (6,)0
            _, _, twists_lj_l, dtwists_lj_l = self.inverse(act_traj)

            # Compute actuator controls and evolute the simulation
            tgt_traj = self.planner.plan(step)

            if self.frame_count <= self.d.time * self.recorder.fps:
                self.time.append(self.d.time)
                self.tgt_trajectory.append(tgt_traj)
                self.act_trajectory.append(act_traj)

                # Get (d)twist_sen, and linacc_sen_obj for later verification
                pose_sen_llj = self.pose_x_sen.inv().dot(self.pose_x_ll.dot(self.pose_ll_llj))
                twist_llj = twists_lj_l[self.id_ll]
                twist_sen = pose_sen_llj.adjoint() @ twist_llj
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                dtwist_llj = dtwists_lj_l[self.id_ll]
                pose_sen_llj_dadjoint = SE3.curlywedge(twist_sen) @ pose_sen_llj.adjoint()
                dtwist_sen = pose_sen_llj_dadjoint @ twist_llj + pose_sen_llj.adjoint() @ dtwist_llj
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                linacc_sen_obji = dyn.extract_linacc_frame_transferred(twist_sen, dtwist_sen, self.pose_sen_obji)
                self.linaccs_sen_obji.append(linacc_sen_obji)

                # Get force-torque measurements
                force = self.sensors.get("force")
                torque = self.sensors.get("torque")
                wrench = np.concatenate([force, torque], axis=None)
                self.fts_sen.append(wrench)

                regressor = dyn.get_regressor_matrix(twist_sen, dtwist_sen)
                self.regressors.append(regressor)

                # Writing a single frame of a dataset =============================
                file_name = f"{self.frame_count:04}.png"
                self.recorder.render(self.d, file_name)  # recorder.cam_id is selected internally

                # Log NeMD ingredients ============================================
                # Items which need to be computed at every frame recoding
                pose_obj_cam = self.pose_x_obj.inv().dot(self.poses.x_cam[self.recorder.cam_id])

                self.file_paths.append(str(self.recorder.complete_image_dir / file_name))
                self.transform_matrices.append(pose_obj_cam.as_matrix().tolist())
                self.poses_sen_obj.append(self.pose_sen_obj.as_matrix().tolist())
                self.poses_sen_obji.append(self.pose_sen_obji.as_matrix().tolist())
                self.twists_sen.append(twist_sen.tolist())
                self.dtwists_sen.append(dtwist_sen.tolist())

                self.recorder.recursive_eval_data["frames"][f"{self.frame_count:04}"] = {"regressor": regressor.tolist(), "ft_sen": wrench.tolist()}

                self.frame_count += 1

            # Get residual of state
            mj_differentiatePos(  # Use this func to differenciate quat properly
                self.m,  # MjModel
                self.qpos_err,  # data container for the residual of qpos
                1,  # timestep used to numerically differentiate the pos
                tgt_traj[0],  # target qpos
                act_traj[0],  # actual qpos
            )

            # Get feedforward signal
            feedforward_ctrl, _, _, _ = self.inverse(tgt_traj)
            # Get feedback signal
            traj_err = act_traj- tgt_traj
            state_err = np.concatenate((self.qpos_err, traj_err[1]))
            feedback_ctrl = self.controller.gain_matrix @ state_err
            self.d.ctrl = feedforward_ctrl - feedback_ctrl

            self.qpos_errors.append(self.qpos_err)
            self.qvel_errors.append(traj_err[1])
            self.qacc_errors.append(traj_err[2])

            mj_step(self.m, self.d)  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Evolve the simulation

        # Post process data =======================================================
        # Cast data into ndarrays for concise conslicing
        tgt_trajectory = np.array(self.tgt_trajectory)
        trajectory = np.array(self.act_trajectory)
        frame_iter = np.arange(self.frame_count)
        fts_sen = np.array(self.fts_sen)
        regressors = np.array(self.regressors)

        # Perturb wrench ==========================================================
        error_rate = 0.05
        seed = 0
        rng = np.random.default_rng(seed)

        perturb_wrench = True  # False
        if perturb_wrench:
            fs_std = error_rate * nla.norm(fts_sen[..., :3], axis=1).max()
            ts_std = error_rate * nla.norm(fts_sen[..., 3:], axis=1).max()
            fts_sen[..., :3] += fs_std * rng.standard_normal((self.frame_count, 3))
            fts_sen[..., 3:] += ts_std * rng.standard_normal((self.frame_count, 3))

        # Compose frames =========================================================
        frames = []
        data_containers = [self.file_paths, self.transform_matrices, self.poses_sen_obj, self.twists_sen, self.dtwists_sen, fts_sen]
        for fpath, tf, pose, t, dt, ft in zip(*data_containers, strict=False):
            frame = {
                "file_path": fpath,
                "transform_matrix": tf,
                "pose_sen_obj": pose,
                "twist_sen": t,
                "dtwist_sen": dt,
                "ft_sen": ft.tolist(),
            }

            frames.append(frame)

        # Visualize data ==========================================================
        # Object linear acceleration and ft sensor measurements rel. to {sensor}
        # Actual and target joint positions
        qpos_fig, qpos_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
        qpos_fig.suptitle("qpos")
        qpos_axes[1].set(xlabel="time [s]")
        yls = ["q0-2 [m]", "q3-5 [rad]"]
        for i in range(len(qpos_axes)):
            slcr = slice(i * 3, (i + 1) * 3)
            vis.ax_plot_lines_w_tgt(qpos_axes[i], self.time, trajectory[:, 0, slcr], tgt_trajectory[:, 0, slcr], yls[i])

        # Object linear acceleration and ft sensor measurements rel. to {sensor}
        acc_ft_fig, acc_ft_axes = plt.subplots(3, 1, tight_layout=True)
        vis.ax_plot_lines(acc_ft_axes[0], frame_iter, self.linaccs_sen_obji, "recovered_linacc_sen_obji [m/s/s]")
        vis.ax_plot_lines(acc_ft_axes[1], frame_iter, fts_sen[:, :3], "frc_sen [N]")
        vis.ax_plot_lines(acc_ft_axes[2], frame_iter, fts_sen[:, 3:], "trq_sen [N*m]")
        for ax in acc_ft_axes:
            ax.hlines(0.0, frame_iter[0], frame_iter[-1], ls="dashed", alpha=0.5)

        plt.show()

        return {"frames": frames, "regressors":regressors, "fts_sen": fts_sen}
