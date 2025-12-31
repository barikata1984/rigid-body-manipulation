from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mujoco._functions import mj_differentiatePos, mj_step
from mujoco._structs import MjData, MjModel, MjOption
from omegaconf import MISSING
from tqdm import tqdm

from factory import instantiate
from controllers import LinearQuadraticRegulatorConfig
from dynamics import dynamics as dyn
from dynamics.dynamics import (
    calculate_frame_dynamics,
    get_linacc,
    get_regressor_matrix,
    setup_robot_dynamics_parameters,
)
from planners import JointPositionPlannerConfig
from recorders import BasicRecorderConfig, StandardRecorderConfig
from sensors import Sensors
from transformations import Poses
from visualization import visualization as vis
from visualization.visualization import cb_rgb  # color palette for consistency
from visualization_ import ax_plot_lines, ax_plot_lines_w_tgt

from .base_simulator import BaseSimulatorConfig

# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams["axes.xmargin"] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=5, suppress=True)


@dataclass
class SimulatorConfig(BaseSimulatorConfig):
    target_class: str = "Simulator"  # type: ignore
    manipulator: str = "xml_models/manipulators/sequential"
    object: str = MISSING
    reset_keyframe: str = MISSING  # | None = None
    # Some tests pass these explicitly; keep them optional for trajectory-driven runs
    duration: float | None = None
    fps: int | None = None
    recorder: StandardRecorderConfig = field(default_factory=StandardRecorderConfig)
    planner: JointPositionPlannerConfig = field(default_factory=JointPositionPlannerConfig)
    controller: LinearQuadraticRegulatorConfig = field(default_factory=LinearQuadraticRegulatorConfig)
    exp_setup: str = "configurations/simulations/base.yaml"
    config_export_path: str | None = None
    displacements: list[float] = MISSING
    target_trajectory: str | None = None
    generate_trajectory: str | None = None
    finite_differentiation_dt: float = 1.0


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


class Simulator:
    m: MjModel
    d: MjData

    #def __init__(self, m: MjModel, d: MjData, recorder, planner, controller):
    def __init__(self,
                 cfg: SimulatorConfig,
                 m: MjModel,
                 d: MjData,
                 ) -> None:

        self.recorder = instantiate(cfg.recorder, m, d)
        self.planner = instantiate(cfg.planner, m, d)
        self.controller = instantiate(cfg.controller, m, d)

        
        # Set a random number generator ===========================================
        rng = np.random.default_rng()
        rng.standard_normal(10)

        self.m = m
        self.d = d

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
        self.qpos_err = np.empty(m.nu)

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
            act_traj = np.stack(self.sensors.get("jointvars", perturbed=True))  # type: ignore
            _, _, twists_lj_l, dtwists_lj_l = self.inverse(act_traj)

            # Compute actuator controls and evolute the simulation
            tgt_traj = self.planner.plan(step)

            if self.frame_count <= self.d.time * self.recorder.fps:
                self.process_frame(tgt_traj, act_traj)

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
            traj_err = act_traj - tgt_traj
            state_err = np.concatenate((self.qpos_err, traj_err[1]))
            feedback_ctrl = self.controller.gain_matrix @ state_err
            self.d.ctrl = feedforward_ctrl - feedback_ctrl

            self.qpos_errors.append(self.qpos_err)
            self.qvel_errors.append(traj_err[1])
            self.qacc_errors.append(traj_err[2])

            # >>> Evolve the simulation >>>
            mj_step(self.m, self.d)
            # <<< Evolve the simulation <<<

        # Post process data =======================================================
        # Cast data into ndarrays for concise conslicing
        tgt_trajectory = np.array(self.tgt_trajectory)
        trajectory = np.array(self.act_trajectory)
        frame_iter = np.arange(self.frame_count)
        fts_sen = np.array(self.fts_sen)
        regressors = np.array(self.regressors)

        # Compose frames =========================================================
        frames = []
        data_containers = [
            self.file_paths,
            self.transform_matrices,
            self.poses_sen_obj,
            self.twists_sen,
            self.dtwists_sen,
            fts_sen,
        ]
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
            vis.ax_plot_lines_w_tgt(
                qpos_axes[i], self.time, trajectory[:, 0, slcr], tgt_trajectory[:, 0, slcr], yls[i]
            )

        # Object linear acceleration and ft sensor measurements rel. to {sensor}
        acc_ft_fig, acc_ft_axes = plt.subplots(3, 1, tight_layout=True)
        vis.ax_plot_lines(acc_ft_axes[0], frame_iter, self.linaccs_sen_obji, "recovered_linacc_sen_obji [m/s/s]")
        vis.ax_plot_lines(acc_ft_axes[1], frame_iter, fts_sen[:, :3], "frc_sen [N]")
        vis.ax_plot_lines(acc_ft_axes[2], frame_iter, fts_sen[:, 3:], "trq_sen [N*m]")
        for ax in acc_ft_axes:
            ax.hlines(0.0, frame_iter[0], frame_iter[-1], ls="dashed", alpha=0.5)

        plt.show()

        return {"frames": frames, "regressors": regressors, "fts_sen": fts_sen}

    def process_frame(self, tgt_traj, act_traj):
        self.time.append(self.d.time)
        self.tgt_trajectory.append(tgt_traj)
        self.act_trajectory.append(act_traj)

        twist_sen, dtwist_sen, regressor = calculate_frame_dynamics(
            act_traj, self.inverse, self.id_ll, self.pose_x_ll, self.pose_ll_llj, self.pose_x_sen
        )
        self.twists_sen.append(twist_sen.tolist())
        self.dtwists_sen.append(dtwist_sen.tolist())

        linacc_sen_obji = dyn.get_linacc(twist_sen, dtwist_sen, self.pose_sen_obji)
        self.linaccs_sen_obji.append(linacc_sen_obji)

        wrench = self.sensors.get("wrench")
        self.fts_sen.append(wrench)

        regressor = dyn.get_regressor_matrix(twist_sen, dtwist_sen)
        self.regressors.append(regressor)

        # Writing a single frame of a dataset =============================
        file_name = f"{self.frame_count:04}.png"
        self.recorder.render(self.d, file_name)  # recorder.cam_id is selected internally

        # Log NeMD ingredients ============================================
        # Items which need to be computed at every frame recoding
        pose_obj_cam = self.pose_x_obj.inv().dot(self.poses.x_cam[self.recorder.cam_id])
        self.transform_matrices.append(pose_obj_cam.as_matrix().tolist())

        self.poses_sen_obj.append(self.pose_sen_obj.as_matrix().tolist())
        self.poses_sen_obji.append(self.pose_sen_obji.as_matrix().tolist())
        self.file_paths.append(str(self.recorder.complete_image_dir / file_name))

        self.recorder.recursive_eval_data["frames"][f"{self.frame_count:04}"] = {
            "regressor": regressor.tolist(),
            "ft_sen": wrench.tolist(),
        }

        self.frame_count += 1
