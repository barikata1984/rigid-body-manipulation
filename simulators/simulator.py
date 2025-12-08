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

from base_config import instantiate
from controllers import LinearQuadraticRegulatorConfig
from dynamics.dynamics import (
    _calculate_frame_dynamics,
    _setup_robot_dynamics_parameters,
    get_linacc,
    get_regressor_matrix,
)
from planners import JointPositionPlannerConfig
from recorders import BasicRecorderConfig, StandardRecorderConfig
from sensors import Sensors
from transformations import Poses
from visualization.visualization import cb_rgb  # color palette for consistency
from visualization_ import ax_plot_lines, ax_plot_lines_w_tgt

from .base_simulator import BaseSimulatorConfig

# Remove redundant space at the head and tail of the horizontal axis's scale
mpl.rcParams["axes.xmargin"] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=5, suppress=True)


@dataclass
class SimulationConfig:
    """Deprecated. Will be deleted soon."""

    manipulator_name: str = "sequential"
    target_name: str = MISSING
    reset_keyframe: str = MISSING
    recorder: BasicRecorderConfig = field(default_factory=BasicRecorderConfig)
    planner: JointPositionPlannerConfig = field(default_factory=JointPositionPlannerConfig)
    controller: LinearQuadraticRegulatorConfig = field(default_factory=LinearQuadraticRegulatorConfig)
    config: str = "configurations/base.yaml"
    config_export_path: str = MISSING


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
    def __init__(
        self,
        cfg: SimulatorConfig,
        m: MjModel,
        d: MjData,
    ):
        self.m = m
        self.d = d

        if cfg.target_trajectory is not None:
            import json

            try:
                with open(cfg.target_trajectory) as f:
                    self.trajectory_data = json.load(f)
                self.duration = self.trajectory_data["duration"]
                self.fps = self.trajectory_data["fps"]
                self.target_jointvars = self.trajectory_data["jointvars"]

            except FileNotFoundError as e:
                print(f"{e}: Target rajectory file not found at {cfg.target_trajectory}")
                return

            except json.JSONDecodeError as e:
                print(
                    f"{e}: Could not decode JSON from {cfg.target_trajectory}. Please ensure it's a valid JSON file."
                )
                return

        if cfg.target_trajectory is not None:
            # Offset the joint pos according to the initial target trajcttory
            self.d.qpos = self.target_jointvars[0]["qpos"]
        else:
            # Fall back to provided duration/fps if no trajectory file
            if cfg.duration is None or cfg.fps is None:
                raise ValueError("Either target_trajectory or (duration & fps) must be provided in SimulatorConfig")
            self.duration = cfg.duration
            self.fps = cfg.fps
            self.target_jointvars = []  # placeholder; controller logic may expect list

        self.finite_differentiation_dt = cfg.finite_differentiation_dt
        self.timestep = MjOption().timestep
        self.n_steps = int(self.duration / self.timestep)
        self.recorder = instantiate(cfg.recorder, m, d, fps=self.fps)
        self.controller = instantiate(cfg.controller, m, d)

        # Instantiate register classes
        self.poses = Poses(self.m, self.d)
        self.sensors = Sensors(self.m, self.d, self.fps)

        (
            self.poses,
            self.id_ll,
            self.pose_ll_llj,
            self.uscrews_lj,
            self.simats_lj_l,
            self.hposes_lj_kj,
            self.inverse,
        ) = _setup_robot_dynamics_parameters(self.m, self.d)

        # Store static poses (pose_obj_obji is still needed here)
        self.pose_obj_obji = self.poses.get_b_biof("target/object")

        # Accessors to the registers that store the data of poses in interest.
        # NOTE: Maybe MuJoco overrives the data on registers once step() is called.
        # So, computing these poses at every step is not necessary.
        self.pose_x_ll = self.poses.x_b[self.id_ll]
        self.pose_x_sen = self.poses.get_x_("site", "target/ft_sensor")
        self.pose_x_obj = self.poses.get_x_("body", "target/object")
        self.pose_x_obji = self.pose_x_obj.dot(self.pose_obj_obji)
        self.pose_obj_cam = self.pose_x_obj.inv().dot(self.poses.x_cam[self.recorder.cam_id])
        self.pose_sen_obj = self.pose_x_sen.inv().dot(self.pose_x_obj)
        self.pose_sen_obji = self.pose_x_sen.inv().dot(self.pose_x_obji)
        self.pose_sen_llj = self.pose_x_sen.inv().dot(self.pose_x_ll.dot(self.pose_ll_llj))  # type: ignore

        # Prepare data containers
        self.file_paths = []
        self.frames = []
        self.fts_sen = []
        self.linaccs_sen_obji = []
        self.n_processed_frames = 0
        self.poses_sen_obj, self.poses_sen_obji = [], []
        self.regressors = []
        self.qpos_error = np.empty(self.m.nu)
        self.time = []
        self.trajectory, self.tgt_trajectory = [], []
        self.transform_matrices = []
        self.twists_sen, self.dtwists_sen = [], []
        self.feedforward_ctrls, self.feedback_ctrls, self.total_ctrls = [], [], []
        self.qpos_errors = []  # Added for position errors
        self.qvel_errors = []  # Added for velocity errors
        self.qacc_errors = []  # Added for acceleration errors

    def run(self):
        for _ in tqdm(range(self.n_steps), desc="Simulation Progress"):
            current_frame_idx = int(self.d.time * self.fps)
            if self.n_processed_frames <= current_frame_idx:
                self.procoess_frame(current_frame_idx)

            # step the simulate
            mj_step(self.m, self.d)

        self._post_process_data()
        self._visualize_results()

        return {"frames": self.frames, "regressors": self.regressors, "fts_sen": self.fts_sen}

    def procoess_frame(self, current_frame_idx):
        act_qpos, act_qvel, act_qacc = self.sensors.get("jointvars", perturbed=True)  # shape: (6,), (6,), (6,)
        act_traj = np.stack((act_qpos, act_qvel, act_qacc))
        self.trajectory.append(act_traj)  # type: ignore

        _, tgt_qpos, tgt_qvel, tgt_qacc, _ = self.target_jointvars[current_frame_idx].values()
        tgt_traj = np.stack((tgt_qpos, tgt_qvel, tgt_qacc))
        self.tgt_trajectory.append(tgt_traj)  # type: ignore

        # Get (d)twist_sen, and linacc_sen_obj for verification
        twist_sen, dtwist_sen, regressor = _calculate_frame_dynamics(
            act_traj, self.inverse, self.id_ll, self.pose_x_ll, self.pose_ll_llj, self.pose_x_sen
        )

        # Compute the residuals and control signals, and set the control singals
        mj_differentiatePos(
            self.m, self.qpos_error, self.finite_differentiation_dt, tgt_traj[0], act_qpos
        )  # tgt should comes first, not sure why this order but refer to LQR.ipynb of MuJoCo
        qvel_error = act_qvel - tgt_traj[1]  # act should comes first
        qacc_error = act_qacc - tgt_traj[2]
        state_error = np.concatenate((self.qpos_error, qvel_error))
        feedforward_ctrl, _, _, _ = self.inverse(tgt_traj)
        feedback_ctrl = self.controller.gain_matrix @ state_error
        ctrl = feedforward_ctrl - feedback_ctrl
        self.d.ctrl = ctrl

        self.feedforward_ctrls.append(feedforward_ctrl)
        self.feedback_ctrls.append(feedback_ctrl)
        self.total_ctrls.append(ctrl)

        self.qpos_errors.append(np.copy(self.qpos_error))  # Added for position errors
        self.qvel_errors.append(np.copy(qvel_error))  # Added for velocity errors
        self.qacc_errors.append(np.copy(qacc_error))  # Added for acceleration errors

        # Get and log the regressor matrix for Least Squares-based identification of the target object's iparams
        regressor = get_regressor_matrix(twist_sen, dtwist_sen)
        self.regressors.append(regressor)  # type: ignore

        # Get and log the linear acceleration at the target object's inertial center w.r.t. the sensor frame
        linacc_sen_obji = get_linacc(twist_sen, dtwist_sen, self.pose_sen_obji)
        self.linaccs_sen_obji.append(linacc_sen_obji)

        # Measure and log the force and torque measurements
        ft = self.sensors.get("ft", perturbed=True)
        self.fts_sen.append(ft)  # type: ignore

        # Render the camera observatio
        file_name = f"{self.n_processed_frames:04}.png"
        self.recorder.render(self.d, file_name)

        # Log the other Neural Mass Distribution ingredients
        self.file_paths.append(str(self.recorder.complete_image_dir / file_name))
        self.transform_matrices.append(self.pose_obj_cam.as_matrix().tolist())  # type: ignore
        self.poses_sen_obj.append(self.pose_sen_obj.as_matrix().tolist())  # type: ignore
        self.poses_sen_obji.append(self.pose_sen_obji.as_matrix().tolist())  # type: ignore
        self.twists_sen.append(twist_sen.tolist())
        self.dtwists_sen.append(dtwist_sen.tolist())

        self.n_processed_frames += 1
        self.time.append(self.d.time)

    def _post_process_data(self):
        self.tgt_trajectory = np.array(self.tgt_trajectory)
        self.trajectory = np.array(self.trajectory)
        self.fts_sen = np.array(self.fts_sen)
        self.regressors = np.array(self.regressors)
        self.feedforward_ctrls = np.array(self.feedforward_ctrls)
        self.feedback_ctrls = np.array(self.feedback_ctrls)
        self.total_ctrls = np.array(self.total_ctrls)
        self.qpos_errors = np.array(self.qpos_errors)  # Added for position errors
        self.qvel_errors = np.array(self.qvel_errors)  # Added for velocity errors
        self.qacc_errors = np.array(self.qacc_errors)  # Added for acceleration errors

        data_containers = [
            self.file_paths,
            self.transform_matrices,
            self.poses_sen_obj,
            self.twists_sen,
            self.dtwists_sen,
            self.fts_sen,
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
            self.frames.append(frame)

    def _visualize_results(self):
        frame_iter = np.arange(self.n_processed_frames)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Actual and target joint positions
        n_row = 2
        n_col = 1
        qpos_fig, qpos_axes = plt.subplots(n_row, n_col, sharex="col", tight_layout=True)
        qpos_fig.suptitle("qpos")
        qpos_axes[1].set(xlabel="time [s]")
        yls = ["q0-2 [m]", "q3-5 [rad]"]

        for i in range(n_row):
            slcr = slice(i * 3, (i + 1) * 3)  # (0:3), (3:6)
            ax_plot_lines_w_tgt(
                qpos_axes[i],
                self.time,
                self.trajectory[:, 0, slcr],  # type: ignore
                self.tgt_trajectory[:, 0, slcr],  # type: ignore
                yls[i],  # type: ignore
            )
            # Add legend mapping joint index to color (solid=actual, dashed=target)
            joint_indices = range(slcr.start, slcr.stop)
            legend_handles: list[Line2D] = []
            for local_j, j in enumerate(joint_indices):
                color = cb_rgb[local_j % len(cb_rgb)]
                legend_handles.append(Line2D([0], [0], color=color, linestyle="-", label=f"q{j} act"))
                legend_handles.append(Line2D([0], [0], color=color, linestyle="--", label=f"q{j} tgt"))
            qpos_axes[i].legend(handles=legend_handles, ncol=3, fontsize="x-small")

        # Feedforward and feedback control signals
        ctrl_fig, ctrl_axes = plt.subplots(self.m.nu, 1, sharex=True, tight_layout=True, figsize=(8, 12))
        ctrl_fig.suptitle("Control Signals (Feedforward vs. Feedback)")
        for j in range(self.m.nu):
            ax = ctrl_axes[j]
            ax.plot(self.time, self.feedforward_ctrls[:, j], color=cb_rgb[0], linestyle="-", label="Feedforward")
            ax.plot(self.time, self.feedback_ctrls[:, j], color=cb_rgb[1], linestyle="--", label="Feedback")
            ax.set_ylabel(f"q{j} [Nm]")
            ax.legend(fontsize="x-small")
            ax.grid(True, linestyle="--", alpha=0.6)
        ctrl_axes[-1].set_xlabel("time [s]")

        # Save plots and data
        try:
            # Save qpos figure
            qpos_debug_dir = Path("debug_log/dev-jointpos_limits/qpos")
            qpos_debug_dir.mkdir(parents=True, exist_ok=True)
            qpos_fname = qpos_debug_dir / f"qpos_{timestamp}.png"
            qpos_fig.savefig(qpos_fname)
            print(f"Saved qpos figure: {qpos_fname}")

            # Save ctrl figure
            ctrl_debug_dir = Path("debug_log/dev-jointpos_limits/ctrl")
            ctrl_debug_dir.mkdir(parents=True, exist_ok=True)
            ctrl_fname = ctrl_debug_dir / f"ctrl_{timestamp}.png"
            ctrl_fig.savefig(ctrl_fname)
            print(f"Saved ctrl figure: {ctrl_fname}")

            # Save raw data to a CSV file
            data_debug_dir = Path("debug_log/dev-jointpos_limits/data")
            data_debug_dir.mkdir(parents=True, exist_ok=True)

            # 1. Create header
            header_parts = []
            for prefix in [
                "tgt_qpos",
                "tgt_qvel",
                "tgt_qacc",
                "act_qpos",
                "act_qvel",
                "act_qacc",
                "qpos_error",  # Added for position errorsu
                "qvel_error",  # Added for velocity errors
                "qacc_error",  # Added for acceleration errors
                "ff_ctrl",
                "fb_ctrl",
                "total_ctrl",
            ]:
                for i in range(self.m.nu):
                    header_parts.append(f"{prefix}_{i}")
            header = "time," + ",".join(header_parts)

            # 2. Reshape and combine data arrays
            time_col = np.array(self.time).reshape(-1, 1)
            tgt_traj_flat = self.tgt_trajectory.reshape(self.n_processed_frames, -1)
            act_traj_flat = self.trajectory.reshape(self.n_processed_frames, -1)

            data_to_save = np.hstack(
                [
                    time_col,
                    tgt_traj_flat,
                    act_traj_flat,
                    self.qpos_errors,  # Added for position errors
                    self.qvel_errors,  # Added for velocity errors
                    self.qacc_errors,  # Added for acceleration errors
                    self.feedforward_ctrls,
                    self.feedback_ctrls,
                    self.total_ctrls,
                ]
            )

            # 3. Save to CSV
            data_fname = data_debug_dir / f"data_{timestamp}.csv"
            np.savetxt(data_fname, data_to_save, delimiter=",", header=header, comments="")
            print(f"Saved data CSV: {data_fname}")

        except Exception as e:
            print(f"Failed to save plots or data: {e}")

        # Object linear acceleration and ft sensor measurements
        acc_ft_fig, acc_ft_axes = plt.subplots(3, 1, tight_layout=True)
        ax_plot_lines(acc_ft_axes[0], frame_iter, np.array(self.linaccs_sen_obji), "recovered_linacc_sen_obji [m/s/s]")
        ax_plot_lines(acc_ft_axes[1], frame_iter, self.fts_sen[:, :3], "frc_sen [N]")  # type: ignore
        ax_plot_lines(acc_ft_axes[2], frame_iter, self.fts_sen[:, 3:], "trq_sen [N*m]")  # type: ignore
        for ax in acc_ft_axes:
            ax.hlines(0.0, frame_iter[0], frame_iter[-1], ls="dashed", alpha=0.5)

        plt.show()
