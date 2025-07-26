from dataclasses import dataclass, field
from functools import partial

import matplotlib as mpl
import numpy as np
from liegroups import SE3
from matplotlib import pyplot as plt
from mujoco._functions import mj_differentiatePos, mj_step
from mujoco._structs import MjData, MjModel, MjOption
from omegaconf import MISSING
from tqdm import tqdm

import dynamics as dyn
from configurations import instantiate
from controllers import LinearQuadraticRegulatorConfig
from recorders import StandardRecorderConfig
from sensors import Sensors
from transformations import Poses
from utilities import get_element_id
from visualization import ax_plot_lines, ax_plot_lines_w_tgt

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
    reset_keyframe: str = "initial_state"
    duration: float = MISSING
    fps: int = MISSING
    recorder: StandardRecorderConfig = field(default_factory=StandardRecorderConfig)
    # planner: JointPositionPlannerConfig = field(default_factory=JointPositionPlannerConfig)
    controller: LinearQuadraticRegulatorConfig = field(default_factory=LinearQuadraticRegulatorConfig)
    exp_setup: str = "experiment_setups/simulators/base.yaml"
    config_export_path: str | None = None
    displacements: list[float] = MISSING
    target_trajectory: str | None = None
    generate_trajectory: str | None = None
    pos_offset: list[float] | None = None


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

        # self.fps = cfg.fps
        self.n_steps = int(cfg.duration / MjOption().timestep)
        self.recorder = instantiate(cfg.recorder, m, d, fps=self.fps)
        # self.planner = instantiate(cfg.planner, m, d, duration=cfg.duration, fps=self.fps)
        self.controller = instantiate(cfg.controller, m, d)

        #        pos_offset = d.qpos.copy().tolist() if cfg.planner.pos_offset is None else cfg.planner.pos_offset
        #        self.trajectories = get_trajectory_interpolated_with_fifth_order_spline(
        #            cfg.duration,
        #            self.fps,
        #            pos_offset,  # [m, m, m, rad, rad, rad]
        #            cfg.displacements,  # [m, m, m, rad, rad, rad]
        #        )

        # Instantiate register classes
        self.poses = Poses(self.m, self.d)
        self.sensors = Sensors(self.m, self.d, cfg.fps)

        # Get ids and indices
        self.id_ll = get_element_id(self.m, "body", "link6")
        self.id_x2ll = slice(0, self.id_ll + 1)

        # Store static poses
        self.pose_obj_obji = self.poses.get_b_biof("target/object")
        self.pose_ll_llj = self.poses.l_lj[self.id_ll]

        # Get unit screws wr2 link joints
        uscrews_lj = []
        for t, ax in zip(self.m.jnt_type, self.m.jnt_axis, strict=False):
            us_lj = np.zeros(6)
            if 2 == t:  # slider joint
                us_lj[:3] += ax
            elif 3 == t:  # hinge joint
                us_lj[3:] += ax
            else:
                raise TypeError(
                    "Only slide or hinge joints, represented as 2 or 3 for an element of m.jnt_type, are supported."
                )
            uscrews_lj.append(us_lj)
        self.uscrews_lj = np.array(uscrews_lj)

        # Transfer spatial inertia matrices and join them
        simats_bi_b = dyn.get_spatial_inertia_matrix(
            self.m.body_mass,
            self.m.body_inertia,
        )
        simats_lj_l = []
        for pose_lj_li, simat_li_l in zip(self.poses.lj_li, simats_bi_b[self.id_x2ll], strict=False):
            simats_lj_l.append(dyn.transfer_simat(pose_lj_li, simat_li_l))
        simats_lj_l = np.array(simats_lj_l)

        pose_x_ll = self.poses.x_b[self.id_ll]  # Use initial pose for setup
        for pose_x_bi, simat_bi_b in zip(
            self.poses.x_bi[self.id_ll + 1 :], simats_bi_b[self.id_ll + 1 :], strict=False
        ):
            pose_x_llj = pose_x_ll.dot(self.pose_ll_llj)  # type: ignore
            pose_bi_llj = pose_x_bi.inv().dot(pose_x_llj)
            simat_llj_b = dyn.transfer_simat(pose_bi_llj.inv(), simat_bi_b)  # type: ignore
            simats_lj_l[self.id_ll] += simat_llj_b
        self.simats_lj_l = simats_lj_l

        # Get link joints' home poses
        hposes_lj_kj = [SE3.identity()]
        for k in range(self.m.njnt):
            hpose_kj_k = self.poses.l_lj[k].inv()
            hpose_l_lj = self.poses.l_lj[k + 1]
            hpose_k_l = self.poses.a_b[k + 1]
            hpose_kj_lj = hpose_kj_k.dot(hpose_k_l.dot(hpose_l_lj))
            hposes_lj_kj.append(hpose_kj_lj.inv())  # type: ignore
        self.hposes_lj_kj = hposes_lj_kj

        # Partially initialize inverse dynamics function
        gacc_x = -1 * np.array([*MjOption().gravity, 0, 0, 0])
        self.inverse = partial(
            dyn.inverse,
            hposes_body_parent=self.hposes_lj_kj,
            simats_body=self.simats_lj_l,
            uscrews_body=self.uscrews_lj,
            twist_0=np.zeros(6),
            dtwist_0=gacc_x,
        )

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
        self.dtwists_sen = []
        self.file_paths = []
        self.frames = []
        self.fts_sen = []
        self.linaccs_sen_obji = []
        self.n_processed_frames = 0
        self.poses_sen_obj = []
        self.poses_sen_obji = []
        self.regressors = []
        self.res_qpos = np.empty(self.m.nu)
        self.tgt_trajectory = []
        self.time = []
        self.trajectory = []
        self.transform_matrices = []
        self.twists_sen = []

    def run(self):
        for _ in tqdm(range(self.n_steps), desc="Simulation Progress"):
            current_frame_idx = int(self.d.time * self.fps)
            if self.n_processed_frames <= current_frame_idx:
                self.procoess_frame(current_frame_idx)

            # step the simulate
            mj_step(self.m, self.d)

        self._post_process_data()
        # self._visualize_results()

        return {"frames": self.frames, "regressors": self.regressors, "fts_sen": self.fts_sen}

    def procoess_frame(self, current_frame_idx):
        qpos, qvel, qacc = self.sensors.get("jointvars", perturbed=True)  # # shape: (6,), (6,), (6,)
        act_traj = np.stack((qpos, qvel, qacc))
        self.trajectory.append(act_traj)  # type: ignore

        tgt_qpos, tgt_qvel, tgt_qacc = self.target_jointvars[current_frame_idx].values()
        tgt_traj = np.stack((tgt_qpos, tgt_qvel, tgt_qacc))
        self.tgt_trajectory.append(tgt_traj)  # type: ignore

        # Get (d)twist_sen, and linacc_sen_obj for verification
        _, _, twists_lj_l, dtwists_lj_l = self.inverse(act_traj)
        twist_llj = twists_lj_l[self.id_ll]
        twist_sen = self.pose_sen_llj.adjoint() @ twist_llj  # type: ignore
        dtwist_llj = dtwists_lj_l[self.id_ll]
        pose_sen_llj_dadjoint = SE3.curlywedge(twist_sen) @ self.pose_sen_llj.adjoint()  # type: ignore
        dtwist_sen = pose_sen_llj_dadjoint @ twist_llj + self.pose_sen_llj.adjoint() @ dtwist_llj  # type: ignore

        # Compute the residuals and control signals, and set the control singals
        mj_differentiatePos(self.m, self.res_qpos, self.m.nu, qpos, tgt_traj[0])
        res_state = np.concatenate((self.res_qpos, tgt_traj[1] - qvel))
        tgt_ctrl, _, _, _ = self.inverse(tgt_traj)
        self.d.ctrl = tgt_ctrl - self.controller.gain_matrix @ res_state

        # Get and log the regressor matrix for Least Squares-based identification of the target object's iparams
        regressor = dyn.get_regressor_matrix(twist_sen, dtwist_sen)
        self.regressors.append(regressor)  # type: ignore

        # Get and log the linear acceleration at the target object's inertial center w.r.t. the sensor frame
        linacc_sen_obji = dyn.get_linacc(twist_sen, dtwist_sen, self.pose_sen_obji)
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

        # Actual and target joint positions
        qpos_fig, qpos_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
        qpos_fig.suptitle("qpos")
        qpos_axes[1].set(xlabel="time [s]")
        yls = ["q0-2 [m]", "q3-5 [rad]"]
        for i in range(len(qpos_axes)):
            slcr = slice(i * 3, (i + 1) * 3)
            ax_plot_lines_w_tgt(
                qpos_axes[i],
                self.time,
                self.trajectory[:, 0, slcr],  # type: ignore
                self.tgt_trajectory[:, 0, slcr],  # type: ignore
                yls[i],  # type: ignore
            )

        # Object linear acceleration and ft sensor measurements
        acc_ft_fig, acc_ft_axes = plt.subplots(3, 1, tight_layout=True)
        ax_plot_lines(acc_ft_axes[0], frame_iter, np.array(self.linaccs_sen_obji), "recovered_linacc_sen_obji [m/s/s]")
        ax_plot_lines(acc_ft_axes[1], frame_iter, self.fts_sen[:, :3], "frc_sen [N]")  # type: ignore
        ax_plot_lines(acc_ft_axes[2], frame_iter, self.fts_sen[:, 3:], "trq_sen [N*m]")  # type: ignore
        for ax in acc_ft_axes:
            ax.hlines(0.0, frame_iter[0], frame_iter[-1], ls="dashed", alpha=0.5)

        plt.show()
