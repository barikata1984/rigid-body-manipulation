from dataclasses import dataclass, field
from functools import partial

import matplotlib as mpl
import numpy as np
from liegroups import SE3
from matplotlib import pyplot as plt
from mujoco._functions import mj_differentiatePos, mj_step
from mujoco._structs import MjData, MjModel, MjOption
from numpy import linalg as nla
from tqdm import tqdm
from tyro import MISSING

import dynamics as dyn
from configurations import instantiate
from controllers import LinearQuadraticRegulatorConfig
from planners import JointPositionPlannerConfig
from recorders import StandardRecorderConfig
from sensors import Sensors
from transformations import Poses
from utilities import get_element_id
from visualization import ax_plot_lines, ax_plot_lines_w_tgt

from .base_simulator import BaseSimulatorConfig


@dataclass
class SimulatorConfig(BaseSimulatorConfig):
    target_class: str = "Simulator"  # type: ignore
    manipulator_dir: str = "xml_models/manipulators/sequential"
    target_dir: str = MISSING
    reset_keyframe: str = "initial_state"
    recorder: StandardRecorderConfig = field(default_factory=StandardRecorderConfig)
    planner: JointPositionPlannerConfig = field(default_factory=JointPositionPlannerConfig)
    controller: LinearQuadraticRegulatorConfig = field(default_factory=LinearQuadraticRegulatorConfig)
    exp_setup: str = "experimental_setups/base.yaml"
    config_export_path: str | None = None


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
mpl.rcParams["axes.xmargin"] = 0
# Reduce the number of digits of values with numpy
np.set_printoptions(precision=5, suppress=True)


class Simulator:
    def __init__(
        self,
        cfg: SimulatorConfig,
        m: MjModel,
        d: MjData,
    ):
        self.m = m
        self.d = d

        self.recorder = instantiate(cfg.recorder, m, d)
        self.planner = instantiate(cfg.planner, m, d)
        self.controller = instantiate(cfg.controller, m, d)

        # Instantiate register classes
        self.poses = Poses(self.m, self.d)
        self.sensors = Sensors(self.m, self.d)

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

        # Set a random number generator
        self.rng = np.random.default_rng()
        self.rng.standard_normal(10)

        # Prepare data containers
        self.res_qpos = np.empty(self.m.nu)
        self.tgt_trajectory = []
        self.trajectory = []
        self.fts_sen = []
        self.time = []
        self.frame_count = 0
        self.regressors = []
        self.frames = []
        self.file_paths = []
        self.transform_matrices = []
        self.poses_sen_obj = []
        self.poses_sen_obji = []
        self.twists_sen = []
        self.dtwists_sen = []
        self.linaccs_sen_obji = []

    def run(self):
        for step_idx in tqdm(range(self.planner.n_steps), desc="Progress"):
            self.step(step_idx)

        self._post_process_data()
        self._visualize_results()

        return {"frames": self.frames, "regressors": self.regressors}

    def step(self, step_idx):
        tgt_traj = self.planner.plan(step_idx)
        tgt_ctrl, _, _, _ = self.inverse(tgt_traj)

        qpos, qvel, qacc = self.d.qpos, self.d.qvel, self.d.qacc
        act_traj = np.stack((qpos, qvel, qacc))
        _, _, twists_lj_l, dtwists_lj_l = self.inverse(act_traj)

        if self.frame_count <= self.d.time * self.recorder.fps:
            self.time.append(self.d.time)
            self.tgt_trajectory.append(tgt_traj)
            self.trajectory.append(act_traj)

            import pdb

            pdb.set_trace()

            # Dynamic poses
            pose_x_ll = self.poses.x_b[self.id_ll]
            pose_x_sen = self.poses.get_x_("site", "target/ft_sensor")
            pose_x_obj = self.poses.get_x_("body", "target/object")
            pose_sen_obj = pose_x_sen.inv().dot(pose_x_obj)
            pose_x_obji = pose_x_obj.dot(self.pose_obj_obji)
            pose_sen_obji = pose_x_sen.inv().dot(pose_x_obji)

            # Get (d)twist_sen, and linacc_sen_obj for verification
            pose_sen_llj = pose_x_sen.inv().dot(pose_x_ll.dot(self.pose_ll_llj))  # type: ignore
            twist_llj = twists_lj_l[self.id_ll]
            twist_sen = pose_sen_llj.adjoint() @ twist_llj  # type: ignore
            dtwist_llj = dtwists_lj_l[self.id_ll]
            pose_sen_llj_dadjoint = SE3.curlywedge(twist_sen) @ pose_sen_llj.adjoint()  # type: ignore
            dtwist_sen = pose_sen_llj_dadjoint @ twist_llj + pose_sen_llj.adjoint() @ dtwist_llj  # type: ignore

            linacc_sen_obji = dyn.extract_linacc_frame_transferred(twist_sen, dtwist_sen, pose_sen_obji)
            self.linaccs_sen_obji.append(linacc_sen_obji)

            # Get force-torque measurements
            force = self.sensors.get("force")
            torque = self.sensors.get("torque")
            wrench = np.concatenate([force, torque], axis=None)
            self.fts_sen.append(wrench)

            regressor = dyn.get_regressor_matrix(twist_sen, dtwist_sen)
            self.regressors.append(regressor)

            # Writing a single frame of a dataset
            file_name = f"{self.frame_count:04}.png"
            self.recorder.render(self.d, file_name)

            # Log NeMD ingredients
            pose_obj_cam = pose_x_obj.inv().dot(self.poses.x_cam[self.recorder.cam_id])
            self.file_paths.append(str(self.recorder.complete_image_dir / file_name))
            self.transform_matrices.append(pose_obj_cam.as_matrix().tolist())  # type: ignore
            self.poses_sen_obj.append(pose_sen_obj.as_matrix().tolist())  # type: ignore
            self.poses_sen_obji.append(pose_sen_obji.as_matrix().tolist())  # type: ignore
            self.twists_sen.append(twist_sen.tolist())
            self.dtwists_sen.append(dtwist_sen.tolist())

            self.frame_count += 1

        mj_differentiatePos(self.m, self.res_qpos, self.m.nu, qpos, tgt_traj[0])
        res_state = np.concatenate((self.res_qpos, tgt_traj[1] - qvel))
        self.d.ctrl = tgt_ctrl - self.controller.gain_matrix @ res_state

        mj_step(self.m, self.d)

    def _post_process_data(self):
        self.tgt_trajectory = np.array(self.tgt_trajectory)
        self.trajectory = np.array(self.trajectory)
        self.fts_sen = np.array(self.fts_sen)
        self.regressors = np.array(self.regressors)

        # Perturb wrench
        error_rate = 0.05
        seed = 0
        rng = np.random.default_rng(seed)
        perturb_wrench = True
        if perturb_wrench:
            fs_std = error_rate * nla.norm(self.fts_sen[..., :3], axis=1).max()
            ts_std = error_rate * nla.norm(self.fts_sen[..., 3:], axis=1).max()
            self.fts_sen[..., :3] += fs_std * rng.standard_normal((self.frame_count, 3))
            self.fts_sen[..., 3:] += ts_std * rng.standard_normal((self.frame_count, 3))

        # Compose frames
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
        frame_iter = np.arange(self.frame_count)

        # Actual and target joint positions
        qpos_fig, qpos_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
        qpos_fig.suptitle("qpos")
        qpos_axes[1].set(xlabel="time [s]")
        yls = ["q0-2 [m]", "q3-5 [rad]"]
        for i in range(len(qpos_axes)):
            slcr = slice(i * 3, (i + 1) * 3)
            ax_plot_lines_w_tgt(
                qpos_axes[i], self.time, self.trajectory[:, 0, slcr], self.tgt_trajectory[:, 0, slcr], yls[i]
            )

        # Object linear acceleration and ft sensor measurements
        acc_ft_fig, acc_ft_axes = plt.subplots(3, 1, tight_layout=True)
        ax_plot_lines(acc_ft_axes[0], frame_iter, np.array(self.linaccs_sen_obji), "recovered_linacc_sen_obji [m/s/s]")
        ax_plot_lines(acc_ft_axes[1], frame_iter, self.fts_sen[:, :3], "frc_sen [N]")
        ax_plot_lines(acc_ft_axes[2], frame_iter, self.fts_sen[:, 3:], "trq_sen [N*m]")
        for ax in acc_ft_axes:
            ax.hlines(0.0, frame_iter[0], frame_iter[-1], ls="dashed", alpha=0.5)

        plt.show()


def simulate(
    m: MjModel,
    d: MjData,
    recorder,
    planner,
    controller,
):
    sim = Simulator(m, d, recorder, planner, controller)
    return sim.run()
