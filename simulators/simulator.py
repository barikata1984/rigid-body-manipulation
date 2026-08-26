from dataclasses import dataclass, field

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from mujoco._functions import mj_differentiatePos, mj_forward, mj_step
from mujoco._structs import MjData, MjModel
from omegaconf import MISSING
from tqdm import tqdm

from controllers import LinearQuadraticRegulatorConfig
from dynamics.dynamics import (
    calculate_frame_dynamics,
    get_linacc,
    setup_robot_dynamics_parameters,
)
from factory import instantiate
from recorders import StandardRecorderConfig
from sensors import Sensors
from visualization.visualization import ax_plot_lines, ax_plot_lines_w_tgt

from .base_simulator import BaseSimulatorConfig


def _refresh_derived_data(model: MjModel, data: MjData) -> None:
    """Synchronize MuJoCo's derived values with its current state.

    ``mj_step`` computes acceleration-dependent sensors before integration, so its
    returned ``qpos``/``qvel`` are one timestep newer than ``qacc``/``sensordata``.
    A forward pass refreshes those derived values without advancing time.
    """
    mj_forward(model, data)


@dataclass
class SimulationData:
    time: list = field(default_factory=list)
    tgt_trajectory: list = field(default_factory=list)
    act_trajectory: list = field(default_factory=list)
    qpos_errors: list = field(default_factory=list)
    qvel_errors: list = field(default_factory=list)
    qacc_errors: list = field(default_factory=list)
    poses_sen_obj: list = field(default_factory=list)
    poses_sen_obji: list = field(default_factory=list)
    twists_sen: list = field(default_factory=list)
    dtwists_sen: list = field(default_factory=list)
    linaccs_sen_obji: list = field(default_factory=list)
    wrenches: list = field(default_factory=list)
    regressors: list = field(default_factory=list)
    file_paths: list = field(default_factory=list)
    transform_matrices: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    frame_count: int = 0


@dataclass
class SimulatorConfig(BaseSimulatorConfig):
    target_class: str = "Simulator"
    manipulator: str = "xml_models/manipulators/sequential"
    object: str = MISSING
    reset_keyframe: str = MISSING
    # Some tests pass these explicitly; keep them optional for trajectory-driven runs
    duration: float | None = None
    fps: int | None = None
    recorder: StandardRecorderConfig = field(default_factory=StandardRecorderConfig)
    controller: LinearQuadraticRegulatorConfig = field(default_factory=LinearQuadraticRegulatorConfig)
    exp_setup: str = "configurations/simulations/base.yaml"
    config_export_path: str | None = None
    target_trajectory: str | None = None
    generate_trajectory: str | None = None
    diffpos_dt: float = 1.0
    get_unperturbed: bool = True
    noise_profile: str = "empirical"
    control_noise: bool = True
    control_derived_velocity: bool = False
    record_noise: bool = True
    record_joint_noise: bool = True
    record_wrench_noise: bool = True
    joint_bias_scale: float = 0.0
    wrench_bias_scale: float = 0.0
    noise_scale: float = 1.0
    translation_noise_scale: float = 1.0
    rotation_noise_scale: float = 1.0
    perturb_wrench: bool = True
    force_noise_scale: float = 1.0
    torque_noise_scale: float = 1.0
    seed: int | None = None


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
#             x | world frame (x \u2208 b)
#             q | joint space
#
#  NOTE: 's' follows the descriptor part of a variable's name to clarify that
#        the variable contains multiple descriptors.
#
#        \u250f\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501 Body namespace: "b"ody and its p"a"rent body \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2513
#
# Bodies: x, link1 (firstlink), ..., link6 or sth (lastlink), attachment, object
#
#                                   \u2517\u2501 "l"ast"l"ink merged with the later ones \u2501\u251b
#
#        \u2517\u2501\u2501 Link namespace: "l"ink and its parent body (= "k", prior to 'l') \u2501\u2501\u251b
#


class Simulator:
    m: MjModel
    d: MjData

    def __init__(
        self,
        cfg: SimulatorConfig,
        *args,
        model: MjModel,
        data: MjData,
        target_trajectory=None,
        **kwargs,
    ) -> None:
        self.target_trajectory = target_trajectory

        m = model
        d = data

        self.recorder = instantiate(cfg.recorder, model=m, data=d)
        self.controller = instantiate(cfg.controller, model=m, data=d)

        if self.target_trajectory is None:
            raise ValueError("target_trajectory is required but was not provided to Simulator")

        self.timestep = m.opt.timestep
        self.n_steps = int(self.target_trajectory.duration / self.timestep)
        self.fps = self.target_trajectory.fps
        self.diffpos_dt = cfg.diffpos_dt

        self.m = m
        self.d = d

        self.sensors = Sensors(
            self.m,
            self.d,
            self.fps,
            noise_scale=cfg.noise_scale,
            translation_noise_scale=cfg.translation_noise_scale,
            rotation_noise_scale=cfg.rotation_noise_scale,
            force_noise_scale=cfg.force_noise_scale,
            torque_noise_scale=cfg.torque_noise_scale,
            noise_profile=cfg.noise_profile,
            joint_bias_scale=cfg.joint_bias_scale,
            wrench_bias_scale=cfg.wrench_bias_scale,
            seed=cfg.seed,
        )
        self.control_noise = cfg.control_noise
        self.control_derived_velocity = cfg.control_derived_velocity
        self.record_noise = cfg.record_noise
        self.record_joint_noise = self.record_noise and cfg.record_joint_noise
        self.record_wrench_noise = self.record_noise and cfg.record_wrench_noise
        self.perturb_wrench = cfg.perturb_wrench
        noise_metadata = self.sensors.metadata()
        noise_metadata.update(
            control_noise=self.control_noise,
            control_velocity_source="recorded_derived" if self.control_derived_velocity else "simulator",
            record_noise=self.record_noise,
            record_joint_noise=self.record_joint_noise,
            record_wrench_noise=self.record_wrench_noise and self.perturb_wrench,
        )
        self.recorder.base_transform["noise_model"] = noise_metadata
        self.recorder.base_transform["noise_seed"] = self.sensors.seed

        _params = setup_robot_dynamics_parameters(self.m, self.d)
        self.poses = _params.poses
        self.id_ll = _params.id_ll
        self.pose_ll_llj = _params.pose_ll_llj
        self.uscrews_lj = _params.uscrews_lj
        self.simats_lj_l = _params.simats_lj_l
        self.hposes_lj_kj = _params.hposes_lj_kj
        self.inverse = _params.inverse_dynamics

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

        # Calculation buffer (not a data storage)
        self.qpos_err = np.empty(m.nu)

        # Data buffers collected during simulation
        self.data = SimulationData()
        self.get_unperturbed = cfg.get_unperturbed
        self.data_unperturbed = SimulationData() if self.get_unperturbed else None

    def run(self):
        if not self.recorder.videowriter.isOpened():
            raise RuntimeError("VideoWriter failed to open. Check codec and output path.")

        for _step in tqdm(range(self.n_steps), desc="Progress"):
            should_record = self.data.frame_count <= self.d.time * self.fps
            if should_record:
                # mj_step leaves derived quantities at the pre-integration state.
                # Refresh only when they will be recorded, rather than at every
                # 2 ms physics step.
                _refresh_derived_data(self.m, self.d)

            observed_jointvars = self.sensors.sample_jointvars()
            true_jointvars = np.stack(self.sensors.get("jointvars", perturbed=False))  # type: ignore
            control_jointvars = (
                self.sensors.sample_control_jointvars(derived_velocity=self.control_derived_velocity)
                if self.control_noise
                else true_jointvars
            )
            record_jointvars = observed_jointvars if self.record_joint_noise else true_jointvars

            if should_record:
                _tgt_traj = self.target_trajectory.frames[self.data.frame_count]
                tgt_traj = np.array(_tgt_traj)
                self._store_current_data(tgt_traj, record_jointvars)
                self.data.frame_count += 1

            frame_idx = min(int(self.d.time * self.fps), len(self.target_trajectory.frames) - 1)
            tgt_traj = np.array(self.target_trajectory.frames[frame_idx])
            self._set_ctrl(tgt_traj, control_jointvars)

            previous_time = float(self.d.time)
            mj_step(self.m, self.d)
            if float(self.d.time) <= previous_time or not all(
                np.isfinite(values).all() for values in (self.d.qpos, self.d.qvel, self.d.qacc)
            ):
                raise RuntimeError(
                    f"MuJoCo simulation became unstable after t={previous_time:.6f} s; "
                    "refusing to emit a partial dataset"
                )

        # Compose frames =========================================================
        data_containers = [
            self.data.file_paths,
            self.data.transform_matrices,
            self.data.poses_sen_obj,
            self.data.twists_sen,
            self.data.dtwists_sen,
            self.data.wrenches,
            self.data.regressors,
        ]

        frames = []
        for fpath, tf, pose, t, dt, w, r in zip(*data_containers, strict=True):
            frame = {
                "file_path": fpath,
                "transform_matrix": tf,
                "pose_sen_obj": pose,
                "twist_sen": t,
                "dtwist_sen": dt,
                "wrench": w.tolist(),
                "regressor": r.tolist(),
            }

            frames.append(frame)

        # Visualize data
        self._visualize(self.data)

        result = {"frames": frames, "regressors": self.data.regressors, "wrenches": self.data.wrenches}

        if self.get_unperturbed:
            # Images, transforms, and poses are noise-independent, so reuse those of the noisy frames
            unperturbed_containers = [
                self.data.file_paths,
                self.data.transform_matrices,
                self.data.poses_sen_obj,
                self.data_unperturbed.twists_sen,
                self.data_unperturbed.dtwists_sen,
                self.data_unperturbed.wrenches,
                self.data_unperturbed.regressors,
                self.data_unperturbed.act_trajectory,
            ]

            unperturbed_frames = []
            for fpath, tf, pose, t, dt, w, r, act in zip(*unperturbed_containers, strict=True):
                unperturbed_frames.append(
                    {
                        "file_path": fpath,
                        "transform_matrix": tf,
                        "pose_sen_obj": pose,
                        "twist_sen": t,
                        "dtwist_sen": dt,
                        "wrench": w.tolist(),
                        "regressor": r.tolist(),
                        "jointvars_clean": act.tolist(),  # [qpos, qvel, qacc] for offline noise re-synthesis
                    }
                )

            result["unperturbed_frames"] = unperturbed_frames
            result["unperturbed_regressors"] = self.data_unperturbed.regressors
            result["unperturbed_wrenches"] = self.data_unperturbed.wrenches

        return result

    def _store_current_data(self, tgt_traj, act_traj):
        self.data.time.append(self.d.time)
        self.data.tgt_trajectory.append(tgt_traj)
        self.data.act_trajectory.append(act_traj)

        twist_sen, dtwist_sen, regressor = calculate_frame_dynamics(
            act_traj, self.inverse, self.id_ll, self.pose_x_ll, self.pose_ll_llj, self.pose_x_sen
        )
        self.data.twists_sen.append(twist_sen.tolist())
        self.data.dtwists_sen.append(dtwist_sen.tolist())

        linacc_sen_obji = get_linacc(twist_sen, dtwist_sen, self.pose_sen_obji)
        self.data.linaccs_sen_obji.append(linacc_sen_obji)

        wrench = self.sensors.get("wrench", perturbed=self.record_wrench_noise and self.perturb_wrench)
        self.data.wrenches.append(wrench)

        self.data.regressors.append(regressor)

        if self.get_unperturbed:
            # perturbed=False reads raw MuJoCo values and draws no rng, so the noisy stream is unaffected
            act_traj_clean = np.stack(self.sensors.get("jointvars", perturbed=False))  # type: ignore
            twist_clean, dtwist_clean, regressor_clean = calculate_frame_dynamics(
                act_traj_clean, self.inverse, self.id_ll, self.pose_x_ll, self.pose_ll_llj, self.pose_x_sen
            )
            self.data_unperturbed.act_trajectory.append(act_traj_clean)
            self.data_unperturbed.twists_sen.append(twist_clean.tolist())
            self.data_unperturbed.dtwists_sen.append(dtwist_clean.tolist())
            # perturbed=False reads raw sensordata and draws no rng
            self.data_unperturbed.wrenches.append(self.sensors.get("wrench", perturbed=False))
            self.data_unperturbed.regressors.append(regressor_clean)

        # Writing a single frame of a dataset =============================
        file_name = f"{self.data.frame_count:04}.png"
        self.recorder.render(self.d, file_name)  # recorder.cam_id is selected internally

        # Log NeMD ingredients ============================================
        # Items which need to be computed at every frame recoding
        pose_obj_cam = self.pose_x_obj.inv().dot(self.poses.x_cam[self.recorder.cam_id])
        self.data.transform_matrices.append(pose_obj_cam.as_matrix().tolist())

        self.data.poses_sen_obj.append(self.pose_sen_obj.as_matrix().tolist())
        self.data.poses_sen_obji.append(self.pose_sen_obji.as_matrix().tolist())
        # Store the path relative to the dataset root so that loaders on other machines can
        # resolve it with os.path.join(dataset_root, file_path)
        image_path = self.recorder.complete_image_dir / file_name
        self.data.file_paths.append(str(image_path.relative_to(self.recorder.dataset_dir)))

    def _set_ctrl(self, tgt_traj, act_traj):
        # Get residual of state
        mj_differentiatePos(  # Use this func to differenciate quat properly
            self.m,  # MjModel
            self.qpos_err,  # data container for the residual of qpos
            self.diffpos_dt,  # timestep used to numerically differentiate the pos
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

        self.data.qpos_errors.append(self.qpos_err)
        self.data.qvel_errors.append(traj_err[1])
        self.data.qacc_errors.append(traj_err[2])

    def _visualize(self, data: SimulationData):
        """Visualize simulation results."""
        mpl.rcParams["axes.xmargin"] = 0
        np.set_printoptions(precision=5, suppress=True)
        # Cast data into ndarrays for visualization
        tgt_trajectory = np.array(data.tgt_trajectory)
        trajectory = np.array(data.act_trajectory)
        frame_iter = np.arange(data.frame_count)
        wrenches = np.array(data.wrenches)

        # Actual and target joint positions
        qpos_fig, qpos_axes = plt.subplots(2, 1, sharex="col", tight_layout=True)
        qpos_fig.suptitle("qpos")
        qpos_axes[1].set(xlabel="time [s]")
        yls = ["q0-2 [m]", "q3-5 [rad]"]
        for i in range(len(qpos_axes)):
            slcr = slice(i * 3, (i + 1) * 3)
            ax_plot_lines_w_tgt(qpos_axes[i], data.time, trajectory[:, 0, slcr], tgt_trajectory[:, 0, slcr], yls[i])

        # Object linear acceleration and ft sensor measurements rel. to {sensor}
        acc_ft_fig, acc_ft_axes = plt.subplots(3, 1, tight_layout=True)
        ax_plot_lines(acc_ft_axes[0], frame_iter, data.linaccs_sen_obji, "recovered_linacc_sen_obji [m/s/s]")
        ax_plot_lines(acc_ft_axes[1], frame_iter, wrenches[:, :3], "frc_sen [N]")
        ax_plot_lines(acc_ft_axes[2], frame_iter, wrenches[:, 3:], "trq_sen [N*m]")
        for ax in acc_ft_axes:
            ax.hlines(0.0, frame_iter[0], frame_iter[-1], ls="dashed", alpha=0.5)

        qpos_path = self.recorder.dataset_dir / "tracking_qpos.png"
        acc_ft_path = self.recorder.dataset_dir / "tracking_acc_ft.png"
        qpos_fig.savefig(str(qpos_path), dpi=150)
        acc_ft_fig.savefig(str(acc_ft_path), dpi=150)
        plt.close(qpos_fig)
        plt.close(acc_ft_fig)
        print(f"Tracking plots saved to {qpos_path} and {acc_ft_path}")
