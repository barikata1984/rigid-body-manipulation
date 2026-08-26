from __future__ import annotations

from collections import deque

import numpy as np
from mujoco._structs import MjData, MjModel

from utilities import get_element_id

from .noise_profiles import covariance_from_profile, get_noise_profile


class Sensors:
    def __init__(
        self,
        m: MjModel,
        d: MjData,
        fps: float,
        noise_scale: float = 1.0,
        force_noise_scale: float = 1.0,
        translation_noise_scale: float = 1.0,
        rotation_noise_scale: float = 1.0,
        torque_noise_scale: float = 1.0,
        seed: int | None = None,
        noise_profile: str = "empirical",
        joint_bias_scale: float = 0.0,
        wrench_bias_scale: float = 0.0,
    ) -> None:
        self.m = m
        self.d = d
        self.fps = fps
        self._sensordata = d.sensordata
        self.profile = get_noise_profile(noise_profile)
        seed_sequence = np.random.SeedSequence(seed)
        self.seed = seed_sequence.entropy
        joint_seed, wrench_seed = seed_sequence.spawn(2)
        self.rng = np.random.default_rng(joint_seed)
        self._wrench_rng = np.random.default_rng(wrench_seed)

        nq = len(d.qpos)
        axis_scale = self._axis_values((translation_noise_scale,) * 3 + (rotation_noise_scale,) * 3, nq)
        self.jointpos_stddev = noise_scale * axis_scale * self._axis_values(self.profile.jointpos_stddev, nq)
        self.joint_bias_stddev = joint_bias_scale * self.jointpos_stddev
        self._joint_bias = self._get_noise(self.joint_bias_stddev)

        self.force_stddev = force_noise_scale * np.asarray(self.profile.wrench_stddev[:3], dtype=float)
        self.torque_stddev = torque_noise_scale * np.asarray(self.profile.wrench_stddev[3:], dtype=float)
        self.wrench_bias_stddev = wrench_bias_scale * np.concatenate((self.force_stddev, self.torque_stddev))
        self._wrench_bias = self._get_noise(self.wrench_bias_stddev, self._wrench_rng)

        # Public compatibility attributes for legacy diagnostics.
        self.jointvel_noise_scaler = np.sqrt(2) / self.profile.velocity_window_s
        self.jointacc_noise_scaler = self.jointvel_noise_scaler * np.sqrt(2) / (2 / fps)

        self._force_idx = get_sensor_measurement_idx(m, name="force")
        self._torque_idx = get_sensor_measurement_idx(m, name="torque")

        history_length = max(4, int(np.ceil(self.profile.velocity_window_s / m.opt.timestep)) + 3)
        self._qpos_history: deque[tuple[float, np.ndarray]] = deque(maxlen=history_length)
        self._joint_observation: np.ndarray | None = None
        self._joint_observation_time: float | None = None
        self._previous_qvel: np.ndarray | None = None
        self._previous_qacc: np.ndarray | None = None

        self._wrench_state = np.zeros(6)
        self._wrench_observation: np.ndarray | None = None
        self._wrench_observation_time: float | None = None
        self._wrench_next_sample_time: float | None = None
        self._setup_wrench_process(force_noise_scale, torque_noise_scale)

    @staticmethod
    def _axis_values(values: tuple[float, ...], size: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if size <= len(array):
            return array[:size].copy()
        raise ValueError(f"Noise profile has {len(array)} joint axes but model requires {size}")

    def _setup_wrench_process(self, force_scale: float, torque_scale: float) -> None:
        self._wrench_covariance = covariance_from_profile(self.profile, force_scale, torque_scale)
        self._wrench_phi = np.asarray(self.profile.wrench_lag1, dtype=float)
        phi_matrix = np.diag(self._wrench_phi)
        innovation_covariance = self._wrench_covariance - phi_matrix @ self._wrench_covariance @ phi_matrix
        # Fail instead of silently changing measured covariance if a future fit is invalid.
        np.linalg.cholesky(innovation_covariance)
        self._wrench_innovation_covariance = innovation_covariance

    def _get_noise(self, stddev: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = self.rng if rng is None else rng
        return generator.normal(np.zeros_like(stddev), stddev)

    def _raw_jointvars(self) -> np.ndarray:
        return np.stack((self.d.qpos.copy(), self.d.qvel.copy(), self.d.qacc.copy()))

    def _legacy_joint_observation(self) -> np.ndarray:
        qpos = self.d.qpos + self._get_noise(self.jointpos_stddev)
        qvel = self.d.qvel + self._get_noise(self.jointpos_stddev * self.jointvel_noise_scaler)
        qacc = self.d.qacc + self._get_noise(self.jointpos_stddev * self.jointacc_noise_scaler)
        return np.stack((qpos, qvel, qacc))

    def _position_at(self, target_time: float) -> np.ndarray:
        history = list(self._qpos_history)
        for (time_a, qpos_a), (time_b, qpos_b) in zip(history, history[1:], strict=False):
            if time_a <= target_time <= time_b:
                if time_b == time_a:
                    return qpos_b
                weight = (target_time - time_a) / (time_b - time_a)
                return (1.0 - weight) * qpos_a + weight * qpos_b
        return history[0][1]

    def _derived_joint_observation(self, time: float) -> np.ndarray:
        window = self.profile.velocity_window_s
        qpos = self.d.qpos.copy() + self._joint_bias + self._get_noise(self.jointpos_stddev)

        if not self._qpos_history:
            qpos_past = self.d.qpos - window * self.d.qvel
            qpos_past += self._joint_bias + self._get_noise(self.jointpos_stddev)
            self._qpos_history.append((time - window, qpos_past))
        self._qpos_history.append((time, qpos.copy()))

        qvel = (qpos - self._position_at(time - window)) / window
        if self._previous_qvel is None or self._joint_observation_time is None:
            qacc = self.d.qacc.copy()
        else:
            dt = time - self._joint_observation_time
            raw_qacc = (qvel - self._previous_qvel) / dt
            gain = self.profile.acceleration_filter_gain
            alpha = gain * dt / (1.0 + gain * dt)
            qacc = alpha * raw_qacc + (1.0 - alpha) * self._previous_qacc

        self._previous_qvel = qvel.copy()
        self._previous_qacc = qacc.copy()
        return np.stack((qpos, qvel, qacc))

    def sample_jointvars(self) -> np.ndarray:
        """Return one observed joint-state sample for the current MuJoCo time.

        Control and recording consume this same cached sample, while independently
        choosing it or the noise-free MuJoCo state.
        """

        time = float(self.d.time)
        if self._joint_observation_time == time and self._joint_observation is not None:
            return self._joint_observation.copy()

        if self.profile.joint_model == "independent_gaussian":
            observation = self._legacy_joint_observation()
        else:
            observation = self._derived_joint_observation(time)

        self._joint_observation = observation
        self._joint_observation_time = time
        return observation.copy()

    def sample_control_jointvars(self, *, derived_velocity: bool = False) -> np.ndarray:
        """Return the controller observation without conflating it with logged state.

        The empirical 34 ms velocity estimate reproduces the externally recorded
        signal. It is not evidence for the estimator used inside the robot servo.
        By default, control therefore sees the noisy encoder position together
        with MuJoCo's instantaneous velocity and acceleration estimates.
        """

        observation = self.sample_jointvars()
        if derived_velocity:
            return observation
        raw = self._raw_jointvars()
        observation[1:] = raw[1:]
        return observation

    def _raw_wrench(self) -> np.ndarray:
        return np.concatenate((self._sensordata[self._force_idx], self._sensordata[self._torque_idx]), axis=None)

    def _advance_wrench_process(self, time: float) -> None:
        sample_period = 1.0 / self.profile.wrench_sample_rate_hz
        if self._wrench_next_sample_time is None:
            self._wrench_state = self._wrench_rng.multivariate_normal(np.zeros(6), self._wrench_covariance)
            self._wrench_next_sample_time = time + sample_period
            return

        while self._wrench_next_sample_time <= time + np.finfo(float).eps:
            innovation = self._wrench_rng.multivariate_normal(np.zeros(6), self._wrench_innovation_covariance)
            self._wrench_state = self._wrench_phi * self._wrench_state + innovation
            self._wrench_next_sample_time += sample_period

    def _empirical_wrench_observation(self) -> np.ndarray:
        time = float(self.d.time)
        if self._wrench_observation_time == time and self._wrench_observation is not None:
            return self._wrench_observation.copy()

        self._advance_wrench_process(time)
        observation = self._raw_wrench() + self._wrench_bias + self._wrench_state
        quantization = np.asarray(self.profile.wrench_quantization, dtype=float)
        nonzero = quantization > 0
        observation[nonzero] = np.round(observation[nonzero] / quantization[nonzero]) * quantization[nonzero]
        self._wrench_observation = observation
        self._wrench_observation_time = time
        return observation.copy()

    def _perturbed_wrench(self) -> np.ndarray:
        if self.profile.wrench_model == "independent_gaussian":
            stddev = np.concatenate((self.force_stddev, self.torque_stddev))
            return self._raw_wrench() + self._get_noise(stddev, self._wrench_rng)
        return self._empirical_wrench_observation()

    def get(self, key: str, perturbed: bool = False) -> np.ndarray | tuple[np.ndarray, ...]:
        if key in {"jointpos", "jointvel", "jointacc", "jointvars"}:
            jointvars = self.sample_jointvars() if perturbed else self._raw_jointvars()
            if key == "jointvars":
                return tuple(jointvars)
            return jointvars[{"jointpos": 0, "jointvel": 1, "jointacc": 2}[key]]

        wrench = self._perturbed_wrench() if perturbed else self._raw_wrench()
        if key == "force":
            return wrench[:3]
        if key == "torque":
            return wrench[3:]
        if key == "wrench":
            return wrench
        raise ValueError(f"Unknown sensor key: {key}")

    def metadata(self) -> dict:
        return {
            "profile": self.profile.name,
            "seed": self.seed,
            "joint_model": self.profile.joint_model,
            "jointpos_stddev": self.jointpos_stddev.tolist(),
            "joint_bias_stddev": self.joint_bias_stddev.tolist(),
            "velocity_window_s": self.profile.velocity_window_s,
            "acceleration_filter_gain": self.profile.acceleration_filter_gain,
            "wrench_model": self.profile.wrench_model,
            "wrench_stddev": [*self.force_stddev.tolist(), *self.torque_stddev.tolist()],
            "wrench_bias_stddev": self.wrench_bias_stddev.tolist(),
            "wrench_sample_rate_hz": self.profile.wrench_sample_rate_hz,
            "wrench_quantization": list(self.profile.wrench_quantization),
        }


def get_sensor_measurement_idx(
    m: MjModel,
    name: str | None = None,
    id: int | None = None,
) -> list[int]:
    if id is None:
        if name is None:
            raise ValueError("'name' have to be set when 'id' is None")
        id = get_element_id(m, "sensor", name)

    idx = np.arange(m.sensor_dim[id]) + m.sensor_dim[:id].sum()
    return idx.tolist()
