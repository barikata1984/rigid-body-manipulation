from __future__ import annotations

import numpy as np
from mujoco._structs import MjData, MjModel

from utilities import get_element_id

from .noise_profiles import NOISE_PROFILE, covariance_from_profile


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
        wrench_bias_scale: float = 0.0,
    ) -> None:
        self.m = m
        self.d = d
        self.fps = fps
        self._sensordata = d.sensordata
        self.profile = NOISE_PROFILE
        seed_sequence = np.random.SeedSequence(seed)
        self.seed = seed_sequence.entropy
        joint_seed, wrench_seed = seed_sequence.spawn(2)
        self.rng = np.random.default_rng(joint_seed)
        self._wrench_rng = np.random.default_rng(wrench_seed)

        nq = len(d.qpos)
        axis_scale = self._axis_values((translation_noise_scale,) * 3 + (rotation_noise_scale,) * 3, nq)
        self.jointpos_stddev = noise_scale * axis_scale * self._axis_values(self.profile.jointpos_stddev, nq)

        self.force_stddev = force_noise_scale * np.asarray(self.profile.wrench_stddev[:3], dtype=float)
        self.torque_stddev = torque_noise_scale * np.asarray(self.profile.wrench_stddev[3:], dtype=float)
        self.wrench_bias_stddev = wrench_bias_scale * np.concatenate((self.force_stddev, self.torque_stddev))
        self._wrench_bias = self._get_noise(self.wrench_bias_stddev, self._wrench_rng)

        self.jointvel_noise_scaler = self.profile.jointvel_noise_scaler
        self.jointacc_noise_scaler = self.profile.jointacc_noise_scaler

        self._force_idx = get_sensor_measurement_idx(m, name="force")
        self._torque_idx = get_sensor_measurement_idx(m, name="torque")

        self._joint_observation: np.ndarray | None = None
        self._joint_observation_time: float | None = None

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

    def _independent_joint_observation(self) -> np.ndarray:
        qpos = self.d.qpos + self._get_noise(self.jointpos_stddev)
        qvel = self.d.qvel + self._get_noise(self.jointpos_stddev * self.jointvel_noise_scaler)
        qacc = self.d.qacc + self._get_noise(self.jointpos_stddev * self.jointacc_noise_scaler)
        return np.stack((qpos, qvel, qacc))

    def sample_jointvars(self) -> np.ndarray:
        """Return one observed joint-state sample for the current MuJoCo time.

        Control and recording consume this same cached sample, while independently
        choosing it or the noise-free MuJoCo state.
        """

        time = float(self.d.time)
        if self._joint_observation_time == time and self._joint_observation is not None:
            return self._joint_observation.copy()

        observation = self._independent_joint_observation()

        self._joint_observation = observation
        self._joint_observation_time = time
        return observation.copy()

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

    def get(self, key: str, perturbed: bool = False) -> np.ndarray | tuple[np.ndarray, ...]:
        if key in {"jointpos", "jointvel", "jointacc", "jointvars"}:
            jointvars = self.sample_jointvars() if perturbed else self._raw_jointvars()
            if key == "jointvars":
                return tuple(jointvars)
            return jointvars[{"jointpos": 0, "jointvel": 1, "jointacc": 2}[key]]

        wrench = self._empirical_wrench_observation() if perturbed else self._raw_wrench()
        if key == "force":
            return wrench[:3]
        if key == "torque":
            return wrench[3:]
        if key == "wrench":
            return wrench
        raise ValueError(f"Unknown sensor key: {key}")

    def metadata(self) -> dict:
        metadata = {
            "profile": self.profile.name,
            "seed": self.seed,
            "joint_model": self.profile.joint_model,
            "jointpos_stddev": self.jointpos_stddev.tolist(),
            "wrench_model": self.profile.wrench_model,
            "wrench_stddev": [*self.force_stddev.tolist(), *self.torque_stddev.tolist()],
            "wrench_bias_stddev": self.wrench_bias_stddev.tolist(),
            "wrench_sample_rate_hz": self.profile.wrench_sample_rate_hz,
            "wrench_quantization": list(self.profile.wrench_quantization),
        }

        metadata["jointvel_stddev"] = (self.jointpos_stddev * self.jointvel_noise_scaler).tolist()
        metadata["jointacc_stddev"] = (self.jointpos_stddev * self.jointacc_noise_scaler).tolist()
        return metadata


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
