import numpy as np
from mujoco._structs import MjData, MjModel

from utilities import get_element_id


class Sensors:
    def __init__(
        self,
        m: MjModel,
        d: MjData,
        fps: float,
        jointpos_noise_stddev: list[float] | None = None,
    ) -> None:
        self.m = m
        self.d = d
        self._sensordata = d.sensordata
        self.jointpos_stddev = np.array(
            [5.0e-4, 5.0e-4, 5.0e-4, 1.0e-3, 1.0e-3, 1.0e-3]  # [m, m, m, rad, rad, rad], may strong
            # [1.0e-4, 1.0e-4, 1.0e-4, 5.0e-4, 5.0e-4, 5.0e-4]  # [m, m, m, rad, rad, rad], may weak
        )
        self.force_stddev = 2 * np.ones(3)  # [N]
        self.torque_stddev = 0.1 * np.ones(3)  # [Nm]
        self.jointvar_noise_scaler = np.sqrt(2) * fps
        self.rng = np.random.default_rng()

    def _get_noise(self, stddev) -> np.ndarray:
        return self.rng.normal(
            np.zeros_like(stddev),
            stddev,
        )

    def _get_jointpos(self, perturbed: bool) -> np.ndarray:
        if perturbed:
            return self.d.qpos + self._get_noise(self.jointpos_stddev)
        else:
            return self.d.qpos

    def _get_jointvel(self, perturbed: bool) -> np.ndarray:
        if perturbed:
            return self.d.qvel + self._get_noise(self.jointpos_stddev * self.jointvar_noise_scaler)
        else:
            return self.d.qvel

    def _get_jointacc(self, perturbed: bool) -> np.ndarray:
        if perturbed:
            return self.d.qacc + self._get_noise(self.jointpos_stddev * self.jointvar_noise_scaler**2)
        else:
            return self.d.qvel

    def _get_jointvars(self, perturbed: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qpos = self._get_jointpos(perturbed)
        qvel = self._get_jointvel(perturbed)
        qacc = self._get_jointacc(perturbed)
        return qpos, qvel, qacc

    def _get_force(self, perturbed: bool) -> np.ndarray:
        idx = get_sensor_measurement_idx(self.m, name="force")
        if perturbed:
            return self._sensordata[idx] + self._get_noise(self.force_stddev)
        else:
            return self._sensordata[idx]

    def _get_torque(self, perturbed: bool) -> np.ndarray:
        idx = get_sensor_measurement_idx(self.m, name="torque")
        if perturbed:
            return self._sensordata[idx] + self._get_noise(self.torque_stddev)
        else:
            return self._sensordata[idx]

    def _get_ft(self, perturbed) -> np.ndarray:
        force = self._get_force(perturbed=perturbed)
        torque = self._get_torque(perturbed=perturbed)
        return np.concatenate([force, torque], axis=None)

    def get(
        self,
        key,
        perturbed: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """Get sensor measurements.

        Args:
            key (_type_): _description_
            perturbed (bool, optional): _description_. Defaults to False.

        Returns:
            np.ndarray | tuple[np.ndarray, ...]: _description_
        """
        if key == "jointpos":
            return self._get_jointpos(perturbed)
        elif key == "jointvel":
            return self._get_jointvel(perturbed)
        elif key == "jointacc":
            return self._get_jointacc(perturbed)
        elif key == "jointvars":
            return self._get_jointvars(perturbed)
        elif key == "force":
            return self._get_force(perturbed)
        elif key == "torque":
            return self._get_torque(perturbed)
        elif key == "ft":
            return self._get_ft(perturbed)
        else:
            raise ValueError(f"Unknown sensor key: {key}")


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
