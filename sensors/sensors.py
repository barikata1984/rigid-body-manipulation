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
            # [5.0e-4, 5.0e-4, 5.0e-4, 1.0e-3, 1.0e-3, 1.0e-3]  # [m, m, m, rad, rad, rad], may too strong
            [1.0e-4, 1.0e-4, 1.0e-4, 5.0e-4, 5.0e-4, 5.0e-4]  # [m, m, m, rad, rad, rad], may too weak
        )
        self.force_stddev = 2 * np.ones(3)  # [N]
        self.torque_stddev = 0.1 * np.ones(3)  # [Nm]
        self.jointvar_noise_scaler = np.sqrt(2) * fps
        self.rng = np.random.default_rng()

    def get_noise(self, stddev) -> np.ndarray:
        return self.rng.normal(
            np.zeros_like(stddev),
            stddev,
        )

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
        match key:
            case "jointpos":
                if perturbed:
                    return self.d.qpos + self.get_noise(self.jointpos_stddev)
                else:
                    return self.d.qpos

            case "jointvel":
                if perturbed:
                    return self.d.qvel + self.get_noise(self.jointpos_stddev * self.jointvar_noise_scaler)
                else:
                    return self.d.qvel

            case "jointacc":
                if perturbed:
                    return self.d.qacc + self.get_noise(self.jointpos_stddev * self.jointvar_noise_scaler**2)
                else:
                    return self.d.qvel

            case "jointvars":
                qpos = self.get("jointpos", perturbed)
                qvel = self.get("jointvel", perturbed)
                qacc = self.get("jointacc", perturbed)
                return qpos, qvel, qacc  # shape: (6,), (6,), (6,)

            case "force":
                idx = get_sensor_measurement_idx(self.m, name=key)
                if perturbed:
                    return self._sensordata[idx] + self.get_noise(self.force_stddev)
                else:
                    return self._sensordata[idx]

            case "torque":
                idx = get_sensor_measurement_idx(self.m, name=key)
                if perturbed:
                    return self._sensordata[idx] + self.get_noise(self.torque_stddev)
                else:
                    return self._sensordata[idx]

            case "ft":
                force = self.get("force", perturbed=True)
                torque = self.get("torque", perturbed=True)
                return np.concatenate([force, torque], axis=None)


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
