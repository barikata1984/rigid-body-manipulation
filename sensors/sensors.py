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
            [1.0e-3, 1.0e-3, 1.0e-3, 5.0e-3, 5.0e-3, 5.0e-3]  # [m, m, m, rad, rad, rad]
        )
        self.noise_scaler = np.sqrt(2) * fps
        self.rng = np.random.default_rng()

    def get_jointpos_noise(self) -> np.ndarray:
        return self.rng.normal(
            np.zeros_like(self.jointpos_stddev),
            self.jointpos_stddev,
        )

    def get(
        self,
        key,
        perturbed: bool = False,
    ):
        match key:
            case "jointpos":
                return self.d.qpos + self.get_jointpos_noise() if perturbed else self.d.qpos
            case "jointvel":
                return self.d.qvel + self.noise_scaler * self.get_jointpos_noise() if perturbed else self.d.qvel
            case "jointacc":
                return self.d.qacc + self.noise_scaler**2 * self.get_jointpos_noise() if perturbed else self.d.qvel
            # shape: (6,), (6,), (6,)
            case "jointvars":
                qpos = self.get("jointpos", perturbed)
                qvel = self.get("jointvel", perturbed)
                qacc = self.get("jointacc", perturbed)
                return qpos, qvel, qacc  # shape: (6,), (6,), (6,)
            case _:  # default case
                idx = get_sensor_measurement_idx(self.m, name=key)
                return self._sensordata[idx]


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
