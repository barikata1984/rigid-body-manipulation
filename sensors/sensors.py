from typing import Optional

import numpy as np
from mujoco._structs import MjData, MjModel

from utilities import get_element_id

class Sensors:
    def __init__(self,
                 m: MjModel,
                 d: MjData,
                 ) -> None:
        self.m = m
        self._sensordata = d.sensordata

    def get(self,
            key,
            ):

        idx = get_sensor_measurement_idx(self.m, key)
        return self._sensordata[idx]


def get_sensor_measurement_idx(m: MjModel,
                               name: Optional[str] = None,
                               id: Optional[int] = None,
                               ) -> list[int]:

    if id is None:
        if name is None:
            raise ValueError("'name' have to be set when 'id' is None")
        id = get_element_id(m, "sensor", name)

    idx = np.arange(m.sensor_dim[id]) + m.sensor_dim[:id].sum()

    return idx.tolist()
