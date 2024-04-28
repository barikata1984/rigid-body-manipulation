from dataclasses import dataclass, field

import numpy as np
from mujoco._structs import MjData, MjModel
from numpy.typing import ArrayLike, NDArray
from omegaconf import MISSING
from omegaconf.errors import MissingMandatoryValue
from scipy import linalg

from dynamics import StateSpaceConfig, StateSpace


@dataclass
class LinearQuadraticRegulatorConfig:
    target_class: str = "LinearQuadraticRegulator"
    state_space: StateSpaceConfig = StateSpaceConfig()
    input_gain: list[float] = MISSING


class LinearQuadraticRegulator:
    def __init__(self,
                 cfg: LinearQuadraticRegulatorConfig,
                 m: MjModel,
                 d: MjData,
                 ) -> None:

        # Fill a potentially missing field of a planner configuration
        try:
            cfg.input_gain
        except MissingMandatoryValue:
            cfg.input_gain = np.ones(m.nu).tolist()  # awkward but omegaconf
                                                      # does not support NDArray

        self.ss = StateSpace(cfg.state_space, m, d)
        self.input_gain = cfg.input_gain
        self.gain_matrix = self.update_control_gain(m, d)

    def update_control_gain(self,
                            m: MjModel,
                            d: MjData,
                            ) -> NDArray:
        self.ss.update_matrices(m, d)

        Q = np.eye(self.ss.ns)  # Initial state cost matrix R = np.diag(self.input_gains)  # Input gain matrix
        R = np.diag(self.input_gain)  # Input gain matrix
        # Compute the feedback gain matrix K
        P = linalg.solve_discrete_are(self.ss.A, self.ss.B, Q, R)
        K = linalg.pinv(R + self.ss.B.T @ P @ self.ss.B) @ self.ss.B.T @ P @ self.ss.A

        return K

