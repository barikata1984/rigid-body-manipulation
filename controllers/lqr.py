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
    input_gains: list[float] = MISSING


class LinearQuadraticRegulator:
    def __init__(self,
                 cfg: LinearQuadraticRegulatorConfig,
                 m: MjModel,
                 d: MjData,
                 ) -> None:

        # Fill a potentially missing field of a planner configuration
        try:
            cfg.input_gains
        except MissingMandatoryValue:
            cfg.input_gains = np.ones(m.nu).tolist()  # awkward but omegaconf
                                                      # does not support NDArray

        self.state_space = StateSpace(cfg.state_space, m, d)
        self.control_gain = compute_gain_matrix(self.state_space, cfg.input_gains)


def compute_gain_matrix(ss: StateSpace,
                        input_gains: ArrayLike,
                        ) -> NDArray:
    Q = np.eye(ss.ns)  # Initial state cost matrix
    R = np.diag(input_gains)  # Input gain matrix
    # Compute the feedback gain matrix K
    P = linalg.solve_discrete_are(ss.A, ss.B, Q, R)
    K = linalg.pinv(R + ss.B.T @ P @ ss.B) @ ss.B.T @ P @ ss.A

    return K

