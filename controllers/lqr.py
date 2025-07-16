from dataclasses import dataclass, field

import numpy as np
from mujoco._structs import MjData, MjModel
from numpy.typing import NDArray
from omegaconf import MISSING
from scipy import linalg

from dynamics import StateSpace, StateSpaceConfig

from .base_controller import BaseControllerConfig


@dataclass
class LinearQuadraticRegulatorConfig(BaseControllerConfig):
    target_class: str = "LinearQuadraticRegulator"
    state_space: StateSpaceConfig = field(default_factory=StateSpaceConfig)
    input_gain: list[float] = MISSING


class LinearQuadraticRegulator:
    def __init__(
        self,
        cfg: LinearQuadraticRegulatorConfig,
        m: MjModel,
        d: MjData,
    ) -> None:
        # Fill a potentially missing field of a planner configuration
        try:
            self.input_gain = np.array(cfg.input_gain)
        except ValueError as e:
            # 変換に失敗した場合、ValueErrorが発生し、このブロックが実行される
            print(
                f"{e}: Value for LinearQuadraticRegulatorConfig's attribute 'duration' is not properly set. Check the setting. It may be set at a string that are not castabble to flaot."
            )

        self.ss = StateSpace(cfg.state_space, m, d)
        self.gain_matrix = self.update_control_gain(m, d)

    def update_control_gain(
        self,
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
