from dataclasses import dataclass, field

import numpy as np
from mujoco._structs import MjData, MjModel
from numpy.typing import NDArray
from omegaconf import MISSING
from omegaconf.errors import MissingMandatoryValue
from scipy import linalg

from dynamics import StateSpace, StateSpaceConfig

from .base_controller import BaseControllerConfig


@dataclass
class LinearQuadraticRegulatorConfig(BaseControllerConfig):
    target_class: str = "LinearQuadraticRegulator"  # type: ignore
    state_space: StateSpaceConfig = field(default_factory=StateSpaceConfig)
    state_gain: list[float] = MISSING
    input_gain: list[float] = MISSING


class LinearQuadraticRegulator:
    def __init__(
        self,
        cfg: LinearQuadraticRegulatorConfig,
        *args,
        **kwargs,
    ) -> None:
        m = kwargs["model"]
        d = kwargs["data"]

        self.ss = StateSpace(cfg.state_space, m, d)

        # Fill a potentially missing field of a planner configuration
        try:
            self.state_gain = np.array(cfg.state_gain)
            self.input_gain = np.array(cfg.input_gain)
        except (ValueError, MissingMandatoryValue) as e:
            raise ValueError(f"state_gain or input_gain is not properly set: {e}")

        if len(self.state_gain) != self.ss.ns:
            raise ValueError(
                f"Length of state_gain ({len(self.state_gain)}) does not match the number of states ({self.ss.ns})."
            )

        self.gain_matrix = self.update_control_gain(m, d)

    def update_control_gain(
        self,
        m: MjModel,
        d: MjData,
    ) -> NDArray:
        self.ss.update_matrices(m, d)

        # State gain matrix
        Q = np.diag(self.state_gain)
        # Input gain matrix
        R = np.diag(self.input_gain)
        # Compute the feedback gain matrix K
        P = linalg.solve_discrete_are(self.ss.A, self.ss.B, Q, R)
        K = linalg.pinv(R + self.ss.B.T @ P @ self.ss.B) @ self.ss.B.T @ P @ self.ss.A

        return K
