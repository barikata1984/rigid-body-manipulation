import re
from dataclasses import dataclass
from math import pi

from mujoco._structs import MjData, MjModel, MjOption
from omegaconf import MISSING

from .base_planner import BasePlannerConfig
from .fifth_order_jspline_interpolation import get_trajectory_interpolated_with_fifth_order_spline


@dataclass
class JointPositionPlannerConfig(BasePlannerConfig):
    displacements: list[float] = MISSING
    target_class: str = "JointPositionPlanner"  # type: ignore
    pos_offset: list[float] | None = None


class JointPositionPlanner:
    def __init__(
        self,
        cfg: JointPositionPlannerConfig,
        m: MjModel,
        d: MjData,
        *args,
        **kwargs,
    ) -> None:
        # Fill a potentially missing field of a planner configuration
        if cfg.pos_offset is None:
            self.pos_offset = d.qpos.copy().tolist()

        self.duration = kwargs.get("duration")
        self._timestep = MjOption().timestep
        self.n_steps = int(self.duration / self._timestep)

        displacements = []
        for _disp in cfg.displacements:
            disp = _disp.__repr__().strip("'")  # not sure this is the best solution...
            try:
                disp = float(disp)
            except ValueError:
                disp = self.safe_eval(self.replace_pi(disp))

            displacements.append(disp)

        self.displacements = displacements
        self.plan = get_trajectory_interpolated_with_fifth_order_spline(
            self.displacements,  # [m, m, m, rad, rad, rad]
            self.pos_offset,  # [m, m, m, rad, rad, rad]
            self._timestep,  # [s]
            self.n_steps,  # [steps]
        )

    def safe_eval(self, expr):
        """Evaluates a mathematical expression if it contains only allowed characters."""
        allowed_chars = "0123456789.+*/-() "
        if all(c in allowed_chars for c in expr):
            return eval(expr)
        else:
            raise ValueError("Invalid characters in expression.")

    def replace_pi(self, text):
        """Replaces occurrences of "pi" with its numerical value in a string."""
        return re.sub(r"\bpi\b", str(pi), text)
