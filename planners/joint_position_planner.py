import re
from dataclasses import dataclass
from math import pi

from mujoco._structs import MjData, MjModel
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
        pos_offset = d.qpos.copy().tolist() if cfg.pos_offset is None else cfg.pos_offset

        duration = kwargs.get("duration")
        fps = kwargs.get("fps")
        n_frames = int(duration * fps)

        print(f"{cfg.displacements=}")

        self.trajectories = get_trajectory_interpolated_with_fifth_order_spline(
            cfg.displacements,  # [m, m, m, rad, rad, rad]
            pos_offset,  # [m, m, m, rad, rad, rad]
            1.0 / fps,  # [s]  # type: ignore
            n_frames,
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
