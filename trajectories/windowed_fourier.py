from dataclasses import dataclass

from omegaconf import OmegaConf

from factory import instantiate

from .base_trajectory import BaseTrajectory
from .fourier import FourierTrajectoryConfig
from .window import WindowTrajectoryConfig


@dataclass
class WindowedFourierTrajectoryConfig(FourierTrajectoryConfig):
    """Configuration for WindowedFourierTrajectory.

    Uses a Fourier trajectory config file and automatically creates
    a matching window trajectory.
    """

    target_class: str = "WindowedFourierTrajectory"


class WindowedFourierTrajectory(BaseTrajectory):
    """
    Windowed Fourier Trajectory.

    Generates a trajectory defined by the product of a Fourier trajectory and a window function:
    Q(t) = w(t) * q_f(t)

    This ensures that the trajectory satisfies zero boundary conditions (pos, vel, acc = 0 at t=0, T)
    provided the window function satisfies them (which WindowTrajectory does).
    """

    def __init__(self, cfg: WindowedFourierTrajectoryConfig, *args, **kwargs):
        # Load Fourier config from file first
        super().__init__(cfg, *args, **kwargs)

        # Merge: fourier_config < windowed_fourier_config (higher priority overrides)
        if cfg.config is not None:
            fourier_cfg = OmegaConf.to_object(OmegaConf.merge(FourierTrajectoryConfig, OmegaConf.load(cfg.config)))
        else:
            fourier_cfg = cfg

        # Instantiate Fourier Trajectory using factory
        self.fourier = instantiate(fourier_cfg, *args, **kwargs)

        # Create Window Trajectory using duration, fps, and num_joints from Fourier config
        win_cfg = WindowTrajectoryConfig(
            duration=self.duration,
            fps=self.fps,
            num_joints=self.fourier.num_joints,
        )
        self.window = instantiate(win_cfg, *args, **kwargs)

    def get_value(self):
        """
        Calculate Q, dQ, ddQ at time t.

        Q = s * q
        dQ = ds*q + s*dq
        ddQ = dds*q + 2*ds*dq + s*ddq
        """

        # 1. Fourier Raw
        q_raw, dq_raw, ddq_raw = self.fourier.get_value()

        # 2. Window Values
        s, ds, dds = self.window.get_value()

        # 3. Product Rule
        q_out = s * q_raw
        dq_out = ds * q_raw + s * dq_raw
        ddq_out = dds * q_raw + 2 * ds * dq_raw + s * ddq_raw

        return q_out, dq_out, ddq_out

    def _generate(self, *args, **kwargs):
        if self.time_array[-1] > self.duration + 1e-9:
            self.time_array = self.time_array[:-1]

        pos, vel, acc = self.get_value()

        return pos, vel, acc
