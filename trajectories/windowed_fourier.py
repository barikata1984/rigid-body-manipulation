from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from factory import instantiate

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig
from .window import WindowTrajectoryConfig


@dataclass
class WindowedFourierTrajectoryConfig(BaseTrajectoryConfig):
    """Configuration for WindowedFourierTrajectory.

    Uses a Fourier trajectory config file and automatically creates
    a matching window trajectory.
    """

    fourier_config: Path | None = None  # Path to Fourier trajectory YAML configuration
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
        if cfg.fourier_config is None:
            raise ValueError("fourier_config must be specified")

        fourier_cfg = OmegaConf.load(cfg.fourier_config)

        # Merge: fourier_config < windowed_fourier_config (higher priority overrides)
        # This allows windowed_fourier_config to override duration/fps from fourier_config
        merged_cfg = OmegaConf.merge(fourier_cfg, OmegaConf.structured(cfg))

        import pdb

        pdb.set_trace()

        # Now call parent init with proper duration/fps
        super().__init__(merged_cfg, *args, **kwargs)

        # Instantiate Fourier Trajectory using factory
        self.fourier_trajectory = instantiate(fourier_cfg, *args, **kwargs)
        self.num_joints = fourier_cfg.num_joints

        # Create Window Trajectory using duration, fps, and num_joints from Fourier config
        win_cfg = WindowTrajectoryConfig(
            duration=self.duration,
            fps=self.fps,
            num_joints=self.num_joints,
        )
        self.window_trajectory = instantiate(win_cfg, *args, **kwargs)

    def get_value(self):
        """
        Calculate Q, dQ, ddQ at time t.

        Q = s * q
        dQ = ds*q + s*dq
        ddQ = dds*q + 2*ds*dq + s*ddq
        """

        # 1. Fourier Raw
        q_raw, dq_raw, ddq_raw = self.fourier_trajectory.get_value()

        # 2. Window Values
        s, ds, dds = self.window_trajectory.get_value()

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
