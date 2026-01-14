from dataclasses import dataclass, field

from omegaconf import MISSING, OmegaConf

from factory import instantiate

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig
from .fourier import FourierTrajectoryConfig
from .window import WindowTrajectory, WindowTrajectoryConfig


@dataclass(kw_only=True)
class WindowedFourierTrajectoryConfig(BaseTrajectoryConfig):
    # Main component
    fourier_trajectory: FourierTrajectoryConfig = field(default_factory=lambda: MISSING)

    # Window component (optional, defaults to standard window)
    window_trajectory: WindowTrajectoryConfig | None = None

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
        super().__init__(cfg, *args, **kwargs)

        # Instantiate Fourier Trajectory
        fourier_cfg = cfg.fourier_trajectory
        if isinstance(fourier_cfg, dict):
            fourier_cfg = OmegaConf.to_object(OmegaConf.merge(FourierTrajectoryConfig(), fourier_cfg))

        # Ensure sub-configs inherit master settings
        fourier_cfg.duration = self.duration
        fourier_cfg.fps = self.fps

        self.fourier_trajectory = instantiate(fourier_cfg, *args, **kwargs)
        self.num_joints = self.fourier_trajectory.num_joints

        # Instantiate Window Trajectory
        if cfg.window_trajectory is None:
            win_cfg = WindowTrajectoryConfig(duration=self.duration, fps=self.fps, num_joints=self.num_joints)
        else:
            win_cfg = cfg.window_trajectory
            win_cfg.duration = self.duration
            win_cfg.fps = self.fps
            win_cfg.num_joints = self.num_joints

        self.window_trajectory = WindowTrajectory(win_cfg, *args, **kwargs)

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

    def generate(self, show_plot: bool = False, plot_path: str | None = None, json_path: str | None = None):
        if self.time_array[-1] > self.duration + 1e-9:
            self.time_array = self.time_array[:-1]

        pos, vel, acc = self.get_value()

        self.plot(pos, vel, acc, show=show_plot, plot_path=plot_path)

        if json_path is not None:
            self.write_to_json(pos, vel, acc, json_path)

        return pos, vel, acc
