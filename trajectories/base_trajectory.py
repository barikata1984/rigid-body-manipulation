import json
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


# 抽象クラスの定義
class BaseTrajectory(ABC):
    def __init__(self, duration: float, fps: float):
        self.duration = duration
        self.fps = fps

        self.time_steps = int(self.duration * self.fps)
        self.time_array = np.linspace(0, self.duration, self.time_steps)

    @abstractmethod
    def generate(self, *args, **kwargs):
        """To be implemented in each child class"""
        pass

    def write_to_json(self, pos, vel, acc, json_path="spline_trajectory.json"):
        """
        Save the trajectory to a JSON file.
        Structure:
        {
            "duration": float,
            "fps": float,
            "frames": [
                {
                    "qpos": [float, ...],
                    "qvel": [float, ...],
                    "qacc": [float, ...]
                },
                ...
            ]
        }
        """
        frames = []
        for i in range(len(self.time_array)):
            frame = {"qpos": pos[i].tolist(), "qvel": vel[i].tolist(), "qacc": acc[i].tolist()}
            frames.append(frame)

        data = {"duration": self.duration, "fps": self.fps, "frames": frames}

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Trajectory JSON saved to {json_path}")

    def plot(self, pos, vel, acc, show: bool = False, plot_path: str | None = None):
        """Plot the trajectory."""

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        self._plot_single_ax(axes[0], pos, "Joint Positions", "Time [s]", "Position [rad]")
        self._plot_single_ax(axes[1], vel, "Joint Velocities", "Time [s]", "Velocity [rad/s]")
        self._plot_single_ax(axes[2], acc, "Joint Accelerations", "Time [s]", "Acceleration [rad/s^2]")

        plt.tight_layout()

        if plot_path:
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")

        if show:
            plt.show()

    def _plot_single_ax(self, ax: Axes, data: np.ndarray, title: str, xlabel: str, ylabel: str):
        # Plot Accelerations
        for j, d in enumerate(data.T):
            ax.plot(self.time_array, d, label=f"Joint {j + 1}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
