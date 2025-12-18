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
    def generate(self, show_plot: bool = False, plot_path: str | None = None, json_path: str | None = None):
        """To be implemented in each child class"""
        pass

    def _write_to_json(self, pos, vel, acc, json_path="spline_trajectory.json"):
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

    def _plot(self, pos, vel, acc, show: bool = False, plot_path: str | None = None):
        """Plot the trajectory."""

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot Positions
        for j in range(self.num_joints):
            axes[0].plot(self.time_array, pos[:, j], label=f"Joint {j + 1}")
        axes[0].set_ylabel("Position [rad]")
        axes[0].set_title("Joint Positions")
        axes[0].legend()
        axes[0].grid(True)

        # Plot Velocities
        for j in range(self.num_joints):
            axes[1].plot(self.time_array, vel[:, j], label=f"Joint {j + 1}")
        axes[1].set_ylabel("Velocity [rad/s]")
        axes[1].set_title("Joint Velocities")
        axes[1].grid(True)

        # Plot Accelerations
        for j in range(self.num_joints):
            axes[2].plot(self.time_array, acc[:, j], label=f"Joint {j + 1}")
        axes[2].set_ylabel("Acceleration [rad/s^2]")
        axes[2].set_title("Joint Accelerations")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True)

        plt.tight_layout()

        if plot_path:
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")

        if show:
            plt.show()

    def _plot_single_ax(self, ax: Axes, data: np.ndarray, title: str, xlabel: str, ylabel: str):
        # Plot Accelerations
        for j in range(self.num_joints):
            ax.plot(self.time_array, data[:, j], label=f"Joint {j + 1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
