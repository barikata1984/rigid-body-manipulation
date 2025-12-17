import json
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


# 抽象クラスの定義
class BaseTrajectory(ABC):

    def __init__(self, duration: float, fps: float):
        self.duration = duration
        self.fps = fps
    
    @abstractmethod
    def generate(self):
        """To be implemented in each child class"""
        pass

    def save_to_json(self, pos, vel, acc, time, filename: str) -> None:
        """Saves the trajectory to a JSON file.

        Args:
            filename: The path to the JSON file.
        """
        data = {
            "duration": self.duration,
            "fps": self.fps,
            "time": time.tolist(),
            "pos": pos.tolist(),
            "vel": vel.tolist(),
            "acc": acc.tolist(),
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Trajectory saved to {filename}")


    def plot(self, pos, vel, acc, time, show: bool = True, filename: str | None = None):
        """Visualizes the trajectory."""

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot Positions
        for j in range(self.num_joints):
            axes[0].plot(time, pos[:, j], label=f"Joint {j + 1}")
        axes[0].set_ylabel("Position [rad]")
        axes[0].set_title("Joint Positions")
        axes[0].legend()
        axes[0].grid(True)

        # Plot Velocities
        for j in range(self.num_joints):
            axes[1].plot(time, vel[:, j], label=f"Joint {j + 1}")
        axes[1].set_ylabel("Velocity [rad/s]")
        axes[1].set_title("Joint Velocities")
        axes[1].grid(True)

        # Plot Accelerations
        for j in range(self.num_joints):
            axes[2].plot(time, acc[:, j], label=f"Joint {j + 1}")
        axes[2].set_ylabel("Acceleration [rad/s^2]")
        axes[2].set_title("Joint Accelerations")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True)

        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            print(f"Plot saved to {filename}")

        if show:
            plt.show()
