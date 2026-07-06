import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from omegaconf import MISSING

from factory import InstantiateConfig


@dataclass
class BaseTrajectoryConfig(InstantiateConfig):
    duration: float = MISSING
    fps: float = MISSING
    module_name: str = "trajectories"

    # CLI-specific arguments (shared across all trajectories)
    show_plot: bool = False  # Show plot window (default: hidden)
    plot_path: Path | None = None  # Path to save plot image
    json_path: Path | None = None  # Path to save trajectory JSON
    config_class: str | None = None  # Config class name (from YAML metadata)


# 抽象クラスの定義
class BaseTrajectory(ABC):
    def __init__(self, cfg: BaseTrajectoryConfig, *args, **kwargs):
        self.duration = cfg.duration
        self.fps = cfg.fps

        self.time_steps = int(self.duration * self.fps)
        self.time_array = np.linspace(0, self.duration, self.time_steps)

    def write_to_json(self, pos, vel, acc, json_path="spline_trajectory.json", metadata=None):
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
            ],
            "metadata": {...}  # optional, omitted when None
        }
        """
        frames = []
        for i in range(len(self.time_array)):
            # frame = {"qpos": pos[i].tolist(), "qvel": vel[i].tolist(), "qacc": acc[i].tolist()}
            frame = [pos[i].tolist(), vel[i].tolist(), acc[i].tolist()]
            frames.append(frame)

        data = {"duration": self.duration, "fps": self.fps, "frames": frames}
        if metadata:
            data["metadata"] = metadata

        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
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
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")

        if show:
            plt.show()

        plt.close(fig)

    def _plot_single_ax(self, ax: Axes, data: np.ndarray, title: str, xlabel: str, ylabel: str):
        # Plot Accelerations
        for j, d in enumerate(data.T):
            ax.plot(self.time_array, d, label=f"Joint {j + 1}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    @staticmethod
    def _build_observation_matrix(
        time_array: np.ndarray,
        kinematics_func: Callable,
        traj_q: np.ndarray,
        traj_dq: np.ndarray,
        traj_ddq: np.ndarray,
    ) -> np.ndarray:
        """Build the observation matrix Y = sum_k A_k.T @ A_k."""
        Y: np.ndarray | None = None
        for i in range(len(time_array)):
            A_k = kinematics_func(traj_q[i], traj_dq[i], traj_ddq[i])
            if Y is None:
                n_params = A_k.shape[1]
                Y = np.zeros((n_params, n_params))
            Y += A_k.T @ A_k
        return Y

    @staticmethod
    def compute_condition_number(
        time_array: np.ndarray,
        kinematics_func: Callable,
        traj_q: np.ndarray,
        traj_dq: np.ndarray,
        traj_ddq: np.ndarray,
    ) -> float:
        """Compute the condition number of the observation matrix Y = sum_k A_k.T @ A_k.

        The matrix size is determined dynamically from the first regressor call,
        so no hardcoded DOF or parameter count is required.

        Args:
            time_array: 1-D array of time steps (length N).
            kinematics_func: callable f(q_i, dq_i, ddq_i) -> regressor (n_rows, n_params).
            traj_q: (N, n_joints) position array.
            traj_dq: (N, n_joints) velocity array.
            traj_ddq: (N, n_joints) acceleration array.

        Returns:
            Condition number (max_eig / min_eig), or 1e9 if the matrix is (near-)singular.
        """
        Y = BaseTrajectory._build_observation_matrix(time_array, kinematics_func, traj_q, traj_dq, traj_ddq)
        eigvals = np.linalg.eigvalsh(Y)
        min_eig = float(np.min(eigvals))
        max_eig = float(np.max(eigvals))

        if min_eig < 1e-9:
            return 1e9
        return max_eig / min_eig

    @staticmethod
    def compute_d_optimal(
        time_array: np.ndarray,
        kinematics_func: Callable,
        traj_q: np.ndarray,
        traj_dq: np.ndarray,
        traj_ddq: np.ndarray,
    ) -> float:
        """D-optimal objective: -log det(Y) = -sum(log(eigvals(Y))).

        Equivalent to -2 * sum(log(sigma_i)) of the stacked regressor W,
        since eigenvalues of Y = W^T W are the squared singular values of W.

        Returns:
            D-optimal value, or 1e9 on numerical failure.
        """
        try:
            Y = BaseTrajectory._build_observation_matrix(time_array, kinematics_func, traj_q, traj_dq, traj_ddq)
            eigvals = np.linalg.eigvalsh(Y)
            eigvals_floored = np.maximum(eigvals, 1e-30)
            return float(-np.sum(np.log(eigvals_floored)))
        except (np.linalg.LinAlgError, ValueError):
            return 1e9

    @staticmethod
    def compute_objective_with_cond(
        time_array: np.ndarray,
        kinematics_func: Callable,
        traj_q: np.ndarray,
        traj_dq: np.ndarray,
        traj_ddq: np.ndarray,
        objective_type: str = "condition_number",
    ) -> tuple[float, float]:
        """Compute objective value and condition number from a single Y matrix.

        Returns:
            (objective_value, condition_number).
        """
        try:
            Y = BaseTrajectory._build_observation_matrix(time_array, kinematics_func, traj_q, traj_dq, traj_ddq)
            eigvals = np.linalg.eigvalsh(Y)
            min_eig = float(np.min(eigvals))
            max_eig = float(np.max(eigvals))

            if min_eig < 1e-9:
                cond = 1e9
            else:
                cond = max_eig / min_eig

            if objective_type == "d_optimal":
                eigvals_floored = np.maximum(eigvals, 1e-30)
                obj = float(-np.sum(np.log(eigvals_floored)))
            else:
                obj = cond

            return obj, cond
        except (np.linalg.LinAlgError, ValueError):
            return 1e9, 1e9

    @abstractmethod
    def _generate(self, *args, **kwargs):
        """To be implemented in each child class"""
        pass

    def _trajectory_metadata(self) -> dict:
        """Trajectory-type-specific metadata merged into the JSON output.

        Base trajectories contribute nothing; subclasses (e.g. ExcitedTrajectory)
        override this to record optimization results such as the condition number.
        """
        return {}

    def generate(self, *args, **kwargs):
        show_plot = kwargs.get("show_plot", False)
        plot_path = kwargs.get("plot_path", None)
        json_path = kwargs.get("json_path", None)
        metadata = kwargs.get("metadata", None)

        if self.time_array[-1] > self.duration + 1e-9:
            self.time_array = self.time_array[:-1]

        pos, vel, acc = self._generate(*args, **kwargs)

        traj_metadata = self._trajectory_metadata()
        if metadata or traj_metadata:
            metadata = {**(metadata or {}), **traj_metadata}

        if json_path is not None:
            self.write_to_json(pos, vel, acc, json_path, metadata=metadata)

        self.plot(pos, vel, acc, show=show_plot, plot_path=plot_path)

        return pos, vel, acc
