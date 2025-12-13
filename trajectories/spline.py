<<<<<<< HEAD
import json

import matplotlib.pyplot as plt
import numpy as np


class QuinticSplineTrajectory:
    """Generates a quintic (5th-order) polynomial trajectory for multiple joints.
    Ensures continuous position, velocity, and acceleration.
    """

    def __init__(
        self,
        start_pos: list[float],
        end_pos: list[float],
        duration: float,
        frequency: float,
        start_vel: list[float] | None = None,
        end_vel: list[float] | None = None,
        start_acc: list[float] | None = None,
        end_acc: list[float] | None = None,
    ):
        """Initialize the trajectory generator.

        Args:
            start_pos: List of start positions for each joint.
            end_pos: List of end positions for each joint.
            duration: Total duration of the trajectory in seconds.
            frequency: Sampling frequency in Hz.
            start_vel: (Optional) List of start velocities. Defaults to 0.
            end_vel: (Optional) List of end velocities. Defaults to 0.
            start_acc: (Optional) List of start accelerations. Defaults to 0.
            end_acc: (Optional) List of end accelerations. Defaults to 0.

        """
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.duration = duration
        self.frequency = frequency
        self.num_joints = len(start_pos)

        if len(end_pos) != self.num_joints:
            raise ValueError("Start and end positions must have the same length.")

        self.start_vel = np.array(start_vel) if start_vel is not None else np.zeros(self.num_joints)
        self.end_vel = np.array(end_vel) if end_vel is not None else np.zeros(self.num_joints)
        self.start_acc = np.array(start_acc) if start_acc is not None else np.zeros(self.num_joints)
        self.end_acc = np.array(end_acc) if end_acc is not None else np.zeros(self.num_joints)

        self.time_steps = int(duration * frequency)
        self.time_array = np.linspace(0, duration, self.time_steps)

        # Pre-calculate coefficients
        self.coeffs = self._calculate_coefficients()

    def _calculate_coefficients(self) -> np.ndarray:
        """Calculates the coefficients a0, a1, a2, a3, a4, a5 for the polynomial:
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        T = self.duration
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        # System of equations for a quintic polynomial
        # q(0) = a0
        # v(0) = a1
        # a(0) = 2*a2
        # q(T) = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4 + a5*T^5
        # v(T) = a1 + 2*a2*T + 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4
        # a(T) = 2*a2 + 6*a3*T + 12*a4*T^2 + 20*a5*T^3

        coeffs = np.zeros((self.num_joints, 6))

        for i in range(self.num_joints):
            q0 = self.start_pos[i]
            v0 = self.start_vel[i]
            acc0 = self.start_acc[i]
            q1 = self.end_pos[i]
            v1 = self.end_vel[i]
            acc1 = self.end_acc[i]

            a0 = q0
            a1 = v0
            a2 = acc0 / 2.0

            # Solve for a3, a4, a5
            # A * [a3, a4, a5]^T = B
            A = np.array(
                [
                    [T3, T4, T5],
                    [3 * T2, 4 * T3, 5 * T4],
                    [6 * T, 12 * T2, 20 * T3],
                ],
            )

            B = np.array(
                [
                    q1 - (a0 + a1 * T + a2 * T2),
                    v1 - (a1 + 2 * a2 * T),
                    acc1 - (2 * a2),
                ],
            )

            x = np.linalg.solve(A, B)

            coeffs[i, 0] = a0
            coeffs[i, 1] = a1
            coeffs[i, 2] = a2
            coeffs[i, 3] = x[0]
            coeffs[i, 4] = x[1]
            coeffs[i, 5] = x[2]

        return coeffs

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates the trajectory.

        Returns:
            positions: (num_steps, num_joints)
            velocities: (num_steps, num_joints)
            accelerations: (num_steps, num_joints)
            time_array: (num_steps,)

        """
        positions = np.zeros((self.time_steps, self.num_joints))
        velocities = np.zeros((self.time_steps, self.num_joints))
        accelerations = np.zeros((self.time_steps, self.num_joints))

        for t_idx, t in enumerate(self.time_array):
            t2 = t * t
            t3 = t2 * t


class QuinticSplineTrajectory:
    """Generates a quintic (5th-order) polynomial trajectory for multiple joints.
    Ensures continuous position, velocity, and acceleration.
    """

    def __init__(
        self,
        start_pos: list[float],
        end_pos: list[float],
        duration: float,
        frequency: float,
        start_vel: list[float] | None = None,
        end_vel: list[float] | None = None,
        start_acc: list[float] | None = None,
        end_acc: list[float] | None = None,
    ):
        """Initialize the trajectory generator.

        Args:
            start_pos: List of start positions for each joint.
            end_pos: List of end positions for each joint.
            duration: Total duration of the trajectory in seconds.
            frequency: Sampling frequency in Hz.
            start_vel: (Optional) List of start velocities. Defaults to 0.
            end_vel: (Optional) List of end velocities. Defaults to 0.
            start_acc: (Optional) List of start accelerations. Defaults to 0.
            end_acc: (Optional) List of end accelerations. Defaults to 0.

        """
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.duration = duration
        self.frequency = frequency
        self.num_joints = len(start_pos)

        if len(end_pos) != self.num_joints:
            raise ValueError("Start and end positions must have the same length.")

        self.start_vel = np.array(start_vel) if start_vel is not None else np.zeros(self.num_joints)
        self.end_vel = np.array(end_vel) if end_vel is not None else np.zeros(self.num_joints)
        self.start_acc = np.array(start_acc) if start_acc is not None else np.zeros(self.num_joints)
        self.end_acc = np.array(end_acc) if end_acc is not None else np.zeros(self.num_joints)

        self.time_steps = int(duration * frequency)
        self.time_array = np.linspace(0, duration, self.time_steps)

        # Pre-calculate coefficients
        self.coeffs = self._calculate_coefficients()

    def _calculate_coefficients(self) -> np.ndarray:
        """Calculates the coefficients a0, a1, a2, a3, a4, a5 for the polynomial:
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        T = self.duration
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        # System of equations for a quintic polynomial
        # q(0) = a0
        # v(0) = a1
        # a(0) = 2*a2
        # q(T) = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4 + a5*T^5
        # v(T) = a1 + 2*a2*T + 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4
        # a(T) = 2*a2 + 6*a3*T + 12*a4*T^2 + 20*a5*T^3

        coeffs = np.zeros((self.num_joints, 6))

        for i in range(self.num_joints):
            q0 = self.start_pos[i]
            v0 = self.start_vel[i]
            acc0 = self.start_acc[i]
            q1 = self.end_pos[i]
            v1 = self.end_vel[i]
            acc1 = self.end_acc[i]

            a0 = q0
            a1 = v0
            a2 = acc0 / 2.0

            # Solve for a3, a4, a5
            # A * [a3, a4, a5]^T = B
            A = np.array(
                [
                    [T3, T4, T5],
                    [3 * T2, 4 * T3, 5 * T4],
                    [6 * T, 12 * T2, 20 * T3],
                ],
            )

            B = np.array(
                [
                    q1 - (a0 + a1 * T + a2 * T2),
                    v1 - (a1 + 2 * a2 * T),
                    acc1 - (2 * a2),
                ],
            )

            x = np.linalg.solve(A, B)

            coeffs[i, 0] = a0
            coeffs[i, 1] = a1
            coeffs[i, 2] = a2
            coeffs[i, 3] = x[0]
            coeffs[i, 4] = x[1]
            coeffs[i, 5] = x[2]

        return coeffs

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates the trajectory.

        Returns:
            positions: (num_steps, num_joints)
            velocities: (num_steps, num_joints)
            accelerations: (num_steps, num_joints)
            time_array: (num_steps,)

        """
        positions = np.zeros((self.time_steps, self.num_joints))
        velocities = np.zeros((self.time_steps, self.num_joints))
        accelerations = np.zeros((self.time_steps, self.num_joints))

        for t_idx, t in enumerate(self.time_array):
            t2 = t * t
            t3 = t2 * t
            t4 = t3 * t
            t5 = t4 * t

            for j in range(self.num_joints):
                a0, a1, a2, a3, a4, a5 = self.coeffs[j]

                positions[t_idx, j] = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
                velocities[t_idx, j] = a1 + 2 * a2 * t + 3 * a3 * t2 + 4 * a4 * t3 + 5 * a5 * t4
                accelerations[t_idx, j] = 2 * a2 + 6 * a3 * t + 12 * a4 * t2 + 20 * a5 * t3

        return positions, velocities, accelerations, self.time_array

    def save_to_json(self, filename: str) -> None:
        """Saves the trajectory to a JSON file.

        Args:
            filename: The path to the JSON file.
        """
        pos, vel, acc, time = self.generate()

        data = {
            "duration": self.duration,
            "frequency": self.frequency,
            "num_joints": self.num_joints,
            "time": time.tolist(),
            "positions": pos.tolist(),
            "velocities": vel.tolist(),
            "accelerations": acc.tolist(),
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Trajectory saved to {filename}")

    def plot(self, show: bool = True, save_path: str | None = None):
        """Visualizes the trajectory."""
        pos, vel, acc, time = self.generate()

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

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        if show:
            plt.show()


if __name__ == "__main__":
    # Example usage
    start_q = [0.0, 0.0, 0.0]
    end_q = [1.0, -0.5, 2.0]
    duration = 2.0
    freq = 100.0

    traj = QuinticSplineTrajectory(start_q, end_q, duration, freq)

    # Save to JSON
    output_path = "configurations/trajectories/spline_trajectory.json"
    traj.save_to_json(output_path)

    traj.plot()
=======
import numpy as np
import matplotlib.pyplot as plt
import os

class QuinticSpline:
    def __init__(self, duration: float, dt: float, start_pos: list, end_pos: list):
        """
        Initialize the Quintic Spline Trajectory Generator.

        Args:
            duration (float): Total duration of the trajectory in seconds.
            dt (float): Time step in seconds.
            start_pos (list): List of start positions for each joint.
            end_pos (list): List of end positions for each joint.
        """
        self.duration = duration
        self.dt = dt
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.num_joints = len(start_pos)
        
        if len(start_pos) != len(end_pos):
            raise ValueError("Start and end positions must have the same length.")

        self.time_array = np.arange(0, duration + dt, dt)
        
        # Coefficients for each joint: shape (num_joints, 6)
        self.coefficients = self._calculate_coefficients()

    def _calculate_coefficients(self):
        """
        Calculate the coefficients a0, a1, a2, a3, a4, a5 for the quintic polynomial:
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        
        Boundary conditions (start and end velocity/acceleration are 0):
        t=0: q=qs, v=0, a=0
        t=T: q=qe, v=0, a=0
        """
        T = self.duration
        qs = self.start_pos
        qe = self.end_pos
        
        # At t=0
        a0 = qs
        a1 = np.zeros(self.num_joints)
        a2 = np.zeros(self.num_joints)
        
        # At t=T, solving the system:
        # a3*T^3 + a4*T^4 + a5*T^5 = qe - qs
        # 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4 = 0
        # 6*a3*T + 12*a4*T^2 + 20*a5*T^3 = 0
        
        # This simplifies to:
        # a3 = 10*(qe - qs)/T^3
        # a4 = -15*(qe - qs)/T^4
        # a5 = 6*(qe - qs)/T^5
        
        delta_q = qe - qs
        a3 = 10 * delta_q / (T**3)
        a4 = -15 * delta_q / (T**4)
        a5 = 6 * delta_q / (T**5)
        
        # Stack coefficients: shape (6, num_joints) -> transpose to (num_joints, 6)
        return np.vstack([a0, a1, a2, a3, a4, a5]).T

    def generate(self):
        """
        Generate the trajectory.

        Returns:
            tuple: (time, positions, velocities, accelerations)
                time: (N,) array
                positions: (N, num_joints) array
                velocities: (N, num_joints) array
                accelerations: (N, num_joints) array
        """
        t = self.time_array
        N = len(t)
        
        positions = np.zeros((N, self.num_joints))
        velocities = np.zeros((N, self.num_joints))
        accelerations = np.zeros((N, self.num_joints))
        
        for i in range(self.num_joints):
            a0, a1, a2, a3, a4, a5 = self.coefficients[i]
            
            positions[:, i] = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
            velocities[:, i] = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
            accelerations[:, i] = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
            
        return t, positions, velocities, accelerations

    def plot(self, save_path="spline_trajectory.png"):
        """
        Plot and save the trajectory.
        """
        t, pos, vel, acc = self.generate()
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot Positions
        for i in range(self.num_joints):
            axes[0].plot(t, pos[:, i], label=f'Joint {i+1}')
        axes[0].set_ylabel('Position [rad] or [m]')
        axes[0].set_title('Quintic Spline Trajectory')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot Velocities
        for i in range(self.num_joints):
            axes[1].plot(t, vel[:, i], label=f'Joint {i+1}')
        axes[1].set_ylabel('Velocity [rad/s] or [m/s]')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot Accelerations
        for i in range(self.num_joints):
            axes[2].plot(t, acc[:, i], label=f'Joint {i+1}')
        axes[2].set_ylabel('Acceleration [rad/s^2] or [m/s^2]')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Trajectory plot saved to {os.path.abspath(save_path)}")
        plt.close()

if __name__ == "__main__":
    # Example usage / Test
    print("Testing QuinticSpline...")
    
    duration = 2.0
    dt = 0.01
    start_pos = [0.0, 0.0, 0.0]
    end_pos = [1.0, -0.5, 0.5]
    
    spline = QuinticSpline(duration, dt, start_pos, end_pos)
    
    # Generate and Plot
    spline.plot("spline_test_plot.png")
    
    print("Test Complete.")
>>>>>>> 1f4fd35 (initial commit of quintic spline)
