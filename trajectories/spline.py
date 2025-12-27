import numpy as np
from base_trajectory import BaseTrajectory


class QuinticSplineTrajectory(BaseTrajectory):
    """Generates a quintic (5th-order) polynomial trajectory for multiple joints.
    Ensures continuous position, velocity, and acceleration.
    """

    def __init__(
        self,
        duration: float,
        fps: float,
        start_pos: list[float],
        end_pos: list[float],
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
        super().__init__(duration, fps)

        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.num_joints = len(start_pos)

        if len(end_pos) != self.num_joints:
            raise ValueError("Start and end positions must have the same length.")

        self.start_vel = np.array(start_vel) if start_vel is not None else np.zeros(self.num_joints)
        self.end_vel = np.array(end_vel) if end_vel is not None else np.zeros(self.num_joints)
        self.start_acc = np.array(start_acc) if start_acc is not None else np.zeros(self.num_joints)
        self.end_acc = np.array(end_acc) if end_acc is not None else np.zeros(self.num_joints)

        self.time_steps = int(self.duration * self.fps)
        self.time_array = np.linspace(0, self.duration, self.time_steps)

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

    def generate(self, show_plot: bool = False, plot_path: str | None = None, json_path: str | None = None):
        """Generates the trajectory.

        Returns:
            positions: (num_steps, num_joints)
            velocities: (num_steps, num_joints)
            accelerations: (num_steps, num_joints)
            time_array: (num_steps,)

        """
        pos = np.zeros((self.time_steps, self.num_joints))
        vel = np.zeros((self.time_steps, self.num_joints))
        acc = np.zeros((self.time_steps, self.num_joints))

        for t_idx, t in enumerate(self.time_array):
            t2 = t * t
            t3 = t2 * t
            t4 = t3 * t
            t5 = t4 * t

            for j in range(self.num_joints):
                a0, a1, a2, a3, a4, a5 = self.coeffs[j]

                pos[t_idx, j] = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
                vel[t_idx, j] = a1 + 2 * a2 * t + 3 * a3 * t2 + 4 * a4 * t3 + 5 * a5 * t4
                acc[t_idx, j] = 2 * a2 + 6 * a3 * t + 12 * a4 * t2 + 20 * a5 * t3

        self.plot(pos, vel, acc, show=show_plot, plot_path=plot_path)

        if json_path is not None:
            self.write_to_json(pos, vel, acc, json_path)

        return pos, vel, acc


if __name__ == "__main__":
    # Example usage
    start_q = [0.0, 0.0, 0.0]
    end_q = [1.0, -0.5, 2.0]
    duration = 2.0
    fps = 100.0

    traj = QuinticSplineTrajectory(duration, fps, start_q, end_q)
    traj.generate(show_plot=True, plot_path="spline.png", json_path="spline.json")
