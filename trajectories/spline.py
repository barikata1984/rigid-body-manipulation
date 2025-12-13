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
