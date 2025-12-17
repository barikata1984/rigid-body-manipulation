import os
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from trajectories.fourier_trajectory import FourierTrajectory

class ExcitationTrajectory:
    def __init__(self, duration:float, fps: int, dof: int, num_harmonics, base_frequency, kinematics_func=None):
        """
        Excitation Trajectory Generator using Finite Fourier Series Optimization.

        Args:
            dof (int): Degrees of Freedom.
            num_harmonics (int): Number of harmonics N.
            base_frequency (float): Fundamental frequency [Hz].
            dt (float): Time step for generation and optimization.
            duration (float, optional): Duration of trajectory. Defaults to 1 period (1/base_frequency).
            kinematics_func (callable, optional): function f(q, dq, ddq) -> regressor_matrix.
                Required for optimization.
        """
        self.duration = duration
        self.fps = fps
        self.dt = 1.0 / self.fps
        
        self.dof = dof
        self.num_harmonics = num_harmonics
        self.base_frequency = base_frequency
            
        self.kinematics_func = kinematics_func
        
        # Internal state
        self._is_optimized = False
        
        # Initialize coefficients with something (random small noise or zeros)
        # Using random to avoid stuck at zero gradient if initial guess matters
        # Flat vector size: dof * num_harmonics * 2 (a and b) + dof (q0) -> actually q0 often can be 0 or mid range
        # Let's keep separate storage for convenience, but for optimize we flatten.
        self.a = np.random.uniform(-0.1, 0.1, (dof, num_harmonics))
        self.b = np.random.uniform(-0.1, 0.1, (dof, num_harmonics))
        self.q0 = np.zeros(dof)

    def _optimize(self, max_iter=100):
        """
        Optimize the trajectory coefficients to minimize the condition number of the regressor matrix.
        """
        if self.kinematics_func is None:
            raise ValueError("kinematics_func must be provided for optimization.")
            
        print(f"Starting optimization (dof={self.dof}, N={self.num_harmonics}, f={self.base_frequency})...")

        # Initial guess (flattened)
        # [a_flat, b_flat, q0]
        x0 = np.concatenate([
            self.a.flatten(),
            self.b.flatten(),
            self.q0
        ])
        
        # Pre-compute time steps for optimization (one full period is sufficient for condition number usually)
        # Or use self.duration
        t_eval = np.arange(0, self.duration, self.dt)
        
        def objective(x):
            # Unpack
            split1 = self.dof * self.num_harmonics
            split2 = split1 * 2
            
            a_flat = x[:split1]
            b_flat = x[split1:split2]
            q0_flat = x[split2:]
            
            a = a_flat.reshape(self.dof, self.num_harmonics)
            b = b_flat.reshape(self.dof, self.num_harmonics)
            q0 = q0_flat # size dof
            
            # Create Trajectory Model
            coeffs = {'a': a, 'b': b, 'q0': q0}
            traj = FourierTrajectory(self.dof, self.num_harmonics, self.base_frequency, coeffs)
            
            # Generate (q, dq, ddq) for all t
            # Vectorized get_value is efficient
            q, dq, ddq = traj.get_value(t_eval)
            
            # Compute Regressor Stack
            # We need to iterate because kinematics_func (calculate_frame_dynamics wrappers) usually expect single time step or act differently
            # The blueprint says calculate_frame_dynamics takes act_traj of shape (3, N) maybe?
            # Looking at simulate.py line 88: _, _, twists, dtwists = self.inverse(act_traj) where act_traj is (3, dof) ?
            # No, simulate.py loop runs PER STEP. act_traj is (dof,) probably or (3, dof) for pos/vel/acc?
            # Actually simulate.py: act_traj = np.stack(self.sensors.get("jointvars")) -> shape (3, dof) likely.
            # So kinematics_func likely processes ONE sample at a time.
            
            # Accumulate Y = sum(A.T @ A)
            Y = np.zeros((10, 10)) # Assuming 10 parameters (mass, com*3, inertias*6)
            
            # Loop over time for regressor
            # This might be slow in Python loop, but necessary if kinematics_func is opaque
            for i in range(len(t_eval)):
                # inputs: q[i], dq[i], ddq[i] -> shape (dof,)
                # kinematics_func expects inputs to form act_traj
                A_k = self.kinematics_func(q[i], dq[i], ddq[i])
                # A_k shape should be (6, 10)
                Y += A_k.T @ A_k
            
            # Condition number
            # kappa = max_eig / min_eig
            eigvals = np.linalg.eigvalsh(Y) # Hermitian/Symmetric eigs
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)
            
            if min_eig < 1e-9:
                return 1e9 # Penalty for singularity
                
            cond_num = max_eig / min_eig
            return cond_num

        # Run Optimization
        # Bounds? Amplitudes shouldn't be too huge.
        bounds = [(-1.0, 1.0)] * len(x0) # Reasonable bounds for joints
        
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'disp': True})
        
        # Update coefficients
        x_opt = res.x
        split1 = self.dof * self.num_harmonics
        split2 = split1 * 2
        
        self.a = x_opt[:split1].reshape(self.dof, self.num_harmonics)
        self.b = x_opt[split1:split2].reshape(self.dof, self.num_harmonics)
        self.q0 = x_opt[split2:]
        
        self._is_optimized = True
        print(f"Optimization finished. Final Condition Number: {res.fun:.4f}")

    def generate(self):
        """
        Generate trajectory. Lazily optimizes if not done.
        """
        if not self._is_optimized:
             # If kinematics_func is provided, we can optimize.
             # If not, we just warn and generate with initial random coeffs?
             if self.kinematics_func:
                 print("Optimization not yet performed. Running optimize() now...")
                 self._optimize(10)
             else:
                 print("Warning: generating without optimization (kinematics_func missing).")
        
        coeffs = {'a': self.a, 'b': self.b, 'q0': self.q0}
        traj = FourierTrajectory(self.dof, self.num_harmonics, self.base_frequency, coeffs)
        t, pos, vel, acc = traj.generate(self.duration, self.dt)
        
        #return traj.generate(self.duration, self.dt)
        self.plot(t, pos, vel, acc)
        self.write_to_json(t, pos, vel, acc)

        return # t, pos, vel, acc

    def get_coefficients(self):
        return {'a': self.a, 'b': self.b, 'q0': self.q0}

    def _plot(self, save_path="excitation_trajectory.png"):
        """
        Plot and save the trajectory.
        """
        t, pos, vel, acc = self.generate()
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot Positions
        for i in range(self.dof):
            axes[0].plot(t, pos[:, i], label=f'Joint {i+1}')
        axes[0].set_ylabel('Position [rad]')
        axes[0].set_title('Excitation Trajectory')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot Velocities
        for i in range(self.dof):
            axes[1].plot(t, vel[:, i], label=f'Joint {i+1}')
        axes[1].set_ylabel('Velocity [rad/s]')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot Accelerations
        for i in range(self.dof):
            axes[2].plot(t, acc[:, i], label=f'Joint {i+1}')
        axes[2].set_ylabel('Acceleration [rad/s^2]')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Excitation plot saved to {os.path.abspath(save_path)}")
        plt.close()

    def plot(self, t, pos, vel, acc, save_path="spline_trajectory.png", show=True):
        """
        Plot and save the trajectory.

        Args:
            save_path (str): Path to save the plot image.
            show (bool): If True, display the plot window.
        """        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        import pdb

        pdb.set_trace()

        # Plot Positions
        for i in range(self.dof):
            axes[0].plot(t, pos[:, i], label=f'Joint {i+1}')
        axes[0].set_ylabel('Position [rad] or [m]')
        axes[0].set_title('Quintic Spline Trajectory')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot Velocities
        for i in range(self.dof):
            axes[1].plot(t, vel[:, i], label=f'Joint {i+1}')
        axes[1].set_ylabel('Velocity [rad/s] or [m/s]')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot Accelerations
        for i in range(self.dof):
            axes[2].plot(t, acc[:, i], label=f'Joint {i+1}')
        axes[2].set_ylabel('Acceleration [rad/s^2] or [m/s^2]')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Trajectory plot saved to {os.path.abspath(save_path)}")
        
        if show:
            plt.show()
            
        plt.close()

    def write_to_json(self, t, pos, vel, acc, save_path="spline_trajectory.json"):
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
        for i in range(len(t)):
            frame = {
                "qpos": pos[i].tolist(),
                "qvel": vel[i].tolist(),
                "qacc": acc[i].tolist()
            }
            frames.append(frame)
            
        data = {
            "duration": self.duration,
            "fps": 1.0 / self.dt if self.dt > 0 else 0.0,
            "frames": frames
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Trajectory JSON saved to {os.path.abspath(save_path)}")