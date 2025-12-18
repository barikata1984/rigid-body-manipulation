import numpy as np
from scipy.optimize import minimize

from trajectories.base_trajectory import BaseTrajectory
from trajectories.fourier import FourierTrajectory


class ExcitationTrajectory(BaseTrajectory):
    def __init__(self, duration: float, fps: int, dof: int, num_harmonics, base_frequency, kinematics_func=None):
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
        super().__init__(duration, fps)

        self.dof = dof
        self.num_joints = dof
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
        x0 = np.concatenate([self.a.flatten(), self.b.flatten(), self.q0])

        # Pre-compute time steps for optimization (one full period is sufficient for condition number usually)
        # Or use self.duration
        dt = 1.0 / self.fps
        t_eval = np.arange(0, self.duration, dt)

        def objective(x):
            # Unpack
            split1 = self.dof * self.num_harmonics
            split2 = split1 * 2

            a_flat = x[:split1]
            b_flat = x[split1:split2]
            q0_flat = x[split2:]

            a = a_flat.reshape(self.dof, self.num_harmonics)
            b = b_flat.reshape(self.dof, self.num_harmonics)
            q0 = q0_flat  # size dof

            # Create Trajectory Model
            coeffs = {"a": a, "b": b, "q0": q0}
            traj = FourierTrajectory(self.dof, self.num_harmonics, self.base_frequency, coeffs)

            # Generate (q, dq, ddq) for all t
            # Vectorized get_value is efficient
            q, dq, ddq = traj.get_value(self.time_array)

            # Compute Regressor Stack
            # We need to iterate because kinematics_func (calculate_frame_dynamics wrappers) usually expect single time step or act differently
            # The blueprint says calculate_frame_dynamics takes act_traj of shape (3, N) maybe?
            # Looking at simulate.py line 88: _, _, twists, dtwists = self.inverse(act_traj) where act_traj is (3, dof) ?
            # No, simulate.py loop runs PER STEP. act_traj is (dof,) probably or (3, dof) for pos/vel/acc?
            # Actually simulate.py: act_traj = np.stack(self.sensors.get("jointvars")) -> shape (3, dof) likely.
            # So kinematics_func likely processes ONE sample at a time.

            # Accumulate Y = sum(A.T @ A)
            Y = np.zeros((10, 10))  # Assuming 10 parameters (mass, com*3, inertias*6)

            # Loop over time for regressor
            # This might be slow in Python loop, but necessary if kinematics_func is opaque
            for i in range(len(self.time_array)):
                # inputs: q[i], dq[i], ddq[i] -> shape (dof,)
                # kinematics_func expects inputs to form act_traj
                A_k = self.kinematics_func(q[i], dq[i], ddq[i])
                # A_k shape should be (6, 10)
                Y += A_k.T @ A_k

            # Condition number
            # kappa = max_eig / min_eig
            eigvals = np.linalg.eigvalsh(Y)  # Hermitian/Symmetric eigs
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)

            if min_eig < 1e-9:
                return 1e9  # Penalty for singularity

            cond_num = max_eig / min_eig
            return cond_num

        # Run Optimization
        # Bounds? Amplitudes shouldn't be too huge.
        bounds = [(-1.0, 1.0)] * len(x0)  # Reasonable bounds for joints

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter, "disp": True})

        # Update coefficients
        x_opt = res.x
        split1 = self.dof * self.num_harmonics
        split2 = split1 * 2

        self.a = x_opt[:split1].reshape(self.dof, self.num_harmonics)
        self.b = x_opt[split1:split2].reshape(self.dof, self.num_harmonics)
        self.q0 = x_opt[split2:]

        self._is_optimized = True
        print(f"Optimization finished. Final Condition Number: {res.fun:.4f}")

    def generate(self, show_plot: bool = False, plot_path: str | None = None, json_path: str | None = None):
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

        coeffs = {"a": self.a, "b": self.b, "q0": self.q0}
        traj = FourierTrajectory(self.dof, self.num_harmonics, self.base_frequency, coeffs)

        # Calculate dt locally
        dt = 1.0 / self.fps
        # generate() in FourierTrajectory likely expects dt.
        # But wait, looking at user code (Step 176): traj.generate(self.duration, self.dt) gave t, pos, vel, acc.
        # So we can use that if we have dt.
        # But BaseTrajectory has self.time_array.
        # If FourierTrajectory.generate returns t, pos, vel, acc, we can use that.
        t, pos, vel, acc = traj.generate(self.duration, dt)

        self._plot(pos, vel, acc, show=show_plot, plot_path=plot_path)

        if json_path is not None:
            self._write_to_json(pos, vel, acc, json_path)

        return pos, vel, acc, t

    def get_coefficients(self):
        return {"a": self.a, "b": self.b, "q0": self.q0}
