
import numpy as np
from scipy.optimize import minimize

from dataclasses import dataclass

from trajectories.base_trajectory import BaseTrajectory, BaseTrajectoryConfig
from trajectories.fourier import FourierTrajectory, FourierTrajectoryConfig


@dataclass
class ExcitationTrajectoryConfig(BaseTrajectoryConfig):
    num_joints: int = 1
    num_harmonics: int = 5
    base_freq: float = 0.1
    target_class: str = "ExcitationTrajectory"


class ExcitationTrajectory(BaseTrajectory):
    def __init__(self, cfg: ExcitationTrajectoryConfig, *args, **kwargs):
        """
        Excitation Trajectory Generator using Finite Fourier Series Optimization.

        Args:
            cfg: Configuration object.
            kinematics_func (callable, optional): function f(q, dq, ddq) -> regressor_matrix.
                Required for optimization.
        """
        super().__init__(cfg, *args, **kwargs)

        self.num_joints = cfg.num_joints
        self.num_harmonics = cfg.num_harmonics
        self.base_freq = cfg.base_freq

        self.kinematics_func = kwargs.get("kinematics_func", None)

        # Internal state
        self._is_optimized = False

        # Initialize coefficients with something (random small noise or zeros)
        # Using random to avoid stuck at zero gradient if initial guess matters
        # Flat vector size: dof * num_harmonics * 2 (a and b) + dof (q0) -> actually q0 often can be 0 or mid range
        # Let's keep separate storage for convenience, but for optimize we flatten.
        self.a = np.random.uniform(-0.1, 0.1, (self.num_joints, self.num_harmonics))
        self.b = np.random.uniform(-0.1, 0.1, (self.num_joints, self.num_harmonics))
        self.q0 = np.zeros(self.num_joints)

    def _optimize(self, max_iter=100):
        """
        Optimize the trajectory coefficients to minimize the condition number of the regressor matrix.
        """
        if self.kinematics_func is None:
            raise ValueError("kinematics_func must be provided for optimization.")

        print(
            f"Starting optimization (num_joints={self.num_joints}, num_harmonics={self.num_harmonics}, base_freq={self.base_freq})..."
        )

        # Initial guess (flattened)
        # [a_flat, b_flat, q0]
        x0 = np.concatenate([self.a.flatten(), self.b.flatten(), self.q0])

        # Track iteration and condition number
        it_count = 0
        current_cond = float("inf")

        def objective(x):
            nonlocal current_cond
            # Unpack
            split1 = self.num_joints * self.num_harmonics
            split2 = split1 * 2

            a_flat = x[:split1]
            b_flat = x[split1:split2]
            q0_flat = x[split2:]

            a = a_flat.reshape(self.num_joints, self.num_harmonics)
            b = b_flat.reshape(self.num_joints, self.num_harmonics)
            q0 = q0_flat

            # Create Trajectory Model
            coeffs = {"a": a, "b": b, "q0": q0}
            f_cfg = FourierTrajectoryConfig(
                duration=self.duration,
                fps=self.fps,
                num_joints=self.num_joints,
                num_harmonics=self.num_harmonics,
                base_freq=self.base_freq,
                coefficients=coeffs,
            )
            traj = FourierTrajectory(f_cfg)

            q, dq, ddq = traj.get_value()

            Y = np.zeros((10, 10))
            for i in range(len(self.time_array)):
                A_k = self.kinematics_func(q[i], dq[i], ddq[i])
                Y += A_k.T @ A_k

            eigvals = np.linalg.eigvalsh(Y)
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)

            if min_eig < 1e-9:
                cond_num = 1e9
            else:
                cond_num = max_eig / min_eig

            current_cond = cond_num
            return cond_num

        def callback(xk):
            nonlocal it_count
            it_count += 1
            print(f"Iteration {it_count}: Condition Number = {current_cond:.4f}")

        # Run Optimization
        bounds = [(-1.0, 1.0)] * len(x0)

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={"maxiter": max_iter, "disp": True},
        )

        # Update coefficients
        x_opt = res.x
        split1 = self.num_joints * self.num_harmonics
        split2 = split1 * 2

        self.a = x_opt[:split1].reshape(self.num_joints, self.num_harmonics)
        self.b = x_opt[split1:split2].reshape(self.num_joints, self.num_harmonics)
        self.q0 = x_opt[split2:]

        self._is_optimized = True
        print(f"Optimization finished. Final Condition Number: {res.fun:.4f}")

    def _generate(self, *args, **kwargs):
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
        f_cfg = FourierTrajectoryConfig(
            duration=self.duration,
            fps=self.fps,
            num_joints=self.num_joints,
            num_harmonics=self.num_harmonics,
            base_freq=self.base_freq,
            coefficients=coeffs,
        )
        traj = FourierTrajectory(f_cfg)

        pos, vel, acc = traj.generate()

        return pos, vel, acc

    def get_coefficients(self):
        return {"a": self.a, "b": self.b, "q0": self.q0}
