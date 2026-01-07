import numpy as np

from .base_trajectory import BaseTrajectory


class FourierTrajectory(BaseTrajectory):
    """
    Finite Fourier Series Trajectory.
    
    Represents the trajectory q_i(t) as:
    q_i(t) = q_{i,0} + sum_{k=1}^{N} ( a_{i,k} sin(2*pi*k*f*t) + b_{i,k} cos(2*pi*k*f*t) )
    
    Note: The paper uses rho for sine coeff and delta for cosine coeff.
    Here we use a generic name but map them consistently.
    a -> sine coefficients (rho in paper)
    b -> cosine coefficients (delta in paper)
    """
    def __init__(
            self,
            duration: float,
            fps:int, 
            num_joints: int, 
            num_harmonics: int, 
            base_freq: float, 
            coefficients=None, 
            q0=None,
            ):
        """
        Args:
            num_joints (int): Number of joints.
            num_harmonics (int): Number of harmonics N.
            base_frequency (float): Fundamental frequency f_b [Hz].
            coefficients (dict, optional): Dictionary containing:
                - 'a': (num_joints, num_harmonics) array (Sine coeffs)
                - 'b': (num_joints, num_harmonics) array (Cosine coeffs)
                - 'q0': (num_joints,) array (Offset)
                If None, initializes with zeros.
            q0 (array, optional): Explicit offset if not in coefficients dictionary.
        """
        super().__init__(duration, fps)

        self.num_joints = num_joints
        self.num_harmonics = num_harmonics
        self.base_freq = base_freq
        self.omega_b = 2 * np.pi * base_freq

        if coefficients is None:
            self.a = np.zeros((num_joints, num_harmonics))
            self.b = np.zeros((num_joints, num_harmonics))
            self.q0 = np.zeros(num_joints)
        else:
            self.a = np.array(coefficients.get('a', np.zeros((num_joints, num_harmonics))))
            self.b = np.array(coefficients.get('b', np.zeros((num_joints, num_harmonics))))
            if 'q0' in coefficients:
                self.q0 = np.array(coefficients['q0'])
            elif q0 is not None:
                self.q0 = np.array(q0)
            else:
                self.q0 = np.zeros(num_joints)

    def get_value(self):
        """
        Calculate q, dq, ddq at time t.
        
        Returns:
            q (num_joints,), dq (num_joints,), ddq (num_joints,)
        """
        q = np.copy(self.q0)
        dq = np.zeros(self.num_joints)
        ddq = np.zeros(self.num_joints)
        
        # Determine if t is scalar or array
        is_scalar = np.isscalar(self.time_array)
        if not is_scalar:
            # If array, expand for consistent operations
            # Output shapes: (N, num_joints)
            t = np.array(self.time_array)
            q = np.tile(self.q0, (len(t), 1))
            dq = np.zeros((len(t), self.num_joints))
            ddq = np.zeros((len(t), self.num_joints))

        for k in range(1, self.num_harmonics + 1):
            omega_k = k * self.omega_b
            
            # Coefficients for k-th harmonic (0-indexed in array)
            idx = k - 1
            a_k = self.a[:, idx] # Sine coeffs
            b_k = self.b[:, idx] # Cosine coeffs
            
            wkt = omega_k * self.time_array
            sin_wkt = np.sin(wkt)
            cos_wkt = np.cos(wkt)
            
            if not is_scalar:
                # Reshape for broadcasting
                # a_k: (num_joints,) -> (1, num_joints)
                a_k = a_k.reshape(1, -1)
                b_k = b_k.reshape(1, -1)
                sin_wkt = sin_wkt.reshape(-1, 1)
                cos_wkt = cos_wkt.reshape(-1, 1)

            # Position
            # q += a*sin + b*cos
            q += a_k * sin_wkt + b_k * cos_wkt
            
            # Velocity
            # dq/dt = a*w*cos - b*w*sin
            dq += omega_k * (a_k * cos_wkt - b_k * sin_wkt)
            
            # Acceleration
            # d2q/dt2 = -a*w^2*sin - b*w^2*cos = -w^2 * (term)
            ddq += -(omega_k**2) * (a_k * sin_wkt + b_k * cos_wkt)
            
        return q, dq, ddq

    def generate(self, show_plot: bool = False, plot_path: str | None = None, json_path: str | None = None):
        """
        Generate trajectory arrays.
        
        Args:
            duration (float): Total duration.
            dt (float): Time step.
            
        Returns:
            t (N,): Time array
            pos (N, num_joints): Position array
            vel (N, num_joints): Velocity array
            acc (N, num_joints): Acceleration array
        """
        # remove last element if it exceeds duration significantly (standard arange behavior check)
        if self.time_array[-1] > self.duration + 1e-9:
             self.time_array = self.time_array[:-1]
             
        pos, vel, acc = self.get_value()

        self.plot(pos, vel, acc, show=show_plot, plot_path=plot_path)

        if json_path is not None:
            self.write_to_json(pos, vel, acc, json_path)
        
        return pos, vel, acc
