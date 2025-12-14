import numpy as np

class FourierTrajectory:
    """
    Finite Fourier Series Trajectory.
    
    Represents the trajectory q_i(t) as:
    q_i(t) = q_{i,0} + sum_{k=1}^{N} ( a_{i,k} sin(2*pi*k*f*t) + b_{i,k} cos(2*pi*k*f*t) )
    
    Note: The paper uses rho for sine coeff and delta for cosine coeff.
    Here we use a generic name but map them consistently.
    a -> sine coefficients (rho in paper)
    b -> cosine coefficients (delta in paper)
    """
    def __init__(self, dof, num_harmonics, base_frequency, coefficients=None, q0=None):
        """
        Args:
            dof (int): Degrees of freedom.
            num_harmonics (int): Number of harmonics N.
            base_frequency (float): Fundamental frequency f_b [Hz].
            coefficients (dict, optional): Dictionary containing:
                - 'a': (dof, num_harmonics) array (Sine coeffs)
                - 'b': (dof, num_harmonics) array (Cosine coeffs)
                - 'q0': (dof,) array (Offset)
                If None, initializes with zeros.
            q0 (array, optional): Explicit offset if not in coefficients dictionary.
        """
        self.dof = dof
        self.num_harmonics = num_harmonics
        self.base_frequency = base_frequency
        self.omega_b = 2 * np.pi * base_frequency

        if coefficients is None:
            self.a = np.zeros((dof, num_harmonics))
            self.b = np.zeros((dof, num_harmonics))
            self.q0 = np.zeros(dof)
        else:
            self.a = np.array(coefficients.get('a', np.zeros((dof, num_harmonics))))
            self.b = np.array(coefficients.get('b', np.zeros((dof, num_harmonics))))
            if 'q0' in coefficients:
                self.q0 = np.array(coefficients['q0'])
            elif q0 is not None:
                self.q0 = np.array(q0)
            else:
                self.q0 = np.zeros(dof)

    def get_value(self, t):
        """
        Calculate q, dq, ddq at time t.
        
        Returns:
            q (dof,), dq (dof,), ddq (dof,)
        """
        q = np.copy(self.q0)
        dq = np.zeros(self.dof)
        ddq = np.zeros(self.dof)
        
        # Determine if t is scalar or array
        is_scalar = np.isscalar(t)
        if not is_scalar:
            # If array, expand for consistent operations
            # Output shapes: (N, dof)
            t = np.array(t)
            q = np.tile(self.q0, (len(t), 1))
            dq = np.zeros((len(t), self.dof))
            ddq = np.zeros((len(t), self.dof))

        for k in range(1, self.num_harmonics + 1):
            omega_k = k * self.omega_b
            
            # Coefficients for k-th harmonic (0-indexed in array)
            idx = k - 1
            a_k = self.a[:, idx] # Sine coeffs
            b_k = self.b[:, idx] # Cosine coeffs
            
            wkt = omega_k * t
            sin_wkt = np.sin(wkt)
            cos_wkt = np.cos(wkt)
            
            if not is_scalar:
                # Reshape for broadcasting
                # a_k: (dof,) -> (1, dof)
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

    def generate(self, duration, dt):
        """
        Generate trajectory arrays.
        
        Args:
            duration (float): Total duration.
            dt (float): Time step.
            
        Returns:
            t (N,): Time array
            pos (N, dof): Position array
            vel (N, dof): Velocity array
            acc (N, dof): Acceleration array
        """
        t = np.arange(0, duration, dt)
        # remove last element if it exceeds duration significantly (standard arange behavior check)
        if t[-1] > duration + 1e-9:
             t = t[:-1]
             
        pos, vel, acc = self.get_value(t)
        
        return t, pos, vel, acc
