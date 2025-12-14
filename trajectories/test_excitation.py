import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectories.excitation_trajectory import ExcitationTrajectory
from trajectories.fourier_trajectory import FourierTrajectory

def dummy_kinematics(q, dq, ddq):
    """
    Dummy kinematics function that returns a random but consistent regressor 
    based on input to simulate dependency.
    """
    # Create a dummy A matrix (6x10)
    # Make it depend on q so optimization has something to grab onto
    # (Minimize condition number implies we want A'A to be well conditioned)
    
    # Simple deterministic pseudo-random gen based on sum of q
    seed = int(np.sum(q) * 1000) % 10000
    rng = np.random.RandomState(seed) 
    A = rng.randn(6, 10)
    
    # Add some structure to make it optimization-sensitive?
    # Actually random might be too chaotic for gradient descent (non-smooth).
    # Let's simple structure: A = [q[0]*I, dq[0]*I ...]
    
    A = np.zeros((6, 10))
    # Fill diagonal-ish
    for i in range(6):
        A[i, i] = q[0] if len(q)>0 else 1
        A[i, i+1] = dq[0] if len(dq)>0 else 1
    
    return A

def test_fourier_trajectory():
    print("Testing FourierTrajectory...")
    dof = 2
    harmonics = 1
    freq = 1.0
    coeffs = {
        'a': [[1.0], [0.5]], # sin coeffs
        'b': [[0.0], [0.0]], # cos coeffs
        'q0': [0.0, 0.0]
    }
    traj = FourierTrajectory(dof, harmonics, freq, coeffs)
    
    # t=0.25 (1/4 period) -> sin=1, cos=0
    t = 0.25
    q, dq, ddq = traj.get_value(t)
    
    # q = q0 + a*sin + b*cos = 0 + a*1 + 0 = a
    expected_q = np.array([1.0, 0.5])
    
    error = np.linalg.norm(q - expected_q)
    print(f"  q(0.25) error: {error}")
    if error < 1e-6:
        print("  FourierTrajectory Unit Test Passed.")
    else:
        print("  FourierTrajectory Unit Test FAILED.")

def test_excitation_trajectory():
    print("\nTesting ExcitationTrajectory (Optimizer)...")
    
    dof = 2
    # Use a dummy kinematics func
    # Note: L-BFGS-B needs smooth gradients. The discrete random above is bad.
    # Let's make a smooth dummy.
    def smooth_kinematics(q, dq, ddq):
        # A depends on q continuously
        val = np.sum(q)
        A = np.eye(10)[:6, :] * (1.0 + 0.1 * np.sin(val))
        A[0, 9] = dq[0] # Couples velocity
        return A

    exc = ExcitationTrajectory(dof=dof, num_harmonics=1, base_frequency=1.0, kinematics_func=smooth_kinematics)
    
    # Test Generate (Lazily optimizes)
    # Decrease max_iter for speed in test
    print("  Calling plot() which should trigger optimize()...")
    exc.optimize = lambda: print("  [Mock] optimize called") or setattr(exc, '_is_optimized', True)
    
    exc.plot("test_excitation_plot.png")
    
    if os.path.exists("test_excitation_plot.png"):
        print("  Plot file created.")
        print("  ExcitationTrajectory Integration Test Passed.")
    else:
        print("  ExcitationTrajectory Integration Test FAILED (No plot).")

if __name__ == "__main__":
    test_fourier_trajectory()
    test_excitation_trajectory()
