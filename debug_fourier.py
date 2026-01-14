from trajectories.fourier import FourierTrajectoryConfig
from trajectories.windowed_fourier import WindowedFourierTrajectory, WindowedFourierTrajectoryConfig


def debug_trajectory():
    # Simulate the config structure from YAML
    # J1: a=[1.0, 0.5], b=[0.5, 0.2]
    # J2: a=[0.8, 0.4], b=[0.4, 0.1]
    # ...
    coeffs_a = [[1.0, 0.5], [0.8, 0.4], [0.6, 0.3], [0.4, 0.2], [0.2, 0.1], [0.1, 0.05]]
    coeffs_b = [[0.5, 0.2], [0.4, 0.1], [0.3, 0.1], [0.2, 0.1], [0.1, 0.0], [0.05, 0.0]]

    fourier_cfg = FourierTrajectoryConfig(
        num_joints=6,
        num_harmonics=2,
        base_freq=0.5,
        coefficients={"a": coeffs_a, "b": coeffs_b},
        q0=[0.0] * 6,
        duration=5.0,
        fps=20.0,
    )

    cfg = WindowedFourierTrajectoryConfig(fourier_trajectory=fourier_cfg, duration=5.0, fps=20.0)

    traj = WindowedFourierTrajectory(cfg)

    print("--- Loaded Coefficients ---")
    print("a:\n", traj.fourier_trajectory.a)
    print("b:\n", traj.fourier_trajectory.b)

    # Generate trajectory
    pos, vel, acc = traj.get_value()

    print("\n--- Trajectory Values (t=1.0s, index=20) ---")
    idx = 20
    print(f"Time: {traj.time_array[idx]}")
    print("Pos:", pos[idx])

    # Check if they are just scaled versions
    # Compare J1 and J2
    ratio_j1_j2 = pos[idx][0] / pos[idx][1]
    print(f"Ratio J1/J2 at t=1.0s: {ratio_j1_j2}")

    print("\n--- Trajectory Values (t=2.5s, index=50) ---")
    idx = 50
    print(f"Time: {traj.time_array[idx]}")
    print("Pos:", pos[idx])
    ratio_j1_j2_2 = pos[idx][0] / pos[idx][1]
    print(f"Ratio J1/J2 at t=2.5s: {ratio_j1_j2_2}")

    if abs(ratio_j1_j2 - ratio_j1_j2_2) < 1e-4:
        print("\nWARNING: Ratio seems constant across time! This implies identical waveform shape.")
    else:
        print("\nRatio changes over time, meaning waveforms are different.")


if __name__ == "__main__":
    debug_trajectory()
