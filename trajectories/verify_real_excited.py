import os
import sys

import numpy as np
from omegaconf import OmegaConf

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamics import calculate_frame_dynamics, setup_robot_dynamics_parameters
from simulators import SimulatorConfig
from simulators.setup import generate_model_data
from trajectories.excited import ExcitedTrajectory, ExcitedTrajectoryConfig
from trajectories.spline import SplineTrajectoryConfig


def main():
    print("=== Starting Real Physics Verification for ExcitedTrajectory ===")

    # 1. Load Configuration
    config_path = "configurations/simulations/base.yaml"
    print(f"Loading config from {config_path}...")
    yaml_config = OmegaConf.load(config_path)
    base_config = SimulatorConfig()
    cfg = OmegaConf.merge(base_config, yaml_config)
    cfg.object = "xml_models/targets/hammer"

    # 2. Load Model
    print("Loading MuJoCo model...")
    import mujoco

    m, d, gt = generate_model_data(cfg)
    print(f"Model loaded. DoF: {m.nu}")

    mujoco.mj_forward(m, d)

    # 3. Setup Dynamics Parameters
    print("Setting up dynamics parameters...")
    (poses, id_ll, pose_ll_llj, uscrews_lj, simats_lj_l, hposes_lj_kj, inverse_dynamics_func) = (
        setup_robot_dynamics_parameters(m, d)
    )

    # 4. Prepare Kinematics Closure
    pose_x_ll = poses.x_b[id_ll]
    pose_x_sen = poses.get_x_("site", "target/ft_sensor")

    def kinematics_func(q, dq, ddq):
        act_traj = np.stack([q, dq, ddq])
        _, _, regressor = calculate_frame_dynamics(
            act_traj, inverse_dynamics_func, id_ll, pose_x_ll, pose_ll_llj, pose_x_sen
        )
        return regressor

    # 5. Define Base Trajectory (Spline)
    # Move from initial configuration to a slightly offset configuration
    start_q = np.array(d.qpos[: m.nu])
    # Create a target within joint limits (assuming limits are reasonable, e.g. -pi to pi)
    # Just move joint 1 by 0.5 rad, joint 3 by -0.3 rad, etc.
    end_q = start_q.copy()
    end_q[0] += 0.5
    end_q[1] -= 0.3
    end_q[2] += 0.4

    print(f"Base Trajectory: {start_q} -> {end_q}")

    duration = 2.0
    fps = 60
    spline_cfg = SplineTrajectoryConfig(
        duration=duration,
        fps=fps,
        type="quintic",
        start_pos=start_q,
        end_pos=end_q,
    )
    # spline = QuinticSplineTrajectory(spline_cfg) # We don't need to instantiate it manually if passing config to Excited

    # 6. Instantiate ExcitedTrajectory
    print("Initializing ExcitedTrajectory...")
    excited_cfg = ExcitedTrajectoryConfig(
        main_trajectory=spline_cfg,  # Pass config, not instance
        num_harmonics=5,
        base_freq=0.5,  # 1 cycle in 2.0s
    )

    excited_traj = ExcitedTrajectory(
        excited_cfg,
        kinematics_func=kinematics_func,
    )

    # 7. Run Optimization
    print("Running Optimization (this uses REAL dynamics)...")
    # This will trigger _optimize internally
    excited_traj.generate(
        max_iter=10,
        show_plot=True,
        plot_path="real_excited_verification.png",
        json_path="real_excited_verification.json",
    )

    # 8. Report
    print("=== Verification Complete ===")


if __name__ == "__main__":
    main()
