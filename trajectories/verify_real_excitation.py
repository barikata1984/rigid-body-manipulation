import os
import sys

from omegaconf import OmegaConf

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators.setup import generate_model_data
from dynamics import make_kinematics_func, setup_robot_dynamics_parameters
from simulators import SimulatorConfig
from trajectories.excitation import ExcitationTrajectory, ExcitationTrajectoryConfig


def main():
    print("=== Starting Real Physics Verification ===")

    # 1. Load Configuration
    config_path = "configurations/simulations/base.yaml"

    print(f"Loading config from {config_path}...")
    yaml_config = OmegaConf.load(config_path)
    base_config = SimulatorConfig()
    cfg = OmegaConf.merge(base_config, yaml_config)

    # Manually set object path as it is missing in base.yaml
    cfg.object = "xml_models/targets/hammer"
    print(f"Using object: {cfg.object}")

    # 2. Load Model
    print("Loading MuJoCo model...")
    import mujoco

    # generate_model_data returns m, d, gt
    m, d, gt = generate_model_data(cfg)
    print(f"Model loaded. DoF: {m.nu}")

    # Initialize kinematics to ensure d.xmat is valid
    print("Initializing kinematics (mj_forward)...")
    mujoco.mj_forward(m, d)

    # 3. Setup Dynamics Parameters
    print("Setting up dynamics parameters...")
    params = setup_robot_dynamics_parameters(m, d)

    # 4. Prepare Kinematics Closure
    pose_x_sen = params.poses.get_x_("site", "target/ft_sensor")
    kinematics_func = make_kinematics_func(params, pose_x_sen)

    # 5. Instantiate Excitation Trajectory
    dof = m.nu
    # 5. Instantiate Excitation Trajectory
    dof = m.nu
    print("Initializing ExcitationTrajectory...")
    cfg = ExcitationTrajectoryConfig(
        duration=2.0,
        num_joints=dof,
        num_harmonics=5,  # Keep complexity
        base_freq=1,  # Higher freq -> shorter duration (5s)
        fps=60,  # Coarser time step for speed (125 steps vs 1000)
    )
    exc = ExcitationTrajectory(
        cfg,
        kinematics_func=kinematics_func,
    )

    # 6. Run Optimization using BFGS
    print("Running Optimization (this uses REAL dynamics)...")
    # Increase iterations to allow convergence to a non-trivial shape
    exc.generate(
        show_plot=True, plot_path="real_excitation_verification.png", json_path="real_excitation_verification.json"
    )  # max_iter=10) # 10 iterations should be enough to show shape change

    # Check optimization result
    coeffs = exc.get_coefficients()
    print("Optimization finished.")
    print(f"Optimized Coeffs Sample (a[0]): {coeffs['a'][0]}")

    # 7. Generate Plot
    # output_plot = "real_excitation_verification.png"
    # exc.plot(output_plot)
    # print(f"Plot saved to {output_plot}")

    print("=== Verification Complete ===")


if __name__ == "__main__":
    main()
