"""
Main simulation script using Hydra.

Usage:
    # Run with base config
    python main.py object=xml_models/objects/box reset_keyframe=initial_state

    # Override specific settings
    python main.py object=path/to/object target_trajectory=path/to/traj.json

    # Use a different config file
    python main.py --config-name=my_config
"""
import json
from pathlib import Path
from shutil import copy

import hydra
import numpy as np
import pandas as pd
from numpy.linalg import lstsq, norm
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from factory import instantiate
from regressions import total_lstsq
from simulators import SimulatorConfig
from simulators.setup import generate_model_data, get_element_id
from utilities import json_to_namespace


# Get absolute path to config directory
_CONF_DIR = str(Path(__file__).parent / "configurations" / "simulations")


@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main simulation entry point using Hydra."""
    print(OmegaConf.to_yaml(cfg))

    m, d, gt = generate_model_data(cfg)

    # Load trajectory and extract excitation indices if available
    excitation_slice = slice(None)  # Default to full trajectory
    target_trajectory = None
    if cfg.get("target_trajectory"):
        with open(cfg.target_trajectory) as f:
            trajectory_data = json.load(f)
        target_trajectory = json_to_namespace(trajectory_data)

    # Fill the recorder's fps with the target_trajectory's
    try:
        cfg.recorder.fps = int(cfg.recorder.fps)
    except (ConfigAttributeError, TypeError):
        if target_trajectory:
            cfg.recorder.fps = int(target_trajectory.fps)

    # Fill (potentially) missing fields of the recorder configuration
    try:
        cfg.recorder.aabb_scale = float(cfg.recorder.aabb_scale)
    except (ConfigAttributeError, TypeError):
        aabb_scale = m.numeric_data[get_element_id(m, "numeric", "target/aabb_scale")]
        cfg.recorder.aabb_scale = float(aabb_scale)

    # Get/set and make the dataset dir
    object_dir = Path(cfg.object)
    try:
        dataset_dir = Path(cfg.recorder.dataset_dir)
    except (ConfigAttributeError, TypeError):
        dataset_dir = Path.cwd() / "datasets" / object_dir.name
        cfg.recorder.dataset_dir = str(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy the ground truth mass distribution file to the dataset file
    object_gt = object_dir / "ground_truth.csv"
    dataset_gt = dataset_dir / "ground_truth.csv"
    if dataset_gt.is_file():
        print("'ground_truth.csv' is not copied to the dataset dir since the file with the same name already exists.")
    else:
        copy(object_gt, dataset_gt)

    # Convert Hydra config to structured config object for instantiation
    simulator_cfg = OmegaConf.to_object(cfg)
    simulation = instantiate(simulator_cfg, model=m, data=d, target_trajectory=target_trajectory)
    result = simulation.run()

    # Show inertial params identified with the least squares method
    gt_total_mass = gt["mass"]
    gt_f_moms = gt_total_mass * gt["com"]  # type: ignore
    gt_moms_i = gt["globalinertia"]
    gt_iparams = np.array([gt_total_mass, *gt_f_moms, *gt_moms_i])

    regressors = np.array(result["regressors"])
    wrenches = np.array(result["wrenches"])
    ls_iparams = lstsq(regressors.reshape(-1, 10), wrenches.reshape(-1))[0]
    tls_iparams = total_lstsq(regressors.reshape(-1, 10), wrenches.reshape(-1))[0]
    l2_ls = norm(ls_iparams - np.array(gt_iparams), 2)
    l2_tls = norm(tls_iparams - np.array(gt_iparams), 2)
    labels = ["total_mass", "mx", "my", "mz", "ixx", "iyy", "izz", "ixy", "iyz", "izx", "l2"]

    df = pd.DataFrame(
        [[*gt_iparams, np.nan], [*ls_iparams, l2_ls], [*tls_iparams, l2_tls]],
        columns=labels,
        index=["gt_iparams", "ls_iparams", "tls_iparams"],
    )

    print("\nLeast Squares Results DataFrame:")
    print(df)

    # Log the identified inertial params and their ground truth
    simulation.recorder.finish(result["frames"], result["regressors"], gt_iparams)


if __name__ == "__main__":
    main()
