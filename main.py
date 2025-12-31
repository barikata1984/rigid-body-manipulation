import json
from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
import tyro
from numpy.linalg import lstsq, norm
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from factory import instantiate
from simulators.setup import generate_model_data, get_element_id
from omegaconf_custom_resolvers import pi_converter
from regressions import total_lstsq
from simulators import SimulatorConfig

OmegaConf.register_new_resolver("pi", pi_converter)


def main():
    cli_config = tyro.cli(SimulatorConfig)
    cli_specified_yaml = cli_config.exp_setup
    yaml_config = OmegaConf.load(cli_specified_yaml)
    base_config = SimulatorConfig()

    cfg = OmegaConf.merge(base_config, yaml_config, cli_config)  # priority: cli > cli-specified yaml > vanilla
    m, d, gt = generate_model_data(cfg)

    # Fill (potentially) missing fields of a logger configulation =================
    try:
        cfg.recorder.aabb_scale = float(cfg.recorder.aabb_scale)
    except MissingMandatoryValue:
        aabb_scale = m.numeric_data[get_element_id(m, "numeric", "target/aabb_scale")]
        cfg.recorder.aabb_scale = float(aabb_scale)

    object_dir = Path(cfg.object)
    object_gt = object_dir / "ground_truth.csv"

    try:
        dataset_dir = Path(cfg.recorder.dataset_dir)
    except MissingMandatoryValue:
        dataset_dir = Path.cwd() / "datasets" / object_dir.name
        cfg.recorder.dataset_dir = dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy the ground truth mass distribution file to the dataset file ============
    dataset_gt = dataset_dir / "ground_truth.csv"
    if dataset_gt.is_file():
        print("'ground_truth.csv' is not copied to the dataset dir since the file with the same name already existsd.")
    else:
        copy(object_gt, dataset_gt)

    # Load trajectory and extract excitation indices if available
    excitation_slice = slice(None)  # Default to full trajectory
    if cfg.target_trajectory:
        with open(cfg.target_trajectory) as f:
            trajectory_data = json.load(f)
        if "excitation" in trajectory_data and "start_index" in trajectory_data["excitation"]:
            start = trajectory_data["excitation"]["start_index"]
            end = trajectory_data["excitation"]["end_index"]
            excitation_slice = slice(start, end)
            print(f"Excitation trajectory slice found: {start} to {end}")

    # Instantiate the simulator ===============================================
    simulator_cfg = OmegaConf.to_object(cfg)
    simulation = instantiate(simulator_cfg, m, d)

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
    simulation.recorder.finish(result["frames"], result["regressors"], gt_iparams)  # video and dataset json generated


if __name__ == "__main__":
    main()
