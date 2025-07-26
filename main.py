from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
import tyro
from numpy.linalg import lstsq, norm
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from configurations import instantiate
from omegaconf_custom_resolvers import pi_converter
from regressions import total_lstsq
from simulator import SimulatorConfig, generate_model_data
from utilities import get_element_id

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

    simulator_cfg = OmegaConf.to_object(cfg)
    simulator = instantiate(simulator_cfg, m, d)

    result = simulator.run()

    # Get ground truth inertial parameters
    gt_total_mass = gt["mass"]
    gt_f_moms = gt_total_mass * gt["com"]  # type: ignore
    gt_moms_i = gt["globalinertia"]
    gt_iparams = [gt_total_mass, *gt_f_moms, *gt_moms_i]  # type: ignore

    # Ordinal/Total least squares
    regressors = result["regressors"].reshape((-1, 10))
    fts_sen = result["fts_sen"].reshape(-1)
    ls_iparams = lstsq(regressors, fts_sen)[0]  # test
    tls_iparams = total_lstsq(regressors, fts_sen)[0]  # test
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

    simulator.recorder.finish(result["frames"], result["regressors"], gt_iparams)


if __name__ == "__main__":
    main()
