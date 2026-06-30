import json
from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
import tyro
from numpy.linalg import lstsq, norm
from omegaconf import OmegaConf

from factory import instantiate
from omegaconf_custom_resolvers import pi_converter
from regressions import total_lstsq
from simulators import SimulatorConfig
from simulators.setup import generate_model_data, get_element_id
from utilities import json_to_namespace

OmegaConf.register_new_resolver("pi", pi_converter)


def resolve_config():
    cli_config = tyro.cli(SimulatorConfig)
    yaml_config = OmegaConf.load(cli_config.exp_setup)
    cfg = OmegaConf.merge(SimulatorConfig, yaml_config, cli_config)
    m, d, gt = generate_model_data(cfg)

    target_trajectory = None
    if cfg.target_trajectory:
        with open(cfg.target_trajectory) as f:
            target_trajectory = json_to_namespace(json.load(f))

    if OmegaConf.is_missing(cfg.recorder, "fps"):
        if target_trajectory is None:
            raise RuntimeError("recorder.fps is not set and no target_trajectory is loaded to infer it from")
        cfg.recorder.fps = int(target_trajectory.fps)
    else:
        cfg.recorder.fps = int(cfg.recorder.fps)

    if OmegaConf.is_missing(cfg.recorder, "aabb_scale"):
        aabb_scale = m.numeric_data[get_element_id(m, "numeric", "target/aabb_scale")]
        cfg.recorder.aabb_scale = float(aabb_scale)
    else:
        cfg.recorder.aabb_scale = float(cfg.recorder.aabb_scale)

    object_dir = Path(cfg.object)
    if OmegaConf.is_missing(cfg.recorder, "dataset_dir"):
        dataset_dir = Path.cwd() / "datasets" / object_dir.name
        cfg.recorder.dataset_dir = dataset_dir
    else:
        dataset_dir = Path(cfg.recorder.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    object_gt = object_dir / "ground_truth.csv"
    dataset_gt = dataset_dir / "ground_truth.csv"
    if dataset_gt.is_file():
        pass
    elif object_gt.is_file():
        copy(object_gt, dataset_gt)
    else:
        print(f"Warning: {object_gt} not found, skipping ground truth copy.")

    return cfg, m, d, gt, target_trajectory


def run_simulation(cfg, m, d, target_trajectory):
    simulator_cfg = OmegaConf.to_object(cfg)
    simulation = instantiate(simulator_cfg, model=m, data=d, target_trajectory=target_trajectory)
    result = simulation.run()
    return simulation, result


def identify_inertial_params(result, gt):
    gt_total_mass = gt["mass"]
    gt_f_moms = gt_total_mass * gt["com"]
    gt_moms_i = gt["globalinertia"]
    gt_iparams = np.array([gt_total_mass, *gt_f_moms, *gt_moms_i])

    regressors = np.array(result["regressors"])
    wrenches = np.array(result["wrenches"])
    ls_iparams = lstsq(regressors.reshape(-1, 10), wrenches.reshape(-1))[0]
    tls_iparams = total_lstsq(regressors.reshape(-1, 10), wrenches.reshape(-1))[0]
    l2_ls = norm(ls_iparams - gt_iparams, 2)
    l2_tls = norm(tls_iparams - gt_iparams, 2)
    labels = ["total_mass", "mx", "my", "mz", "ixx", "iyy", "izz", "ixy", "iyz", "izx", "l2"]

    df = pd.DataFrame(
        [[*gt_iparams, np.nan], [*ls_iparams, l2_ls], [*tls_iparams, l2_tls]],
        columns=labels,
        index=["gt_iparams", "ls_iparams", "tls_iparams"],
    )
    print("\nLeast Squares Results DataFrame:")
    print(df)

    return gt_iparams, ls_iparams, tls_iparams


def main():
    cfg, m, d, gt, target_trajectory = resolve_config()
    simulation, result = run_simulation(cfg, m, d, target_trajectory)
    gt_iparams, ls_iparams, tls_iparams = identify_inertial_params(result, gt)
    simulation.recorder.finish(result["frames"], gt_iparams, ls_iparams, tls_iparams)


if __name__ == "__main__":
    main()
