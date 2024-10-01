from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
from omegaconf.errors import MissingMandatoryValue

from core import load_config, generate_model_data, autoinstantiate, get_element_id, simulate


if __name__ == "__main__":

    cfg = load_config()  # priority: cli > cli-specified .yaml > base.yaml > hard-coded
    m, d, gt = generate_model_data(cfg)
    globalinertia = gt["globalinertia"]

    # Fill (potentially) missing fields of a logger configulation =================
    try:
        cfg.logger.aabb_scale
    except MissingMandatoryValue:
        target_object_id = get_element_id(m, 'numeric', 'target/aabb_scale')
        aabb_scale = m.numeric_data[target_object_id]
        cfg.logger.aabb_scale = float(aabb_scale)

    try:
        dir = cfg.logger.dataset_dir
    except MissingMandatoryValue:
        dir = cfg.target_name

    dataset_dir = Path.cwd() / "datasets" / dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cfg.logger.dataset_dir = dataset_dir

    # Copy the ground truth mass distribution file to the dataset file ============
    target_gt = Path.cwd() / "xml_models" / "targets" / dir / "ground_truth.csv"
    dataset_gt = dataset_dir / "ground_truth.csv"
    if dataset_gt.is_file():
        print("'ground_truth.csv' is not copied to the dataset dir since the file "
              "with the same name already existsd.")
    else:
        copy(target_gt, dataset_gt)

    # Instantiate necessary classes ===============================================
    logger = autoinstantiate(cfg.logger, m, d)
    planner = autoinstantiate(cfg.planner, m, d)
    controller = autoinstantiate(cfg.controller, m, d)

    result = simulate(m, d, logger, planner, controller)  # main process

    # Show inertial params identified with the least squares method
    gt_total_mass = gt["mass"]
    gt_f_moms = gt_total_mass * gt["com"]  # type: ignore
    gt_moms_i = gt["globalinertia"]
    globaliparams = [gt_total_mass, *gt_f_moms, *gt_moms_i, 0]  # np.nan]  # type: ignore

    lstsq = result["lstsq"]
    score = np.abs(lstsq[0] - gt_total_mass)
    score += np.abs(lstsq[1:4] - gt_f_moms).sum() / cfg.logger.aabb_scale
    score += np.abs(lstsq[4:10] - gt_moms_i).sum() / cfg.logger.aabb_scale**2
    score /= gt_total_mass
    lstsq = lstsq.tolist() + [score]

    indices = ["run_name", 
               "total_mass",
               "mx", "my", "mz",
               "ixx", "iyy", "izz", "ixy", "iyz", "izx",
               "score",
               ]

    comparison = pd.DataFrame([["global_gt", *globaliparams],
                               ["lstsq", *lstsq]],
                              columns=indices,  # type: ignore
                              )

    print(f"comparison:\n{comparison}")

    # Log the identified inertial params and their ground truth
    logger.transform["globalinertia"] = comparison.to_json()
    logger.finish()  # video and dataset json generated

