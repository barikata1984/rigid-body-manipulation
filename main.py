from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
from omegaconf.errors import MissingMandatoryValue

from core import load_config, generate_model_data, autoinstantiate, get_element_id, simulate


class Scorer:
    def __init__(self, gt_total_mass, gt_f_moms, gt_moms_i):
        self.gt_total_mass = gt_total_mass
        self.gt_f_moms = gt_f_moms
        self.gt_moms_i = gt_moms_i

    @property
    def gt_iparams(self):
        return (self.gt_total_mass, *self.gt_f_moms, *self.gt_moms_i)

    def _get_partial_score(self, est, gt, scale=None):
        score = np.power(est - gt, 2).sum()
        return score if scale is None else score / np.power(scale, 2)

    def calculate(self, estimate):
        aabb_scale = cfg.logger.aabb_scale
        score  = self._get_partial_score(estimate[0], self.gt_total_mass,
                                   scale=self.gt_total_mass*np.power(aabb_scale, 0))  # eliminate [kg]
        score += self._get_partial_score(estimate[1:4], self.gt_f_moms,
                                   scale=self.gt_total_mass*np.power(aabb_scale, 1))  # eliminate [kg*m]
        score += self._get_partial_score(estimate[4:10], self.gt_moms_i,
                                   scale=self.gt_total_mass*np.power(aabb_scale, 2))  # eliminate [kg*m^2]
        score /= 10

        return score


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

    dataset_dir = Path.cwd() / "datasets" / f"TEST_{dir}"  # SETTING DATASET DIR NAME
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
    scorer = Scorer(gt_total_mass, gt_f_moms, gt_moms_i)

    # Log the identified inertial params and their ground truth
    #logger.transform["globalinertia"] = comparison.to_json()
    logger.finish(result["frames"], result["regressors"], scorer)  # video and dataset json generated

