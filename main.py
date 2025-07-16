from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy

import tyro
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from tyro import MISSING

from configurations import instantiate
from controllers import LinearQuadraticRegulatorConfig
from planners import JointPositionPlannerConfig
from recorders import StandardRecorderConfig
from simulator import Simulator, generate_model_data
from utilities import get_element_id


@dataclass
class SimulationConfig:
    manipulator_dir: str = "xml_models/manipulators/sequential"
    target_dir: str = MISSING
    reset_keyframe: str = "initial_state"
    recorder: StandardRecorderConfig = field(default_factory=StandardRecorderConfig)
    planner: JointPositionPlannerConfig = field(default_factory=JointPositionPlannerConfig)
    controller: LinearQuadraticRegulatorConfig = field(default_factory=LinearQuadraticRegulatorConfig)
    exp_setup: str = "experimental_setups/base.yaml"
    config_export_path: str | None = None


if __name__ == "__main__":
    cli_config = tyro.cli(SimulationConfig)
    cli_specified_yaml = cli_config.exp_setup
    yaml_config = OmegaConf.load(cli_specified_yaml)
    base_config = SimulationConfig()

    cfg = OmegaConf.merge(base_config, yaml_config, cli_config)  # priority: cli > cli-specified yaml > vanilla
    m, d, gt = generate_model_data(cfg)

    # Fill (potentially) missing fields of a logger configulation =================

    try:
        cfg.recorder.aabb_scale = float(cfg.recorder.aabb_scale)
    except MissingMandatoryValue:
        target_object_id = get_element_id(m, "numeric", "target/aabb_scale")
        aabb_scale = m.numeric_data[target_object_id]
        cfg.recorder.aabb_scale = float(aabb_scale)

    target_dir = Path(cfg.target_dir)
    target_gt = target_dir / "ground_truth.csv"

    try:
        dataset_dir = Path(cfg.recorder.dataset_dir)
    except MissingMandatoryValue:
        dataset_dir = Path.cwd() / "datasets" / target_dir.name
        cfg.recorder.dataset_dir = dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy the ground truth mass distribution file to the dataset file ============
    dataset_gt = dataset_dir / "ground_truth.csv"
    if dataset_gt.is_file():
        print("'ground_truth.csv' is not copied to the dataset dir since the file with the same name already existsd.")
    else:
        copy(target_gt, dataset_gt)

    recorder = instantiate(cfg.recorder, m, d)
    planner = instantiate(cfg.planner, m, d)
    controller = instantiate(cfg.controller, m, d)
    simulator = Simulator(m, d, recorder, planner, controller)

    result = simulator.run()  # m, d, recorder, planner, controller)  # main process

    gt_total_mass = gt["mass"]
    gt_f_moms = gt_total_mass * gt["com"]
    gt_moms_i = gt["globalinertia"]
    gt_iparams = [gt_total_mass, *gt_f_moms, *gt_moms_i]

    recorder.finish(result["frames"], result["regressors"], gt_iparams)
