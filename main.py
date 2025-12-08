from pathlib import Path
from shutil import copy

from omegaconf.errors import MissingMandatoryValue

from core import autoinstantiate, generate_model_data, get_element_id, load_config, simulate


def main():
    cfg = load_config()  # priority: cli > cli-specified .yaml > base.yaml > hard-coded
    m, d, gt = generate_model_data(cfg)

    # Fill (potentially) missing fields of a logger configulation =================
    try:
        cfg.recorder.aabb_scale
    except MissingMandatoryValue:
        target_object_id = get_element_id(m, "numeric", "target/aabb_scale")
        aabb_scale = m.numeric_data[target_object_id]
        cfg.recorder.aabb_scale = float(aabb_scale)

    try:
        dir = cfg.recorder.dataset_dir
    except MissingMandatoryValue:
        dir = cfg.target_name

    dataset_dir = Path.cwd() / "datasets" / f"{dir}"  # SETTING DATASET DIR NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cfg.recorder.dataset_dir = dataset_dir

    # Copy the ground truth mass distribution file to the dataset file ============
    target_gt = Path.cwd() / "xml_models" / "targets" / dir / "ground_truth.csv"
    dataset_gt = dataset_dir / "ground_truth.csv"
    if dataset_gt.is_file():
        print("'ground_truth.csv' is not copied to the dataset dir since the file with the same name already existsd.")
    else:
        copy(target_gt, dataset_gt)

    # Instantiate necessary classes ===============================================
    recorder = autoinstantiate(cfg.recorder, m, d)
    planner = autoinstantiate(cfg.planner, m, d)
    controller = autoinstantiate(cfg.controller, m, d)

    # import pdb

    # pdb.set_trace()  # Debugger breakpoint

    result = simulate(m, d, recorder, planner, controller)  # main process

    # Show inertial params identified with the least squares method
    gt_total_mass = gt["mass"]
    gt_f_moms = gt_total_mass * gt["com"]  # type: ignore
    gt_moms_i = gt["globalinertia"]
    gt_iparams = [gt_total_mass, *gt_f_moms, *gt_moms_i]

    # Log the identified inertial params and their ground truth
    # logger.transform["globalinertia"] = comparison.to_json()
    recorder.finish(result["frames"], result["regressors"], gt_iparams)  # video and dataset json generated


if __name__ == "__main__":
    main()
