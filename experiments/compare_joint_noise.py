"""Run the usual simulator CLI, without rendering, and compare paired observations.

Example: python -m experiments.compare_joint_noise --object ... --target-trajectory ...
    --no-control-noise --seed 42
    --recorder.dataset-dir /tmp/independent_50_150
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from factory import instantiate
from main import identify_inertial_params, resolve_config
from simulators.simulator import Simulator
from utilities import get_element_id


def main():
    cfg, model, data, gt, trajectory = resolve_config()
    if not cfg.get_unperturbed or not cfg.record_noise or cfg.control_noise:
        raise ValueError("Use --get-unperturbed --record-noise --no-control-noise for paired observations")
    out = Path(cfg.recorder.dataset_dir)
    recorder = SimpleNamespace(
        base_transform={},
        videowriter=SimpleNamespace(isOpened=lambda: True),
        cam_id=get_element_id(model, "camera", cfg.recorder.track_cam_name),
        render=lambda *args: None,
        complete_image_dir=out / "complete",
        dataset_dir=out,
    )

    def make_component(config, **kwargs):
        return recorder if config.target_class == "StandardRecorder" else instantiate(config, **kwargs)

    with (
        patch("simulators.simulator.instantiate", make_component),
        patch.object(Simulator, "_visualize", lambda *args: None),
    ):
        sim = Simulator(OmegaConf.to_object(cfg), model=model, data=data, target_trajectory=trajectory)
        result = sim.run()
    time = np.array(sim.data.time)
    noisy = np.array(sim.data.act_trajectory)
    clean = np.array(sim.data_unperturbed.act_trajectory)
    np.savez(out / "joint_observations.npz", time=time, clean=clean, noisy=noisy)
    OmegaConf.save(cfg, out / "config.yaml")
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True, layout="constrained")
    names = (("x", "y", "z"), ("q3", "q4", "q5"))
    units = (("m", "m/s", "m/s²"), ("rad", "rad/s", "rad/s²"))
    sigma = (
        sim.sensors.jointpos_stddev[None, :]
        * np.array([1, sim.sensors.jointvel_noise_scaler, sim.sensors.jointacc_noise_scaler])[:, None]
    )
    for row in range(2):
        for col, title in enumerate(("Position", "Velocity", "Acceleration")):
            ax = axes[row, col]
            for j, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
                k = 3 * row + j
                ax.plot(time, clean[:, col, k], color=color, lw=1.3, label=f"{names[row][j]} clean")
                ax.plot(
                    time, noisy[:, col, k], color=color, ls="--", lw=0.7, alpha=0.75, label=f"{names[row][j]} noisy"
                )
            ax.set(
                title=f"{title} | noise σ = {sigma[col, row * 3]:.3g} {units[row][col]}",
                ylabel=f"{'Translation' if row == 0 else 'Rotation'} [{units[row][col]}]",
            )
            ax.grid(alpha=0.2)
            ax.legend(ncol=3, fontsize=8, loc="upper right")
            if row == 1:
                ax.set_xlabel("Time [s]")
    fig.suptitle(f"{Path(cfg.object).name}: {sim.sensors.profile.name} / MuJoCo truth (seed {sim.sensors.seed})")
    fig.savefig(out / "joint_noise_comparison.png", dpi=180)
    fig.savefig(out / "joint_noise_comparison.svg")
    plt.close(fig)
    clean_result = {"regressors": result["unperturbed_regressors"], "wrenches": result["unperturbed_wrenches"]}
    truth, ls_noisy, _ = identify_inertial_params(result, gt, sim.pose_sen_obj.inv())
    _, ls_clean, _ = identify_inertial_params(clean_result, gt, sim.pose_sen_obj.inv())
    pd.DataFrame(
        {"gt": truth, "ols_clean": ls_clean, "ols_noisy": ls_noisy},
        index=["mass", "mx", "my", "mz", "ixx", "iyy", "izz", "ixy", "iyz", "izx"],
    ).to_csv(out / "inertia_comparison.csv")
    print(f"Plot: {out / 'joint_noise_comparison.png'}")


if __name__ == "__main__":
    main()
