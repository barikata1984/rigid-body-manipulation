import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from simulators import SimulatorConfig
from simulators.camera_calibration import (
    CameraCalibrationConfig,
    ForegroundMeasurement,
    calibrate_distance,
    foreground_ratio,
    load_calibration,
    mask_touches_border,
    representative_qpos,
    save_calibration,
)


def test_foreground_ratio_and_border_margin():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:5, 3:7] = True
    assert foreground_ratio(mask) == pytest.approx(12 / 80)
    assert not mask_touches_border(mask, margin=2)
    mask[1, 3] = True
    assert mask_touches_border(mask, margin=2)


def test_calibrate_distance_uses_median_and_no_crop():
    config = CameraCalibrationConfig(
        mode="calibrate",
        target_foreground_ratio=0.1,
        min_distance_factor=1.0,
        max_distance_factor=8.0,
        distance_tolerance=1.0e-5,
        ratio_tolerance=2.0e-3,
    )

    def evaluate(factor):
        ratio = 0.4 / factor**2
        return [
            ForegroundMeasurement(ratio=ratio, touches_border=factor < 1.5),
            ForegroundMeasurement(ratio=ratio * 1.1, touches_border=factor < 1.5),
            ForegroundMeasurement(ratio=ratio * 0.9, touches_border=factor < 1.5),
        ]

    result = calibrate_distance(evaluate, config)
    assert result.distance_factor == pytest.approx(2.0, abs=0.02)
    assert result.measured_foreground_ratio == pytest.approx(0.1, abs=0.002)
    assert result.sample_count == 3


def test_calibrate_distance_rejects_unreachable_target():
    config = CameraCalibrationConfig(
        mode="calibrate",
        target_foreground_ratio=0.5,
        min_distance_factor=1.0,
        max_distance_factor=4.0,
    )

    with pytest.raises(ValueError, match="too large"):
        calibrate_distance(
            lambda factor: [ForegroundMeasurement(ratio=0.1 / factor, touches_border=False)],
            config,
        )


def test_representative_qpos_is_evenly_spaced():
    frames = [[[float(i)], [0.0], [0.0]] for i in range(10)]
    samples = representative_qpos(frames, sample_count=4, default_qpos=np.array([99.0]), nq=1)
    np.testing.assert_allclose(samples, [[0.0], [3.0], [6.0], [9.0]])


def test_save_and_load_calibration(tmp_path: Path):
    config = CameraCalibrationConfig()
    result = calibrate_distance(
        lambda _factor: [ForegroundMeasurement(ratio=0.1, touches_border=False)],
        config,
    )
    path = tmp_path / "camera_calibration.json"
    save_calibration(path, "xml_models/targets/hammer", 0.195, result)
    loaded = load_calibration(path, "xml_models/targets/hammer", 0.195)
    assert loaded.distance_factor == pytest.approx(result.distance_factor)
    payload = json.loads(path.read_text())
    assert payload["version"] == 1
    assert "hammer" in payload["objects"]


def test_yaml_camera_config_is_not_overridden_by_cli_defaults(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import main

    cli_config = SimulatorConfig(object="dummy")
    yaml_config = OmegaConf.create(
        {
            "camera_calibration": {
                "mode": "calibrate",
                "distance_factor": 6.0,
                "target_foreground_ratio": 0.05,
            }
        }
    )
    cfg = OmegaConf.merge(SimulatorConfig, yaml_config, cli_config)
    monkeypatch.setattr(main.sys, "argv", ["main.py"])
    merged = main._merge_camera_calibration_config(cfg, yaml_config, cli_config)
    assert merged.camera_calibration.mode == "calibrate"
    assert merged.camera_calibration.distance_factor == pytest.approx(6.0)

    cli_config.camera_calibration.mode = "load"
    cli_config.camera_calibration.calibration_path = "coefficients.json"
    monkeypatch.setattr(
        main.sys,
        "argv",
        ["main.py", "--camera-calibration.mode", "load", "--camera-calibration.calibration-path", "coefficients.json"],
    )
    cfg = OmegaConf.merge(SimulatorConfig, yaml_config, cli_config)
    merged = main._merge_camera_calibration_config(cfg, yaml_config, cli_config)
    assert merged.camera_calibration.mode == "load"
    assert merged.camera_calibration.calibration_path == "coefficients.json"
