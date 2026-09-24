"""The shipped ".json" must be the perturbed series, not the noise-free reference."""

import json
from pathlib import Path

from recorders.standard_recorder import StandardRecorder


def _recorder(tmp_path: Path) -> StandardRecorder:
    rec = object.__new__(StandardRecorder)
    rec.dataset_dir = tmp_path
    rec.primary_prefix = "transforms"
    rec.base_transform = {"aabb_scale": 1, "noise_model": {"output_series": None}}
    return rec


def _ship(tmp_path: Path) -> None:
    """Mirror main(): perturbed series via the default prefix, noise-free one alongside."""
    rec = _recorder(tmp_path)
    frames = [{"file_path": f"complete/{i:04d}.png"} for i in range(10)]
    iparams = list(range(10))

    rec.base_transform["noise_model"]["output_series"] = "selected_record"
    rec.write_split_transforms(frames, iparams, iparams, iparams, copy_images=False)

    rec.base_transform["noise_model"]["output_series"] = "unperturbed_reference"
    rec.write_split_transforms(
        frames, iparams, iparams, iparams, name_prefix="unperturbed_transforms", copy_images=False
    )


def test_bare_json_is_the_perturbed_series(tmp_path: Path) -> None:
    _ship(tmp_path)

    shipped = tmp_path / "transforms.json"
    assert shipped.exists()
    assert json.loads(shipped.read_text())["noise_model"]["output_series"] == "selected_record"

    backup = tmp_path / "unperturbed_transforms.json.bak"
    assert backup.exists()
    assert json.loads(backup.read_text())["noise_model"]["output_series"] == "unperturbed_reference"

    assert not (tmp_path / "unperturbed_transforms.json").exists()


def test_only_one_file_matches_the_downstream_glob(tmp_path: Path) -> None:
    _ship(tmp_path)
    assert [p.name for p in sorted(tmp_path.glob("*.json"))] == ["transforms.json"]
