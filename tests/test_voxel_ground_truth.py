import csv

import numpy as np
import pytest

from utilities.voxel_ground_truth import summarize_mass_distribution, write_ground_truth, write_object_cad_gt


# Voxel spacing is 2.0 on every axis, so a consistent SI density is mass / 2**3.
def _cube_rows(density_scale=1.0):
    density = 0.125 / density_scale
    return [[x, y, z, 1.0, density] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]


def test_summarize_headerless_mass_distribution(tmp_path):
    source = tmp_path / "gt_mass_distr.csv"
    with source.open("w", newline="") as f:
        csv.writer(f).writerows(_cube_rows())

    result = summarize_mass_distribution(source, "cube", chunksize=3)

    assert result.mass == 8.0
    np.testing.assert_allclose(result.com, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(result.inertia_com, 16.0 * np.eye(3), atol=1e-15)
    np.testing.assert_allclose(
        result.principal_rotation.T @ np.diag(result.principal_inertia) @ result.principal_rotation,
        result.inertia_com,
        atol=1e-15,
    )
    assert result.aabb_scale == 2.0


def test_legacy_summary_is_ignored_and_csv_is_written(tmp_path):
    source = tmp_path / "ground_truth.csv"
    with source.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "aabb_scale", "total_mass", "mx", "my", "mz"])
        writer.writerow(["wrong-summary", 99, 999, 999, 999, 999])
        writer.writerow(["x", "y", "z", "mass", "mass_density"])
        writer.writerows(_cube_rows())

    result = summarize_mass_distribution(source, "cube", aabb_scale=3.0, chunksize=4)
    output = write_object_cad_gt(result, tmp_path / "object_cad_gt.csv")

    with output.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[1][0] == "cube"
    assert float(rows[1][1]) == 3.0
    assert float(rows[1][2]) == 8.0


def test_write_ground_truth_uses_voxel_summary_and_preserves_distribution(tmp_path):
    source = tmp_path / "gt_mass_distr.csv"
    voxel_rows = _cube_rows()
    with source.open("w", newline="") as f:
        csv.writer(f).writerows(voxel_rows)

    result = summarize_mass_distribution(source, "cube", chunksize=3)
    output = write_ground_truth(result, source, tmp_path / "ground_truth.csv")

    with output.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][-1] == "inertia_tyep"
    assert rows[1][0] == "cube"
    assert float(rows[1][2]) == 8.0
    np.testing.assert_allclose([float(value) for value in rows[1][3:6]], np.zeros(3), atol=1e-15)
    np.testing.assert_allclose([float(value) for value in rows[1][6:9]], [16.0, 16.0, 16.0])
    assert rows[1][-1] == "aabb"
    assert tuple(rows[2]) == ("x", "y", "z", "mass", "mass_density")
    np.testing.assert_allclose(np.asarray(rows[3:], dtype=float), np.asarray(voxel_rows))


def _write_rows(path, rows):
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


def test_si_density_is_detected_and_left_unscaled(tmp_path):
    source = _write_rows(tmp_path / "si.csv", _cube_rows())
    result = summarize_mass_distribution(source, "cube", chunksize=3)

    assert result.density_unit == "kg/m^3"
    assert result.density_scale == 1.0
    output = write_ground_truth(result, source, tmp_path / "ground_truth.csv")
    with output.open(newline="") as f:
        densities = [float(row[4]) for row in list(csv.reader(f))[3:]]
    np.testing.assert_allclose(densities, 0.125)


def test_tonne_per_cubic_millimetre_density_is_converted_to_si(tmp_path):
    source = _write_rows(tmp_path / "tmm3.csv", _cube_rows(density_scale=1e12))
    result = summarize_mass_distribution(source, "cube", chunksize=3)

    assert result.density_unit == "t/mm^3"
    assert result.density_scale == 1e12
    output = write_ground_truth(result, source, tmp_path / "ground_truth.csv")
    with output.open(newline="") as f:
        rows = list(csv.reader(f))[3:]
    np.testing.assert_allclose([float(row[4]) for row in rows], 0.125, rtol=1e-12)
    # Masses and coordinates must survive the rewrite untouched.
    np.testing.assert_allclose(
        np.asarray([row[:4] for row in rows], dtype=float),
        np.asarray([row[:4] for row in _cube_rows()], dtype=float),
    )


def test_unknown_density_unit_raises(tmp_path):
    rows = [[x, y, z, 1.0, 1.0] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    source = _write_rows(tmp_path / "unknown.csv", rows)

    with pytest.raises(ValueError, match="Unrecognised mass_density unit"):
        summarize_mass_distribution(source, "cube", chunksize=3)
