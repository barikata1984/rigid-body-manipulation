from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from transforms3d.euler import mat2euler

CSV_COLUMNS = (
    "id",
    "aabb_scale",
    "total_mass",
    "cx",
    "cy",
    "cz",
    "ixx",
    "iyy",
    "izz",
    "ixy",
    "iyz",
    "izx",
    "pixx",
    "piyy",
    "pizz",
    "rx",
    "ry",
    "rz",
)
MASS_DISTRIBUTION_COLUMNS = ("x", "y", "z", "mass", "mass_density")
# Factor converting each recognised mass_density unit to SI kg/m^3.
DENSITY_UNIT_SCALES = {"kg/m^3": 1.0, "t/mm^3": 1e12}
GROUND_TRUTH_COLUMNS = (
    "id",
    "aabb_scale",
    "total_mass",
    "mx",
    "my",
    "mz",
    "ixx",
    "iyy",
    "izz",
    "ixy",
    "iyz",
    "izx",
    "inertia_tyep",
)


@dataclass(frozen=True)
class VoxelGroundTruth:
    object_id: str
    aabb_scale: float
    mass: float
    com: np.ndarray
    inertia_com: np.ndarray
    principal_inertia: np.ndarray
    principal_rotation: np.ndarray
    euler_sxyz: np.ndarray
    density_unit: str
    density_scale: float

    def csv_row(self) -> list[float | str]:
        inertia = self.inertia_com
        return [
            self.object_id,
            self.aabb_scale,
            self.mass,
            *self.com,
            inertia[0, 0],
            inertia[1, 1],
            inertia[2, 2],
            inertia[0, 1],
            inertia[1, 2],
            inertia[2, 0],
            *self.principal_inertia,
            *self.euler_sxyz,
        ]


def _data_start_row(path: Path) -> int:
    """Return the number of rows preceding the five-column voxel table."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row_index, row in enumerate(reader):
            if tuple(cell.strip() for cell in row[:5]) == MASS_DISTRIBUTION_COLUMNS:
                return row_index + 1
            try:
                [float(cell) for cell in row[:5]]
            except (ValueError, TypeError):
                continue
            if len(row) >= 5:
                return row_index
    raise ValueError(f"No voxel mass-distribution rows found in {path}")


def _axis_grid(values: set[float]) -> tuple[np.ndarray, float]:
    """Return the sorted axis coordinates and their smallest positive spacing."""
    coordinates = np.array(sorted(values), dtype=np.float64)
    if coordinates.size == 0:
        raise ValueError("Voxel mass distribution has no coordinates")
    spacing = 0.0
    if coordinates.size > 1:
        differences = np.diff(coordinates)
        positive = differences[differences > 0.0]
        if positive.size:
            spacing = float(np.min(positive))
    return coordinates, spacing


def _grid_aabb_scale(axis_values: list[set[float]]) -> float:
    boundary_magnitudes = []
    for values in axis_values:
        coordinates, spacing = _axis_grid(values)
        boundary_magnitudes.append(float(np.max(np.abs(coordinates))) + 0.5 * spacing)
    return max(boundary_magnitudes)


def _voxel_volume(axis_values: list[set[float]]) -> float:
    volume = 1.0
    for values in axis_values:
        _, spacing = _axis_grid(values)
        if spacing <= 0.0:
            raise ValueError("Cannot derive voxel spacing from a single-plane grid")
        volume *= spacing
    return volume


def _detect_density_unit(ratio_min: float, ratio_max: float, voxel_volume: float, path: Path) -> tuple[str, float]:
    """Identify the density unit from ``mass / mass_density``, which must equal the voxel volume.

    Inventor exports densities in t/mm^3, whose numeric values are 1e-12 times the
    SI ones, so the ratio comes out 1e12 times too large.
    """
    for unit, scale in DENSITY_UNIT_SCALES.items():
        expected = voxel_volume * scale
        if max(abs(ratio_min - expected), abs(ratio_max - expected)) <= 1e-6 * expected:
            return unit, scale
    raise ValueError(
        f"Unrecognised mass_density unit in {path}: mass/mass_density spans "
        f"[{ratio_min:.15g}, {ratio_max:.15g}] but the voxel volume is {voxel_volume:.15g} m^3"
    )


def summarize_mass_distribution(
    input_path: str | Path,
    object_id: str,
    *,
    aabb_scale: float | None = None,
    chunksize: int = 200_000,
) -> VoxelGroundTruth:
    """Compute mass properties from voxel-center point masses.

    The input may be a raw headerless ``gt_mass_distr.csv`` or the legacy
    ``ground_truth.csv`` format containing a two-row summary and a voxel-table
    header. The legacy summary is deliberately ignored: all properties are
    recomputed from the voxel rows.
    """
    path = Path(input_path)
    skiprows = _data_start_row(path)
    mass = 0.0
    first_moment = np.zeros(3, dtype=np.float64)
    second_moment = np.zeros((3, 3), dtype=np.float64)
    axis_values: list[set[float]] = [set(), set(), set()]
    ratio_min = np.inf
    ratio_max = 0.0

    chunks = pd.read_csv(
        path,
        skiprows=skiprows,
        header=None,
        names=MASS_DISTRIBUTION_COLUMNS,
        usecols=range(5),
        dtype=np.float64,
        chunksize=chunksize,
    )
    for chunk in chunks:
        values = chunk.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite value in voxel mass distribution {path}")
        if np.any(values[:, 3] < 0.0):
            raise ValueError(f"Negative voxel mass in {path}")

        positions = values[:, :3]
        masses = values[:, 3]
        mass += float(np.sum(masses))
        first_moment += np.sum(masses[:, None] * positions, axis=0)
        second_moment += (positions * masses[:, None]).T @ positions
        for axis in range(3):
            axis_values[axis].update(np.unique(positions[:, axis]))

        occupied = values[:, 4] > 0.0
        if np.any(occupied):
            ratios = masses[occupied] / values[occupied, 4]
            ratio_min = min(ratio_min, float(np.min(ratios)))
            ratio_max = max(ratio_max, float(np.max(ratios)))

    if mass <= 0.0:
        raise ValueError(f"Total voxel mass must be positive, got {mass} from {path}")

    com = first_moment / mass
    inertia_origin = np.trace(second_moment) * np.eye(3) - second_moment
    parallel_axis = mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    inertia_com = inertia_origin - parallel_axis
    inertia_com = 0.5 * (inertia_com + inertia_com.T)

    principal_inertia, eigenvectors = np.linalg.eigh(inertia_com)
    if np.any(principal_inertia <= 0.0):
        raise ValueError(f"Voxel inertia is not positive definite: {principal_inertia}")
    if np.linalg.det(eigenvectors) < 0.0:
        eigenvectors[:, -1] *= -1.0

    # setup.get_target_object_ground_truth reconstructs the diagonal tensor as
    # R @ I_body @ R.T, so its R has principal axes in rows.
    principal_rotation = eigenvectors.T
    reconstructed = principal_rotation.T @ np.diag(principal_inertia) @ principal_rotation
    np.testing.assert_allclose(reconstructed, inertia_com, rtol=1e-10, atol=1e-14)
    euler_sxyz = np.asarray(mat2euler(principal_rotation, axes="sxyz"), dtype=np.float64)

    if ratio_max <= 0.0:
        raise ValueError(f"No positive mass_density value in {path}")
    density_unit, density_scale = _detect_density_unit(ratio_min, ratio_max, _voxel_volume(axis_values), path)
    print(f"Detected mass_density unit {density_unit} in {path} (scale {density_scale:g} to kg/m^3)")

    resolved_aabb_scale = _grid_aabb_scale(axis_values) if aabb_scale is None else aabb_scale
    return VoxelGroundTruth(
        object_id=object_id,
        aabb_scale=float(resolved_aabb_scale),
        mass=mass,
        com=com,
        inertia_com=inertia_com,
        principal_inertia=principal_inertia,
        principal_rotation=principal_rotation,
        euler_sxyz=euler_sxyz,
        density_unit=density_unit,
        density_scale=density_scale,
    )


def write_object_cad_gt(result: VoxelGroundTruth, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        writer.writerow(result.csv_row())
    return output


def write_ground_truth(
    result: VoxelGroundTruth,
    mass_distribution_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write the summary plus voxel rows expected by the NeMD trainer."""
    source = Path(mass_distribution_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("Ground-truth output must differ from the mass-distribution input")

    com = result.com
    inertia_com = result.inertia_com
    inertia_origin = inertia_com + result.mass * (
        np.dot(com, com) * np.eye(3) - np.outer(com, com)
    )
    summary = [
        result.object_id,
        result.aabb_scale,
        result.mass,
        *(result.mass * com),
        inertia_origin[0, 0],
        inertia_origin[1, 1],
        inertia_origin[2, 2],
        inertia_origin[0, 1],
        inertia_origin[1, 2],
        inertia_origin[2, 0],
        "aabb",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    data_start = _data_start_row(source)
    with output.open("w", newline="") as dst:
        writer = csv.writer(dst, lineterminator="\n")
        writer.writerow(GROUND_TRUTH_COLUMNS)
        writer.writerow(summary)
        writer.writerow(MASS_DISTRIBUTION_COLUMNS)
        with source.open(newline="") as src:
            for _ in range(data_start):
                next(src)
            if result.density_scale == 1.0:
                shutil.copyfileobj(src, dst)
            else:
                for row in csv.reader(src):
                    row[4] = repr(float(row[4]) * result.density_scale)
                    writer.writerow(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the legacy object_cad_gt.csv schema from a voxel mass distribution."
    )
    parser.add_argument("input", type=Path, help="gt_mass_distr.csv or ground_truth.csv")
    parser.add_argument("output", type=Path, help="destination object_cad_gt.csv")
    parser.add_argument("--object-id", required=True, help="identifier written to the first column")
    parser.add_argument("--aabb-scale", type=float, default=None, help="override the grid-derived AABB half extent")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--ground-truth-output",
        type=Path,
        default=None,
        help="also write a trainer-compatible ground_truth.csv containing the voxel rows",
    )
    args = parser.parse_args()

    result = summarize_mass_distribution(
        args.input,
        args.object_id,
        aabb_scale=args.aabb_scale,
        chunksize=args.chunksize,
    )
    output = write_object_cad_gt(result, args.output)
    print(f"Wrote {output}")
    if args.ground_truth_output is not None:
        ground_truth_output = write_ground_truth(result, args.input, args.ground_truth_output)
        print(f"Wrote {ground_truth_output}")
    print(f"mass={result.mass:.15g}")
    print(f"com={result.com.tolist()}")
    print(f"inertia_com={result.inertia_com.tolist()}")
    print(f"aabb_scale={result.aabb_scale:.15g}")
    print(f"density_unit={result.density_unit} density_scale={result.density_scale:g}")


if __name__ == "__main__":
    main()
