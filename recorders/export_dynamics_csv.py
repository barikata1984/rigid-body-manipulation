"""Flatten the dynamics rows of a dataset's transforms file into a single CSV.

The identification scripts want the excitation-side quantities (object pose in the sensor
frame, twist, acceleration, measured wrench, optionally the regressor) as a plain table, one
row per frame. Merged datasets additionally carry the frame number each dynamics row came from
in the source run, which is emitted as ``source_index`` so a row can be traced back.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from recorders.merge import _primary_transforms

# (key, number of scalars once flattened) in the order the columns are written.
DYNAMICS_COLUMNS = (
    ("pose_sen_obj", 16),
    ("twist_sen", 6),
    ("dtwist_sen", 6),
    ("wrench", 6),
    ("regressor", 60),
)


def _flatten(value: object) -> list[float]:
    if isinstance(value, list):
        return [x for v in value for x in _flatten(v)]
    return [value]  # type: ignore[list-item]


def export(dataset_dir: Path, out_csv: Path | None = None, include_regressor: bool = False) -> Path:
    """Write the CSV next to the transforms file (or at ``out_csv``) and return its path."""
    with open(_primary_transforms(dataset_dir)) as f:
        meta = json.load(f)
    frames = meta["frames"]
    if not frames:
        raise SystemExit(f"{dataset_dir}: transforms file has no frames")

    keys = [(k, n) for k, n in DYNAMICS_COLUMNS if k in frames[0] and (include_regressor or k != "regressor")]
    if not keys:
        raise SystemExit(f"{dataset_dir}: no dynamics keys found in the first frame")
    source_indices = meta.get("merge_sources", {}).get("dynamics", {}).get("source_indices")

    header = ["frame"]
    if source_indices is not None:
        header.append("source_index")
    header += [f"{k}_{i}" for k, n in keys for i in range(n)]

    out_csv = out_csv or dataset_dir / "dynamics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, frame in enumerate(frames):
            row: list[object] = [i]
            if source_indices is not None:
                row.append(source_indices[i])
            for k, n in keys:
                flat = _flatten(frame[k])
                if len(flat) != n:
                    raise SystemExit(f"frame {i}: {k} has {len(flat)} scalars, expected {n}")
                row += flat
            writer.writerow(row)
    return out_csv


@dataclass
class ExportConfig:
    dataset_dir: Path
    """Dataset directory containing the transforms file."""
    out_csv: Path | None = None
    """Output path (defaults to ``<dataset_dir>/dynamics.csv``)."""
    include_regressor: bool = False
    """Also emit the 60 flattened regressor columns."""


def main() -> None:
    cfg = tyro.cli(ExportConfig)
    path = export(cfg.dataset_dir, cfg.out_csv, cfg.include_regressor)
    with open(path) as f:
        rows = sum(1 for _ in f)
    print(f"wrote {path} ({rows - 1} data rows)")


if __name__ == "__main__":
    main()
