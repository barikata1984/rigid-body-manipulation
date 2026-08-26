"""Merge two simulation runs into a single NeMD/NeRF dataset directory.

The downstream loader (``wisp/datasets/formats/nemd_standard_dataset.py``) reads one frame
entry per dataset row and stacks both the image channels and the dynamics channels to
``(num_frames, ...)`` tensors, so image and dynamics cannot be shipped as two independent
streams. Merging therefore pairs the runs row by row: the spline run supplies the image and
camera pose (all-around viewpoints, undisturbed by excitation), the excited run supplies the
dynamics row (pose/twist/acceleration/wrench/regressor). The two losses that consume them are
independent per row, so the pairing costs nothing.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro

# Intrinsics/geometry both runs must agree on; the merged frames share a single camera model.
CAMERA_KEYS = ("camera_angle_x", "camera_angle_y", "cx", "cy", "fl_x", "fl_y", "h", "w", "aabb_scale")

# Per-frame keys taken from the excited run. Everything else in a frame comes from the spline
# run. "wrench"/"ft_sen" are what the loader reads as the measured wrench; "regressor" is unused
# by the loader but carried for the identification scripts in this repository.
DYNAMICS_KEYS = (
    "pose_sen_obj",
    "pose_sen_obji",
    "twist_sen",
    "dtwist_sen",
    "linacc_sen_obji",
    "wrench",
    "ft_sen",
    "regressor",
    "jointvars_clean",
)


def _primary_transforms(run_dir: Path) -> Path:
    """Return the root-level ``*.json`` a downstream loader would glob.

    Older runs left the per-split dumps unrenamed, so a bare ``transforms.json`` (the
    recorder's shipped series) wins when several are present.
    """
    candidates = sorted(run_dir.glob("*.json"))
    for preferred in ("transforms.json", "unperturbed_transforms.json"):
        if run_dir / preferred in candidates:
            return run_dir / preferred
    if len(candidates) != 1:
        raise SystemExit(f"{run_dir}: expected exactly one root *.json, found {[p.name for p in candidates]}")
    return candidates[0]


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:  # cross-device, or a filesystem without hard links
        shutil.copy(src, dst)


def _even_indices(total: int, count: int) -> list[int]:
    """Pick ``count`` indices spread evenly over ``range(total)``, endpoints included.

    Strictly increasing whenever ``count <= total``, which is the only way it is called.
    """
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def _pair_frame(index: int, image_frame: dict, dyn_frame: dict, spline_dir: Path, out_dir: Path) -> dict:
    """Stage the spline image/mask under a merged index and graft the excited dynamics onto it."""
    name = f"{index:04d}.png"
    src = spline_dir / image_frame["file_path"]
    if not src.is_file():
        raise SystemExit(f"missing image referenced by {spline_dir}: {src}")
    _link_or_copy(src, out_dir / "complete" / name)
    src_mask = spline_dir / "masks" / Path(image_frame["file_path"]).name
    if src_mask.is_file():
        _link_or_copy(src_mask, out_dir / "masks" / name)

    merged = {k: v for k, v in image_frame.items() if k not in DYNAMICS_KEYS}
    merged.update({k: v for k, v in dyn_frame.items() if k in DYNAMICS_KEYS})
    merged["file_path"] = f"complete/{name}"
    merged["image_source"] = "spline"
    merged["dynamics_source"] = "excited"
    return merged


def merge(spline_dir: Path, excited_dir: Path, out_dir: Path, force: bool = False) -> Path:
    """Write a merged dataset into ``out_dir`` and return the path of its transforms file."""
    metas = {}
    for tag, run_dir in (("spline", spline_dir), ("excited", excited_dir)):
        with open(_primary_transforms(run_dir)) as f:
            metas[tag] = json.load(f)

    mismatched = [k for k in CAMERA_KEYS if metas["spline"].get(k) != metas["excited"].get(k)]
    if mismatched and not force:
        raise SystemExit(
            f"camera/geometry keys differ between runs: {mismatched} (pass --force to take the spline run)"
        )

    spline_frames = metas["spline"]["frames"]
    excited_frames = metas["excited"]["frames"]
    n = min(len(spline_frames), len(excited_frames))
    if n == 0:
        raise SystemExit("one of the runs has no frames")
    missing = sorted((set(DYNAMICS_KEYS) & set(spline_frames[0])) - set(excited_frames[0]))
    if missing:
        raise SystemExit(f"excited run lacks dynamics keys present in the spline run: {missing}")
    # The shorter run is used whole; the longer one is subsampled evenly across its full span so
    # the merged rows still cover the entire trajectory instead of only its beginning.
    spline_idx = _even_indices(len(spline_frames), n)
    excited_idx = _even_indices(len(excited_frames), n)
    if len(spline_frames) != len(excited_frames):
        longer = "spline" if len(spline_frames) > n else "excited"
        picked = spline_idx if longer == "spline" else excited_idx
        print(
            f"warning: frame counts differ (spline {len(spline_frames)}, excited {len(excited_frames)}); "
            f"subsampling the {longer} run evenly to {n} frames (indices {picked[0]}..{picked[-1]})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "complete").mkdir(exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)

    merged_frames = [
        _pair_frame(i, spline_frames[spline_idx[i]], excited_frames[excited_idx[i]], spline_dir, out_dir)
        for i in range(n)
    ]

    out = {k: v for k, v in metas["spline"].items() if k != "frames"}
    # Identification outputs come from the excited run: that is the run designed to excite them.
    for key in ("ls", "tls"):
        if key in metas["excited"]:
            out[key] = metas["excited"][key]
    out["merge_sources"] = {
        "image": {
            "role": "image+camera_pose",
            "run_dir": str(spline_dir),
            "frames": len(spline_frames),
            "source_indices": spline_idx,
        },
        "dynamics": {
            "role": "dynamics",
            "run_dir": str(excited_dir),
            "frames": len(excited_frames),
            "source_indices": excited_idx,
        },
        "merged_frames": n,
        "subsampling": "even",  # indices spread evenly over the source run, endpoints included
        "dynamics_keys": [k for k in DYNAMICS_KEYS if k in merged_frames[0]],
    }
    out["frames"] = merged_frames
    with open(out_dir / "transforms.json", "w") as f:
        json.dump(out, f, indent=2)

    gt = spline_dir / "ground_truth.csv"
    if gt.is_file():
        shutil.copy(gt, out_dir / "ground_truth.csv")

    return out_dir / "transforms.json"


@dataclass
class MergeConfig:
    spline_dir: Path
    """Run directory supplying images and camera poses (spline trajectory)."""
    excited_dir: Path
    """Run directory supplying the dynamics rows (excited trajectory)."""
    out_dir: Path
    """Directory to write the merged dataset into."""
    force: bool = False
    """Merge even when the two runs disagree on camera intrinsics or aabb_scale."""


def main() -> None:
    cfg = tyro.cli(MergeConfig)
    path = merge(cfg.spline_dir, cfg.excited_dir, cfg.out_dir, cfg.force)
    with open(path) as f:
        meta = json.load(f)
    src = meta["merge_sources"]
    print(f"wrote {path} with {len(meta['frames'])} frames")
    print(f"  images+poses from {src['image']['run_dir']} ({src['image']['frames']} frames available)")
    print(f"  dynamics from    {src['dynamics']['run_dir']} ({src['dynamics']['frames']} frames available)")


if __name__ == "__main__":
    main()
