"""Fixed camera-distance calibration from rendered foreground occupancy.

The simulator uses a camera attached to the last robot link.  Its local
``z`` position is therefore a single, object-specific constant for a whole
dataset.  This module contains the configuration, the renderer-backed
measurement code, and a renderer-independent bisection routine so that the
calibration policy can be tested without an OpenGL context.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco._structs import MjData, MjModel
from mujoco.renderer import Renderer

from utilities import get_element_id


@dataclass
class CameraCalibrationConfig:
    """Camera calibration and fixed-distance settings.

    ``mode='fixed'`` is deliberately the default so existing datasets keep
    their historical camera distance.  ``calibrate`` computes one fixed
    factor and optionally writes it to ``calibration_path``; ``load`` reads a
    previously computed factor and applies it without rendering calibration
    frames.
    """

    mode: str = "fixed"
    distance_factor: float = 4.0
    calibration_path: str | None = None
    target_foreground_ratio: float = 0.10
    min_distance_factor: float = 1.0
    max_distance_factor: float = 32.0
    distance_tolerance: float = 1.0e-3
    ratio_tolerance: float = 5.0e-3
    sample_count: int = 16
    border_margin: int = 2


@dataclass(frozen=True)
class ForegroundMeasurement:
    """Occupancy and crop status for one representative pose."""

    ratio: float
    touches_border: bool


@dataclass(frozen=True)
class CameraCalibrationResult:
    """Result of a fixed-distance calibration run."""

    distance_factor: float
    target_foreground_ratio: float
    measured_foreground_ratio: float
    min_foreground_ratio: float
    max_foreground_ratio: float
    sample_count: int
    border_margin: int
    distance_factor_min: float
    distance_factor_max: float
    distance_tolerance: float
    ratio_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def foreground_ratio(mask: np.ndarray) -> float:
    """Return the fraction of image pixels occupied by a binary mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"foreground mask must be a non-empty 2-D array, got shape {mask.shape}")
    return float(mask.mean())


def mask_touches_border(mask: np.ndarray, margin: int = 1) -> bool:
    """Whether a foreground mask reaches the requested image border margin.

    A margin of two rejects masks whose bounding box reaches either of the
    outermost two rows or columns.  Empty masks are not considered cropped;
    the calibration range checks separately reject an unusable target.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"foreground mask must be a non-empty 2-D array, got shape {mask.shape}")
    if margin < 0:
        raise ValueError(f"border margin must be non-negative, got {margin}")
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return False
    height, width = mask.shape
    return bool(
        ys.min() < margin
        or xs.min() < margin
        or ys.max() >= height - margin
        or xs.max() >= width - margin
    )


def _validate_config(config: CameraCalibrationConfig) -> None:
    if config.mode not in {"fixed", "calibrate", "load"}:
        raise ValueError(f"camera calibration mode must be 'fixed', 'calibrate', or 'load', got {config.mode!r}")
    if config.distance_factor <= 0:
        raise ValueError("distance_factor must be positive")
    if not 0 < config.target_foreground_ratio < 1:
        raise ValueError("target_foreground_ratio must lie strictly between zero and one")
    if config.min_distance_factor <= 0 or config.max_distance_factor <= 0:
        raise ValueError("distance search bounds must be positive")
    if config.min_distance_factor >= config.max_distance_factor:
        raise ValueError("min_distance_factor must be smaller than max_distance_factor")
    if config.distance_tolerance <= 0 or config.ratio_tolerance < 0:
        raise ValueError("distance_tolerance must be positive and ratio_tolerance non-negative")
    if config.sample_count < 1:
        raise ValueError("sample_count must be at least one")
    if config.border_margin < 0:
        raise ValueError("border_margin must be non-negative")


def _median_measurement(measurements: Sequence[ForegroundMeasurement]) -> float:
    if not measurements:
        raise ValueError("at least one foreground measurement is required")
    ratios = np.asarray([measurement.ratio for measurement in measurements], dtype=float)
    if not np.all(np.isfinite(ratios)) or np.any((ratios < 0) | (ratios > 1)):
        raise ValueError(f"foreground ratios must be finite values in [0, 1], got {ratios.tolist()}")
    return float(np.median(ratios))


def calibrate_distance(  # noqa: C901
    evaluate: Callable[[float], Sequence[ForegroundMeasurement]],
    config: CameraCalibrationConfig,
) -> CameraCalibrationResult:
    """Find a fixed distance factor whose median occupancy hits the target.

    ``evaluate(factor)`` must render the same representative poses for every
    candidate factor.  Since occupancy decreases with camera distance, the
    routine uses a bracketed bisection.  A candidate is only accepted when no
    representative mask touches ``border_margin``.  If the requested target
    is outside the safe search interval, a descriptive ``ValueError`` is
    raised instead of silently producing a cropped or badly scaled dataset.
    """

    _validate_config(config)

    low = float(config.min_distance_factor)
    high = float(config.max_distance_factor)
    low_measurements = list(evaluate(low))
    high_measurements = list(evaluate(high))
    low_ratio = _median_measurement(low_measurements)
    high_ratio = _median_measurement(high_measurements)
    low_safe = not any(m.touches_border for m in low_measurements)
    high_safe = not any(m.touches_border for m in high_measurements)

    # The search assumes the usual monotonic relation between distance and
    # visible area.  Give an actionable diagnostic when the range is wrong.
    if low_ratio < high_ratio:
        raise ValueError(
            "foreground occupancy is not monotone over the camera search range: "
            f"ratio({low:g})={low_ratio:.6g} < ratio({high:g})={high_ratio:.6g}"
        )
    if high_ratio > config.target_foreground_ratio + config.ratio_tolerance:
        raise ValueError(
            "target foreground ratio is too large for the safe search range: "
            f"ratio at max distance ({high:g}) is {high_ratio:.6g}, "
            f"target is {config.target_foreground_ratio:.6g}"
        )
    if low_safe and low_ratio < config.target_foreground_ratio - config.ratio_tolerance:
        raise ValueError(
            "target foreground ratio is too large for the safe search range: "
            f"ratio at min distance ({low:g}) is {low_ratio:.6g}, "
            f"target is {config.target_foreground_ratio:.6g}"
        )
    if not high_safe:
        raise ValueError(
            f"maximum camera distance ({high:g}) still crops at least one representative pose; "
            "increase max_distance_factor"
        )

    # If the near endpoint is already safe, it can be the desired result.  If
    # it is cropped, move the lower bracket out until it is safe; occupancy
    # remains above the target by construction in the useful case.
    if low_safe:
        safe_low = low
        safe_low_measurements = low_measurements
    else:
        safe_low = high
        safe_low_measurements = high_measurements
        for _ in range(64):
            candidate = 0.5 * (low + safe_low)
            measurements = list(evaluate(candidate))
            if any(m.touches_border for m in measurements):
                low = candidate
            else:
                safe_low = candidate
                safe_low_measurements = measurements
            if safe_low - low <= config.distance_tolerance:
                break

    safe_low_ratio = _median_measurement(safe_low_measurements)
    if safe_low_ratio < config.target_foreground_ratio - config.ratio_tolerance:
        raise ValueError(
            "no-crop constraint forces the object below the target foreground ratio: "
            f"nearest safe ratio is {safe_low_ratio:.6g}, target is {config.target_foreground_ratio:.6g}"
        )

    # Search between the nearest safe point (ratio >= target) and the far
    # endpoint (ratio <= target).  Keep the safe point as a candidate so the
    # final result can never be a cropped view.
    near = safe_low
    near_measurements = safe_low_measurements
    far = high
    far_measurements = high_measurements
    for _ in range(128):
        if far - near <= config.distance_tolerance:
            break
        candidate = 0.5 * (near + far)
        measurements = list(evaluate(candidate))
        ratio = _median_measurement(measurements)
        safe = not any(m.touches_border for m in measurements)
        if not safe or ratio > config.target_foreground_ratio:
            # A cropped or too-large view must be farther away.
            near = candidate
            near_measurements = measurements
        else:
            far = candidate
            far_measurements = measurements

    # Prefer the endpoint with the smaller target error.  Both endpoints are
    # safe; using the actual measured values avoids reporting an interpolated
    # value that was never rendered.
    near_ratio = _median_measurement(near_measurements)
    far_ratio = _median_measurement(far_measurements)
    if abs(near_ratio - config.target_foreground_ratio) <= abs(far_ratio - config.target_foreground_ratio):
        result_factor, result_measurements, result_ratio = near, near_measurements, near_ratio
    else:
        result_factor, result_measurements, result_ratio = far, far_measurements, far_ratio

    if abs(result_ratio - config.target_foreground_ratio) > config.ratio_tolerance:
        raise ValueError(
            "camera search did not reach target foreground ratio within tolerance: "
            f"measured {result_ratio:.6g}, target {config.target_foreground_ratio:.6g}, "
            f"tolerance {config.ratio_tolerance:.6g}"
        )
    if any(m.touches_border for m in result_measurements):
        raise RuntimeError("internal error: selected camera factor violates the no-crop constraint")

    ratios = [m.ratio for m in result_measurements]
    return CameraCalibrationResult(
        distance_factor=float(result_factor),
        target_foreground_ratio=float(config.target_foreground_ratio),
        measured_foreground_ratio=float(result_ratio),
        min_foreground_ratio=float(min(ratios)),
        max_foreground_ratio=float(max(ratios)),
        sample_count=len(result_measurements),
        border_margin=int(config.border_margin),
        distance_factor_min=float(config.min_distance_factor),
        distance_factor_max=float(config.max_distance_factor),
        distance_tolerance=float(config.distance_tolerance),
        ratio_tolerance=float(config.ratio_tolerance),
    )


def representative_qpos(
    trajectory_frames: Iterable[Any] | None,
    sample_count: int,
    default_qpos: np.ndarray,
    nq: int,
) -> list[np.ndarray]:
    """Select deterministic, evenly spaced qpos samples from trajectory frames."""

    if sample_count < 1:
        raise ValueError("sample_count must be at least one")
    if nq < 1:
        raise ValueError(f"nq must be positive, got {nq}")

    if trajectory_frames is None:
        frames: list[Any] = []
    else:
        frames = list(trajectory_frames)
    if not frames:
        qpos = np.asarray(default_qpos, dtype=float).reshape(-1)
        if len(qpos) < nq:
            raise ValueError(f"default qpos has length {len(qpos)}, but model requires nq={nq}")
        return [qpos[:nq].copy()]

    indices = np.linspace(0, len(frames) - 1, min(sample_count, len(frames)), dtype=int)
    samples: list[np.ndarray] = []
    for index in np.unique(indices):
        frame = np.asarray(frames[int(index)], dtype=float)
        # Trajectory frames are [qpos, qvel, qacc].  Accept a plain qpos row as
        # well so the calibration helper remains useful with simple fixtures.
        qpos = frame[0] if frame.ndim >= 2 else frame
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        if len(qpos) < nq:
            raise ValueError(f"trajectory qpos sample has length {len(qpos)}, but model requires nq={nq}")
        samples.append(qpos[:nq].copy())
    return samples


def target_geom_ids(model: MjModel) -> np.ndarray:
    """Return geom IDs belonging to ``target/object`` and its child bodies."""

    object_body_id = get_element_id(model, "body", "target/object")
    included_bodies = {object_body_id}
    for body_id in range(model.nbody):
        parent_id = int(model.body_parentid[body_id])
        if parent_id in included_bodies:
            included_bodies.add(body_id)
    body_ids = np.asarray(sorted(included_bodies), dtype=int)
    return np.flatnonzero(np.isin(np.asarray(model.geom_bodyid, dtype=int), body_ids))


def _segmentation_mask(segmentation: np.ndarray, geom_ids: np.ndarray) -> np.ndarray:
    """Extract target geom pixels from MuJoCo's (object id, object type) map."""

    if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
        raise ValueError(f"segmentation image must have shape (H, W, 2), got {segmentation.shape}")
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    return (segmentation[:, :, 1] == geom_type) & np.isin(segmentation[:, :, 0], geom_ids)


def calibrate_model_camera(
    model: MjModel,
    data: MjData,
    camera_id: int,
    aabb_scale: float,
    config: CameraCalibrationConfig,
    trajectory_frames: Iterable[Any] | None = None,
    height: int = 800,
    width: int = 800,
) -> CameraCalibrationResult:
    """Calibrate one compiled MuJoCo model using segmentation renders.

    The mask intentionally includes only target geoms.  This is different
    from a recorder-wide foreground/alpha mask; the calibration measures
    target occupancy, while representative qpos samples also expose
    occlusion or border intrusion from the rest of the robot scene.
    """

    _validate_config(config)
    if height < 1 or width < 1:
        raise ValueError(f"renderer dimensions must be positive, got height={height}, width={width}")

    if aabb_scale <= 0:
        raise ValueError(f"aabb_scale must be positive, got {aabb_scale}")
    if camera_id < 0 or camera_id >= model.ncam:
        raise ValueError(f"camera_id {camera_id} is outside [0, {model.ncam})")

    qpos_samples = representative_qpos(trajectory_frames, config.sample_count, data.qpos, model.nq)
    geom_ids = target_geom_ids(model)
    if len(geom_ids) == 0:
        raise ValueError("target/object has no geoms to measure in segmentation")

    renderer = Renderer(model, height=height, width=width)
    renderer.enable_segmentation_rendering()
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_qacc = data.qacc.copy()

    def evaluate(distance_factor: float) -> list[ForegroundMeasurement]:
        set_camera_distance(model, data, camera_id, aabb_scale, distance_factor)
        measurements: list[ForegroundMeasurement] = []
        for qpos in qpos_samples:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera_id)
            segmentation = renderer.render()
            mask = _segmentation_mask(segmentation, geom_ids)
            measurements.append(
                ForegroundMeasurement(
                    ratio=foreground_ratio(mask),
                    touches_border=mask_touches_border(mask, config.border_margin),
                )
            )
        return measurements

    try:
        return calibrate_distance(evaluate, config)
    finally:
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.qacc[:] = saved_qacc
        mujoco.mj_forward(model, data)
        renderer.close()


def set_camera_distance(
    model: MjModel,
    data: MjData,
    camera_id: int,
    aabb_scale: float,
    distance_factor: float,
) -> None:
    """Set the fixed local camera distance and refresh camera extrinsics."""

    if distance_factor <= 0:
        raise ValueError(f"distance_factor must be positive, got {distance_factor}")
    model.cam_pos[camera_id, :] = (0.0, 0.0, float(distance_factor) * float(aabb_scale))
    model.cam_pos0[camera_id, :] = model.cam_pos[camera_id, :]
    model.cam_poscom0[camera_id, :] = model.cam_pos[camera_id, :]
    mujoco.mj_forward(model, data)


def _object_key(object_path: str | Path) -> str:
    return Path(object_path).name


def save_calibration(
    path: str | Path,
    object_path: str | Path,
    aabb_scale: float,
    result: CameraCalibrationResult,
) -> None:
    """Persist a calibration record, preserving other object records."""

    path = Path(path)
    payload: dict[str, Any] = {}
    if path.is_file():
        with path.open() as file:
            loaded = json.load(file)
        if not isinstance(loaded, dict):
            raise ValueError(f"calibration file must contain a JSON object: {path}")
        payload = loaded
    records = payload.setdefault("objects", {})
    if not isinstance(records, dict):
        raise ValueError(f"calibration file 'objects' must be a JSON object: {path}")
    object_path = Path(object_path)
    records[_object_key(object_path)] = {
        "object": str(object_path),
        "aabb_scale": float(aabb_scale),
        **result.to_dict(),
    }
    payload["version"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def load_calibration(
    path: str | Path,
    object_path: str | Path,
    aabb_scale: float,
) -> CameraCalibrationResult:
    """Load and validate a previously saved object-specific calibration."""

    path = Path(path)
    with path.open() as file:
        payload = json.load(file)
    records = payload.get("objects") if isinstance(payload, dict) else None
    record = records.get(_object_key(object_path)) if isinstance(records, dict) else None
    if record is None:
        raise KeyError(f"no camera calibration for object {_object_key(object_path)!r} in {path}")
    if not np.isclose(float(record["aabb_scale"]), float(aabb_scale), rtol=1.0e-6, atol=1.0e-9):
        raise ValueError(
            f"camera calibration for {_object_key(object_path)!r} was generated with aabb_scale="
            f"{record['aabb_scale']}, but current object has {aabb_scale}"
        )
    fields = {
        "distance_factor",
        "target_foreground_ratio",
        "measured_foreground_ratio",
        "min_foreground_ratio",
        "max_foreground_ratio",
        "sample_count",
        "border_margin",
        "distance_factor_min",
        "distance_factor_max",
        "distance_tolerance",
        "ratio_tolerance",
    }
    missing = fields - record.keys()
    if missing:
        raise ValueError(f"camera calibration record in {path} is missing fields: {sorted(missing)}")
    return CameraCalibrationResult(**{field: record[field] for field in fields})


def resolve_distance_factor(
    config: CameraCalibrationConfig,
    object_path: str | Path,
    aabb_scale: float,
) -> float:
    """Resolve the fixed factor for non-calibrating model construction."""

    _validate_config(config)
    if config.mode == "load":
        if config.calibration_path is None:
            raise ValueError("camera calibration mode 'load' requires calibration_path")
        return load_calibration(config.calibration_path, object_path, aabb_scale).distance_factor
    return float(config.distance_factor)

