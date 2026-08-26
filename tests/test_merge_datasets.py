import json

import numpy as np
import pytest

from recorders.merge import merge


def _make_run(root, tag, n, **overrides):
    root.mkdir(parents=True)
    (root / "complete").mkdir()
    (root / "masks").mkdir()
    frames = []
    for i in range(n):
        name = f"{i:04d}.png"
        (root / "complete" / name).write_text(f"{tag}-img-{i}")
        (root / "masks" / name).write_text(f"{tag}-mask-{i}")
        frames.append(
            {
                "file_path": f"complete/{name}",
                "transform_matrix": [[f"{tag}-pose-{i}"] * 4] * 4,
                "pose_sen_obj": [[f"{tag}-sen-{i}"] * 4] * 4,
                "twist_sen": [f"{tag}-twist-{i}"] * 6,
                "dtwist_sen": [f"{tag}-dtwist-{i}"] * 6,
                "wrench": [f"{tag}-wrench-{i}"] * 6,
                "regressor": [[f"{tag}-reg-{i}"] * 10] * 6,
            }
        )
    meta = {"cx": 400.0, "cy": 400.0, "h": 800, "w": 800, "aabb_scale": 0.2, "ls": [1.0], "tls": [2.0]}
    meta.update(overrides)
    meta["frames"] = frames
    (root / "transforms.json").write_text(json.dumps(meta))
    return root


def test_merge_splits_roles_between_runs(tmp_path):
    spline = _make_run(tmp_path / "spline", "spline", 3)
    excited = _make_run(tmp_path / "excited", "excited", 3, ls=[9.0], tls=[8.0])
    out = tmp_path / "merged"

    merge(spline, excited, out)
    meta = json.loads((out / "transforms.json").read_text())

    assert len(meta["frames"]) == 3
    frame = meta["frames"][1]
    # image side comes from the spline run
    assert frame["file_path"] == "complete/0001.png"
    assert frame["transform_matrix"] == [["spline-pose-1"] * 4] * 4
    assert (out / "complete" / "0001.png").read_text() == "spline-img-1"
    assert (out / "masks" / "0001.png").read_text() == "spline-mask-1"
    # dynamics side comes from the excited run
    assert frame["pose_sen_obj"] == [["excited-sen-1"] * 4] * 4
    assert frame["twist_sen"] == ["excited-twist-1"] * 6
    assert frame["dtwist_sen"] == ["excited-dtwist-1"] * 6
    assert frame["wrench"] == ["excited-wrench-1"] * 6
    assert frame["regressor"] == [["excited-reg-1"] * 10] * 6
    assert frame["image_source"] == "spline" and frame["dynamics_source"] == "excited"
    # identification outputs come from the excited run
    assert meta["ls"] == [9.0] and meta["tls"] == [8.0]
    assert meta["merge_sources"]["image"]["run_dir"] == str(spline)
    assert meta["merge_sources"]["dynamics"]["run_dir"] == str(excited)
    assert meta["merge_sources"]["merged_frames"] == 3
    assert meta["merge_sources"]["dynamics"]["source_indices"] == [0, 1, 2]


def test_longer_run_is_subsampled_evenly_over_its_whole_span(tmp_path):
    spline = _make_run(tmp_path / "spline", "spline", 3)
    excited = _make_run(tmp_path / "excited", "excited", 9)
    out = tmp_path / "merged"
    merge(spline, excited, out)
    meta = json.loads((out / "transforms.json").read_text())

    assert len(meta["frames"]) == 3
    src = meta["merge_sources"]
    assert src["subsampling"] == "even"
    assert src["image"]["source_indices"] == [0, 1, 2]  # shorter run used whole
    assert src["dynamics"]["source_indices"] == [0, 4, 8]  # both endpoints, even spacing
    # the dynamics payload really comes from those source rows
    assert [f["wrench"][0] for f in meta["frames"]] == ["excited-wrench-0", "excited-wrench-4", "excited-wrench-8"]


def test_subsampling_covers_the_span_for_uneven_ratios(tmp_path):
    spline = _make_run(tmp_path / "spline", "spline", 4)
    excited = _make_run(tmp_path / "excited", "excited", 10)
    out = tmp_path / "merged"
    merge(spline, excited, out)
    idx = json.loads((out / "transforms.json").read_text())["merge_sources"]["dynamics"]["source_indices"]

    assert len(idx) == 4
    assert idx[0] == 0 and idx[-1] == 9  # spans the full run
    assert idx == sorted(idx) and len(set(idx)) == len(idx)  # increasing, no repeats
    gaps = [b - a for a, b in zip(idx, idx[1:], strict=False)]
    assert max(gaps) - min(gaps) <= 1  # evenly spaced


def test_merge_rejects_camera_mismatch(tmp_path):
    spline = _make_run(tmp_path / "spline", "spline", 1)
    excited = _make_run(tmp_path / "excited", "excited", 1, w=640)
    with pytest.raises(SystemExit, match="w"):
        merge(spline, excited, tmp_path / "merged")
    merge(spline, excited, tmp_path / "merged2", force=True)  # opt-out works


def test_merged_dataset_looks_like_a_single_run(tmp_path):
    spline = _make_run(tmp_path / "spline", "spline", 3)
    excited = _make_run(tmp_path / "excited", "excited", 3)
    out = tmp_path / "merged"
    merge(spline, excited, out)

    # mimic the downstream loader: one root *.json, every file_path resolvable from the root
    roots = sorted(out.glob("*.json"))
    assert len(roots) == 1
    meta = json.loads(roots[0].read_text())
    assert all((out / f["file_path"]).is_file() for f in meta["frames"])
    assert np.array([f["wrench"] for f in meta["frames"]]).shape == (3, 6)
