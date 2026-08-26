import csv
import json

from recorders.export_dynamics_csv import export


def _frame(i):
    return {
        "file_path": f"complete/{i:04d}.png",
        "pose_sen_obj": [[100 * i + 4 * r + c for c in range(4)] for r in range(4)],
        "twist_sen": [1000 * i + j for j in range(6)],
        "dtwist_sen": [2000 * i + j for j in range(6)],
        "wrench": [3000 * i + j for j in range(6)],
        "regressor": [[4000 * i + 10 * r + c for c in range(10)] for r in range(6)],
    }


def _write(root, meta_extra):
    root.mkdir(parents=True, exist_ok=True)
    meta = {"frames": [_frame(i) for i in range(3)], **meta_extra}
    (root / "transforms.json").write_text(json.dumps(meta))
    return root


def _read(path):
    with open(path) as f:
        return list(csv.reader(f))


def test_merged_dataset_emits_source_index_and_no_regressor(tmp_path):
    root = _write(tmp_path / "merged", {"merge_sources": {"dynamics": {"source_indices": [0, 2, 4]}}})
    rows = _read(export(root))
    assert rows[0][:2] == ["frame", "source_index"]
    assert len(rows) == 4
    assert len(rows[0]) == 2 + 16 + 6 + 6 + 6
    assert not any(c.startswith("regressor") for c in rows[0])
    assert rows[2][:2] == ["1", "2"]
    # pose flattened row-major, then twist/dtwist/wrench for frame 1
    assert [float(x) for x in rows[2][2:18]] == [float(100 + v) for v in range(16)]
    assert [float(x) for x in rows[2][18:24]] == [1000.0 + j for j in range(6)]
    assert [float(x) for x in rows[2][30:36]] == [3000.0 + j for j in range(6)]


def test_single_run_without_merge_sources_and_with_regressor(tmp_path):
    root = _write(tmp_path / "single", {})
    rows = _read(export(root, include_regressor=True))
    assert rows[0][0] == "frame"
    assert "source_index" not in rows[0]
    assert len(rows[0]) == 1 + 16 + 6 + 6 + 6 + 60
    assert [float(x) for x in rows[1][-60:]] == [float(10 * r + c) for r in range(6) for c in range(10)]


def test_missing_keys_are_not_emitted(tmp_path):
    meta = {"frames": [{k: v for k, v in _frame(0).items() if k != "wrench"}]}
    root = tmp_path / "partial"
    root.mkdir()
    (root / "transforms.json").write_text(json.dumps(meta))
    header = _read(export(root))[0]
    assert not any(c.startswith("wrench") for c in header)
    assert len(header) == 16 + 6 + 6 + 1
