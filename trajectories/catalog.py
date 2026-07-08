import dataclasses
import json
from pathlib import Path

CATALOG_PATH = Path(__file__).parent.parent / "configurations" / "trajectories" / "catalog.jsonl"


def append_catalog_entry(
    timestamp: str,
    subcommand: str,
    output_dir: str,
    config,
    condition_number: float | None = None,
    catalog_path: Path = CATALOG_PATH,
) -> None:
    """Append one JSON-lines record describing a generated trajectory.

    Non-serializable config values (e.g. ``Path``) are stringified via
    ``json.dumps(default=str)`` so the write never fails on exotic field types.

    ``condition_number`` records the optimization result for excited trajectories;
    it is ``None`` for trajectory types without a condition number.
    """
    entry = {
        "timestamp": timestamp,
        "subcommand": subcommand,
        "output_dir": output_dir,
        "condition_number": condition_number,
        "config": dataclasses.asdict(config),
    }
    with open(catalog_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
