import dataclasses
import json
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "catalog.jsonl"


def append_catalog_entry(
    timestamp: str,
    subcommand: str,
    output_dir: str,
    config,
    catalog_path: Path = CATALOG_PATH,
) -> None:
    """Append one JSON-lines record describing a generated trajectory.

    Non-serializable config values (e.g. ``Path``) are stringified via
    ``json.dumps(default=str)`` so the write never fails on exotic field types.
    """
    entry = {
        "timestamp": timestamp,
        "subcommand": subcommand,
        "output_dir": output_dir,
        "config": dataclasses.asdict(config),
    }
    with open(catalog_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
