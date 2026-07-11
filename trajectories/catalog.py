import json
from pathlib import Path

CATALOG_PATH = Path(__file__).parent.parent / "configurations" / "trajectories" / "catalog.json"


def append_catalog_entry(
    timestamp: str,
    subcommand: str,
    output_dir: str,
    config,
    condition_number: float | None = None,
    catalog_path: Path = CATALOG_PATH,
) -> None:
    """Append a slim index entry to the catalog (pretty-printed JSON array).

    The full generation config is no longer duplicated here: each run's own
    ``trajectory.json`` embeds it under ``metadata.generation_config``, so the
    catalog only needs enough fields to locate and distinguish runs. The legacy
    ``catalog.jsonl`` is left untouched as the sole record of full configs for
    runs generated before this change.
    """
    yaml_path = getattr(config, "config", None)
    entry = {
        "timestamp": timestamp,
        "subcommand": subcommand,
        "output_dir": output_dir,
        "condition_number": condition_number,
        "config": str(yaml_path) if yaml_path is not None else None,
        "objective_type": getattr(config, "objective_type", None),
        "num_harmonics": getattr(config, "num_harmonics", None),
        "base_freq": getattr(config, "base_freq", None),
    }

    entries = []
    if catalog_path.exists():
        text = catalog_path.read_text()
        if text.strip():
            entries = json.loads(text)
    entries.append(entry)

    with open(catalog_path, "w") as f:
        json.dump(entries, f, indent=2, default=str)
