from recorders.standard_recorder import split_file_name

SPLITS = ["", "_train", "_valid", "_test"]


def _written(primary: str, prefixes: list[str]) -> list[str]:
    return [split_file_name(p, s, primary) for p in prefixes for s in SPLITS]


def test_exactly_one_bare_json_and_it_is_the_primary() -> None:
    """Downstream globs "*.json"; more than one hit makes it pick the wrong series."""
    for primary in ("transforms", "unperturbed_transforms"):
        names = _written(primary, ["transforms", "unperturbed_transforms"])
        bare = [n for n in names if not n.endswith(".bak")]
        assert bare == [f"{primary}.json"], (primary, bare)


def test_perturbed_only_run_still_ships_a_json() -> None:
    """With no noise-free series, the perturbed one stays the shipped file."""
    bare = [n for n in _written("transforms", ["transforms"]) if not n.endswith(".bak")]
    assert bare == ["transforms.json"]
