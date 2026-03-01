from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _is_experiment_dir(exp_dir: Path, marker_rules: Iterable[Iterable[str]]) -> bool:
    for rule in marker_rules:
        if all((exp_dir / marker).exists() for marker in rule):
            return True
    return False


def discover_experiments(runs_root: Path, marker_rules: list[list[str]] | None = None) -> list[Path]:
    """Recursively discover experiment directories by marker rules."""
    rules = marker_rules or [["results.csv", "args.yaml"]]
    exps = set()

    # Fast path for default marker.
    if rules == [["results.csv", "args.yaml"]]:
        for path in runs_root.rglob("results.csv"):
            exp_dir = path.parent
            if (exp_dir / "args.yaml").exists():
                exps.add(exp_dir)
        return sorted(exps)

    for directory in (p for p in runs_root.rglob("*") if p.is_dir()):
        if _is_experiment_dir(directory, rules):
            exps.add(directory)
    return sorted(exps)
