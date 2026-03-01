from dataclasses import dataclass
from typing import Any


@dataclass
class Experiment:
    exp_id: str
    rel_path: str
    fingerprint: str
    args: dict[str, Any]
    best_metrics: dict[str, Any]
    artifacts: list[dict[str, Any]]
