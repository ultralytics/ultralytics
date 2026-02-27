from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Experiment:
    exp_id: str
    rel_path: str
    fingerprint: str
    args: Dict[str, Any]
    best_metrics: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
