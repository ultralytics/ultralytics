import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


def _to_number_if_possible(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return value


def _numeric_score(value: Any) -> Optional[float]:
    parsed = _to_number_if_possible(value)
    if isinstance(parsed, (int, float)):
        return float(parsed)
    return None


def _parse_with_stdlib(csv_path: Path, candidates: List[str]) -> Dict[str, Any]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    chosen_col: Optional[str] = next((c for c in candidates if c in rows[0]), None)
    if chosen_col:
        scored_rows = [(r, _numeric_score(r.get(chosen_col))) for r in rows]
        valid_rows = [item for item in scored_rows if item[1] is not None]
        if valid_rows:
            best_row = max(valid_rows, key=lambda item: item[1])[0]
        else:
            best_row = rows[-1]
    else:
        best_row = rows[-1]
    return {str(k): _to_number_if_possible(v) for k, v in best_row.items()}


def parse_best_metrics(csv_path: Path, schema: Dict[str, Any] | Any) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        schema = {}
    selection = schema.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    candidates: List[str] = selection.get("best_by", [])
    if not isinstance(candidates, list):
        candidates = []
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path)
        if df.empty:
            return {}
        chosen_col = next((c for c in candidates if c in df.columns), None)
        if chosen_col:
            row = df.loc[df[chosen_col].idxmax()]
        else:
            row = df.iloc[-1]
        return {str(k): (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
    except Exception:
        return _parse_with_stdlib(csv_path, candidates)
