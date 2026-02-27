from pathlib import Path
from typing import Any, Dict
import ast
import json


def _coerce_scalar(value: str) -> Any:
    v = value.strip().strip('"').strip("'")
    raw = value.strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            return parsed
        except Exception:
            try:
                return ast.literal_eval(raw)
            except Exception:
                pass
    if raw.startswith("{") and raw.endswith("}"):
        try:
            parsed = json.loads(raw)
            return parsed
        except Exception:
            try:
                return ast.literal_eval(raw)
            except Exception:
                pass
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Fallback parser that supports basic nested dict/list via indentation."""
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    lines = text.splitlines()

    def _next_child_is_list(current_index: int, current_indent: int) -> bool:
        for nxt in lines[current_index + 1 :]:
            if not nxt.strip() or nxt.lstrip().startswith("#"):
                continue
            ind = len(nxt) - len(nxt.lstrip(" "))
            if ind <= current_indent:
                return False
            return nxt.strip().startswith("- ")
        return False

    for i, raw in enumerate(lines):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                continue
            item = line[2:].strip()
            if not item:
                container: Any = {}
                parent.append(container)
                stack.append((indent, container))
                continue
            if ":" in item:
                k, v = item.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v == "":
                    container = {}
                    parent.append({k: container})
                    stack.append((indent, container))
                else:
                    parent.append({k: _coerce_scalar(v)})
                continue
            parent.append(_coerce_scalar(item))
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value == "":
            container: Any = [] if _next_child_is_list(i, indent) else {}
            if isinstance(parent, dict):
                parent[key] = container
            stack.append((indent, container))
        else:
            if isinstance(parent, dict):
                parent[key] = _coerce_scalar(value)

    return root


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return _parse_simple_yaml(raw)
