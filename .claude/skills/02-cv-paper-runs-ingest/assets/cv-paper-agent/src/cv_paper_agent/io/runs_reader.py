from pathlib import Path


def index_artifacts(exp_dir: Path, runs_root: Path) -> list[dict]:
    items = []
    for p in exp_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            items.append(
                {
                    "path": p.relative_to(runs_root).as_posix(),
                    "size": p.stat().st_size,
                }
            )
    return sorted(items, key=lambda x: x["path"])
