from pathlib import Path


def ensure_repo_exists(repo_root: Path) -> None:
    if not repo_root.exists():
        raise FileNotFoundError(f"repo root not found: {repo_root}")
