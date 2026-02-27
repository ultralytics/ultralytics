from pathlib import Path


def to_posix(path: Path) -> str:
    return path.as_posix()
