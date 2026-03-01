from pathlib import Path

from .hashing import sha256_text
from .posix_path import to_posix


def _file_sig(path: Path) -> str:
    if not path.exists():
        return "missing"
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def experiment_fingerprint(exp_dir: Path, runs_root: Path) -> str:
    rel = to_posix(exp_dir.relative_to(runs_root))
    token = "|".join(
        [
            rel,
            _file_sig(exp_dir / "results.csv"),
            _file_sig(exp_dir / "args.yaml"),
        ]
    )
    return sha256_text(token)
