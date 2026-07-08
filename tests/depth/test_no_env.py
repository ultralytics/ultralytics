# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pathlib
import re


def test_no_depth_env_reads_in_source():
    """All DEPTH_* knobs are cfg args now; no os.environ reads of them may remain."""
    root = pathlib.Path(__file__).resolve().parents[2] / "ultralytics"
    offenders = []
    pattern = re.compile(r"(environ|getenv).*DEPTH_|DEPTH_.*(environ|getenv)")
    for path in root.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, "Found DEPTH_* env reads:\n" + "\n".join(offenders)
