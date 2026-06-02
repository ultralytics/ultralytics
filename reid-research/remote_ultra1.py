#!/usr/bin/env python
"""ssh/scp/rsync wrapper for the ultra1 remote ReID training box.

Connection details come from `~/.ssh/config` (alias `ultra1`) — including any ProxyJump
that paramiko-based remote.py / remote2.py cannot honor. No host/port/user/key encoded here.

Usage:
  python remote_ultra1.py run   "<shell command>"      run via `ssh ultra1 -- bash -lc ...`, stream stdout/stderr
  python remote_ultra1.py put   <local> <remote>        scp local -> ultra1:remote
  python remote_ultra1.py get   <remote> <local>        scp ultra1:remote -> local
  python remote_ultra1.py rsync <local> <remote>        rsync -avz --progress local -> ultra1:remote
  python remote_ultra1.py pull  <remote> <local>        rsync -avz --progress ultra1:remote -> local
"""

from __future__ import annotations

import shlex
import subprocess
import sys

HOST = "ultra1"  # alias resolved via ~/.ssh/config (ProxyJump etc. honored automatically)


def run(cmd: str) -> int:
    """Run `cmd` on ultra1 inside `bash -lc` so conda / yolo / .venv are on PATH."""
    full = ["ssh", HOST, "--", "bash", "-lc", cmd]
    print(f"$ {' '.join(shlex.quote(x) for x in full)}", flush=True)
    return subprocess.run(full).returncode


def put(local: str, remote: str) -> int:
    full = ["scp", "-r", local, f"{HOST}:{remote}"]
    print(f"$ {' '.join(shlex.quote(x) for x in full)}", flush=True)
    return subprocess.run(full).returncode


def get(remote: str, local: str) -> int:
    full = ["scp", "-r", f"{HOST}:{remote}", local]
    print(f"$ {' '.join(shlex.quote(x) for x in full)}", flush=True)
    return subprocess.run(full).returncode


def rsync_push(local: str, remote: str) -> int:
    full = ["rsync", "-avz", "--progress", local, f"{HOST}:{remote}"]
    print(f"$ {' '.join(shlex.quote(x) for x in full)}", flush=True)
    return subprocess.run(full).returncode


def rsync_pull(remote: str, local: str) -> int:
    full = ["rsync", "-avz", "--progress", f"{HOST}:{remote}", local]
    print(f"$ {' '.join(shlex.quote(x) for x in full)}", flush=True)
    return subprocess.run(full).returncode


ACTIONS = {
    "run": lambda a: run(a[2]),
    "put": lambda a: put(a[2], a[3]),
    "get": lambda a: get(a[2], a[3]),
    "rsync": lambda a: rsync_push(a[2], a[3]),
    "pull": lambda a: rsync_pull(a[2], a[3]),
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ACTIONS:
        print(__doc__)
        sys.exit(2)
    sys.exit(ACTIONS[sys.argv[1]](sys.argv))
