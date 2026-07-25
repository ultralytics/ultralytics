"""Periodic rsync of local save_dir to a shared NFS mirror.

A daemon thread running on rank 0 mirrors ``trainer.save_dir`` to a target
directory on shared NFS every ``interval_sec`` seconds, and does a final
sync on training end. Non-blocking: rsync runs in a separate process and
errors are swallowed so training is never interrupted. If save_dir already
resolves under the NFS mirror root the sync is skipped with a warning.
"""

import subprocess
import threading
from pathlib import Path

from ultralytics.utils import LOGGER, RANK


def setup(nfs_mirror_root: str | Path, interval_sec: int = 600, exclude: tuple[str, ...] = ()):
    """Build (on_train_start, on_train_end) callbacks that mirror save_dir to NFS.

    Args:
        nfs_mirror_root (str | Path): NFS directory that will contain the run dir; the save_dir basename is appended to
            form the final target.
        interval_sec (int, optional): Seconds between rsync passes.
        exclude (tuple[str, ...], optional): Rsync patterns to omit from the mirror.

    Returns:
        (tuple): (on_train_start callback, on_train_end callback).
    """
    stop_event = threading.Event()
    state: dict = {}
    nfs_root = str(Path(nfs_mirror_root).resolve())

    def _rsync(src: str, dst: str) -> None:
        Path(dst).mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync",
            "-a",
            "--partial",
            "--exclude=*.pt.tmp",
            *(f"--exclude={x}" for x in exclude),
            f"{src.rstrip('/')}/",
            f"{dst.rstrip('/')}/",
        ]
        subprocess.run(cmd, check=False, capture_output=True)

    def _loop() -> None:
        while not stop_event.wait(interval_sec):
            try:
                _rsync(state["src"], state["dst"])
            except Exception as e:
                LOGGER.warning(f"nfs_sync: rsync failed, will retry next interval: {e}")

    def on_train_start(trainer) -> None:
        if RANK not in (-1, 0):
            return
        try:
            src = str(Path(trainer.save_dir).resolve())
            if src.startswith(nfs_root):
                LOGGER.warning(
                    f"nfs_sync: save_dir resolves under NFS mirror root; skipping mirror "
                    f"(training continues, but set project to a local path for decoupling): {src}"
                )
                return
            dst = str(Path(nfs_mirror_root) / Path(src).name)
            state.update(src=src, dst=dst)
            threading.Thread(target=_loop, daemon=True, name="nfs-sync").start()
            LOGGER.info(f"nfs_sync: mirroring {src} -> {dst} every {interval_sec}s")
        except Exception as e:
            LOGGER.warning(f"nfs_sync: setup failed, continuing without NFS mirror: {e}")

    def on_train_end(trainer) -> None:
        if RANK not in (-1, 0) or "src" not in state:
            return
        stop_event.set()
        try:
            _rsync(state["src"], state["dst"])
            LOGGER.info(f"nfs_sync: final sync complete -> {state['dst']}")
        except Exception as e:
            LOGGER.warning(f"nfs_sync: final sync failed: {e}")

    return on_train_start, on_train_end
