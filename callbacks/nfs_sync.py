"""Periodic rsync of local save_dir to a shared NFS mirror.

A daemon thread on ``RANK in (-1, 0)`` mirrors ``save_dir`` every ``interval_sec`` seconds, plus a final
sync when the returned ``stop`` is called. rsync runs out of process and its errors are swallowed, so
training is never interrupted. A save_dir already under the mirror root is skipped with a warning.

The caller passes ``save_dir`` because it owns it (``paths.run_paths`` mints it). Never start from a
trainer's ``__init__``: a DDP launcher builds a trainer it never trains, so it and the rank-0 child would
both mirror one save_dir. Runners start it directly, and under DDP the launcher is the right process
because it outlives the children writing into save_dir. From a trainer, use ``_setup_train``.

Usage:
    sync_stop = nfs_sync.start(save_dir)
    sync_stop()  # or: trainer.add_callback("on_train_end", sync_stop)
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from ultralytics.utils import LOGGER, RANK

from .paths import NFS_MIRROR_ROOT, SYNC_INTERVAL_SEC


def start(
    save_dir: str | Path,
    nfs_mirror_root: str | Path = NFS_MIRROR_ROOT,
    interval_sec: int = SYNC_INTERVAL_SEC,
    exclude: tuple[str, ...] = (),
):
    """Start a rank-0 daemon thread mirroring save_dir to NFS, and return its stop callable.

    Args:
        save_dir (str | Path): Local run directory to mirror.
        nfs_mirror_root (str | Path, optional): NFS directory that will contain the run dir. The save_dir basename is
            appended to form the final target.
        interval_sec (int, optional): Seconds between rsync passes.
        exclude (tuple[str, ...], optional): Rsync patterns to omit from the mirror.

    Returns:
        (Callable): Stops the thread and does a final sync. Always callable, including on non-zero ranks and when the
            mirror was skipped, so it is safe to register directly as an ``on_train_end`` callback.
    """
    stop_event = threading.Event()
    src = str(Path(save_dir).resolve())
    dst = str(Path(nfs_mirror_root) / Path(src).name)

    def _rsync() -> None:
        try:
            Path(dst).mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "--exclude=*.pt.tmp",
                    *(f"--exclude={x}" for x in exclude),
                    f"{src.rstrip('/')}/",
                    f"{dst.rstrip('/')}/",
                ],
                check=False,
                capture_output=True,
            )
        except Exception as e:
            LOGGER.warning(f"nfs_sync: rsync failed: {e}")

    def _loop() -> None:
        while not stop_event.wait(interval_sec):
            _rsync()

    def stop(_trainer=None) -> None:
        """Stop the mirror thread and do a final sync. Takes an ignored positional so it works as a callback."""
        if stop_event.is_set():
            return
        stop_event.set()
        _rsync()
        LOGGER.info(f"nfs_sync: final sync complete -> {dst}")

    if RANK not in (-1, 0):
        stop_event.set()  # pre-set so stop() is a no-op rather than a second return path
    elif src.startswith(str(Path(nfs_mirror_root).resolve())):
        stop_event.set()
        LOGGER.warning(
            f"nfs_sync: save_dir resolves under the NFS mirror root, skipping mirror "
            f"(training continues, but set project to a local path for decoupling): {src}"
        )
    else:
        threading.Thread(target=_loop, daemon=True, name="nfs-sync").start()
        LOGGER.info(f"nfs_sync: mirroring {src} -> {dst} every {interval_sec}s")
    return stop
