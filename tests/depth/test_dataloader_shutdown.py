"""Regression tests for graceful InfiniteDataLoader worker shutdown.

At interpreter exit, multiprocessing's ``_exit_function`` SIGTERMs any still-alive daemon child and
joins it; torch's SIGCHLD watchdog then raises "DataLoader worker (pid N) is killed by signal:
Terminated" for dataloader workers that were never unregistered — the noisy end-of-training atexit
traceback. ``close_dataloaders()`` is registered with atexit after multiprocessing's hook (LIFO, so
it runs first) and drains every persistent iterator gracefully, leaving nothing for multiprocessing
to SIGTERM.
"""

import torch
from torch.utils.data import Dataset

from ultralytics.data.build import InfiniteDataLoader, close_dataloaders


class _DS(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        return torch.zeros(4)


def _live_workers(dl):
    return [w for w in getattr(dl.iterator, "_workers", []) if w.is_alive()]


def test_close_dataloaders_drains_workers():
    """The exit hook joins all persistent-iterator workers so none are alive (or registered) at exit."""
    dl = InfiniteDataLoader(_DS(), batch_size=4, num_workers=2, shuffle=False)
    for _ in zip(range(2), dl):  # partial epoch: persistent iterator keeps workers alive, prefetching
        pass
    assert _live_workers(dl)
    close_dataloaders()
    assert not _live_workers(dl)
    close_dataloaders()  # idempotent: second call must not raise on already-drained iterators


def test_reset_shuts_down_previous_worker_generation():
    """reset() drains the old iterator's workers deterministically and the loader stays usable."""
    dl = InfiniteDataLoader(_DS(), batch_size=4, num_workers=2, shuffle=False)
    for _ in zip(range(2), dl):
        pass
    old = list(dl.iterator._workers)
    dl.reset()
    for w in old:
        w.join(timeout=10)
    assert not any(w.is_alive() for w in old)
    for _ in zip(range(2), dl):  # new iterator works after reset
        pass
    close_dataloaders()
