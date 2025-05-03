import numpy as np
from torch.utils.data import DataLoader, get_worker_info

from ultralytics.data.base import BaseDataset


def identity_transform(x):
    """A no-op transform for testing."""
    return x


class AccessDataset(BaseDataset):
    """Dataset that fills and verifies RAM cache across workers."""

    def __init__(self, img_path, imgsz, cache="ram", **kwargs):
        """Initialize dataset and populate RAM cache for testing."""
        super().__init__(img_path, imgsz, cache=cache, **kwargs)
        super().__init__(img_path, imgsz, cache=cache, **kwargs)

    def get_img_files(self, img_path):
        """Return a dummy image list to bypass filesystem lookups."""
        return ["dummy.jpg"]

    def get_labels(self):
        """Return a dummy label list to bypass label parsing."""
        return [(0, 0, 0)]

    def check_cache_ram(self, safety_margin=0.5):
        """Always indicate that RAM caching is allowed."""
        return True

    def cache_images(self):
        """Pre-populate self.ims with a known pattern."""
        import numpy as _np

        self.ims = [_np.arange(5) for _ in range(self.ni)]

    def build_transforms(self, hyp=None):
        """Return a top-level no-op transform to ensure picklability."""
        return identity_transform

    def __len__(self):
        """Return dataset size based on number of images."""
        return self.ni

    def __getitem__(self, idx):
        """Verify the RAM cache is present in each worker."""
        worker = get_worker_info()
        assert hasattr(self, "ims"), f"Cache not initialized in worker {worker}"
        assert self.ims[idx] is not None, f"Worker {worker} saw no RAM cache at idx {idx}"
        return self.ims[idx]


def test_ram_cache_accessible_in_workers():
    """Ensure that RAM-cached images are accessible in each DataLoader worker."""
    ds = AccessDataset(img_path=["dummy"], imgsz=32, cache="ram")
    loader = DataLoader(ds, batch_size=2, num_workers=2)

    for batch in loader:
        for arr in batch:
            assert (arr == np.arange(5)).all(), "Cached array contents mismatch"
