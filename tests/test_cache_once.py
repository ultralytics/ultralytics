import pytest
from torch.utils.data import DataLoader, get_worker_info
from ultralytics.data.base import BaseDataset

class DummyDataset(BaseDataset):
    def __init__(self, img_path, imgsz, batch, cache="ram", **kwargs):
        # Bypass actual file loading by overriding get_img_files
        super().__init__(img_path, imgsz, batch, cache=cache, **kwargs)
        # Counter for cache_images invocations
        self.cache_count = 0

    def get_img_files(self, img_path):
        # Return a dummy non-empty list to skip FileNotFoundError
        return ["dummy.jpg"]

    def cache_images(self):
        # Increment counter instead of performing real caching
        self.cache_count += 1
        return

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx


def test_cache_only_in_main():
    # Initialize DummyDataset with dummy img_path; get_img_files override prevents errors
    ds = DummyDataset(img_path=["dummy_path"], imgsz=32, batch=1, cache="ram")
    loader = DataLoader(ds, batch_size=2, num_workers=4)

    # Consume one epoch
    for batch in loader:
        pass

    # Ensure cache_images() was called exactly once (in the main process)
    assert ds.cache_count == 1, f"Expected cache_images() once, got {ds.cache_count}"
