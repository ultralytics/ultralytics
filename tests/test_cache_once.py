from torch.utils.data import DataLoader

from ultralytics.data.base import BaseDataset


def identity_transform(x):
    """Return the input unchanged."""
    return x


class DummyDataset(BaseDataset):
    """A dummy dataset that only caches once in the main process."""

    def __init__(self, img_path, imgsz, batch, cache="ram", **kwargs):
        """Initialize with paths, image size, batch size, and cache type."""
        self.cache_count = 0
        super().__init__(img_path, imgsz, batch, cache, **kwargs)

    def get_img_files(self, img_path):
        """Return the list of image filenames for testing."""
        return ["dummy.jpg"]

    def get_labels(self):
        """Return an empty list of labels."""
        return []

    def check_cache_ram(self, safety_margin=0.5):
        """Always return True to simulate available RAM cache."""
        return True

    def check_cache_disk(self):
        """Always return True to simulate available disk cache."""
        return True

    def cache_images(self):
        """Simulate caching images by incrementing the cache counter."""
        self.cache_count += 1

    def build_transforms(self, hyp=None):
        """Return the identity transform function."""
        return identity_transform

    def __len__(self):
        """Return a fixed dataset length of 10."""
        return 10

    def __getitem__(self, idx):
        """Return the index as the “data” for testing."""
        return idx


def test_cache_only_in_main():
    """Test that cache_images() is called exactly once in the main process."""
    ds = DummyDataset(img_path=["dummy_path"], imgsz=32, batch=1, cache="ram")
    loader = DataLoader(ds, batch_size=2, num_workers=4)

    for _ in loader:
        pass

    assert ds.cache_count == 1, f"Expected cache_images() once, got {ds.cache_count}"
