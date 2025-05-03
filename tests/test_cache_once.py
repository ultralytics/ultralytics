from torch.utils.data import DataLoader

from ultralytics.data.base import BaseDataset


def identity_transform(x):
    return x


class DummyDataset(BaseDataset):
    def __init__(self, img_path, imgsz, batch, cache="ram", **kwargs):
        self.cache_count = 0
        super().__init__(img_path, imgsz, batch, cache, **kwargs)

    def get_img_files(self, img_path):
        return ["dummy.jpg"]

    def get_labels(self):
        return []

    def check_cache_ram(self, safety_margin=0.5):
        return True

    def check_cache_disk(self):
        return True

    def cache_images(self):
        self.cache_count += 1
        return

    def build_transforms(self, hyp=None):
        return identity_transform

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx


def test_cache_only_in_main():
    ds = DummyDataset(img_path=["dummy_path"], imgsz=32, batch=1, cache="ram")
    loader = DataLoader(ds, batch_size=2, num_workers=4)

    for _ in loader:
        pass

    assert ds.cache_count == 1, f"Expected cache_images() once, got {ds.cache_count}"
