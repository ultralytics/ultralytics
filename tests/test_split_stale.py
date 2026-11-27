# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from ultralytics.data.split import split_classify_dataset


def test_split_classify_dataset_removes_stale_images(tmp_path):
    source = tmp_path / "dataset"
    class_dir = source / "cat"
    class_dir.mkdir(parents=True)
    for i in range(4):
        (class_dir / f"img{i}.jpg").touch()

    split = split_classify_dataset(source, train_ratio=0.5)
    assert len(list(split.rglob("*.jpg"))) == 4

    (class_dir / "img0.jpg").unlink()
    split_classify_dataset(source, train_ratio=0.5)
    assert len(list(split.rglob("*.jpg"))) == 3
