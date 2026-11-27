# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json


def test_convert_coco_keeps_duplicate_annotations(tmp_path):
    """Duplicate COCO annotations for the same box must survive conversion as separate label rows."""
    annotations = [
        {"id": i, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0} for i in (1, 2)
    ]
    data = {
        "images": [{"id": 1, "file_name": "image.jpg", "width": 100, "height": 100}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    (tmp_path / "instances_train.json").write_text(json.dumps(data), encoding="utf-8")

    from ultralytics.data.converter import convert_coco

    save_dir = tmp_path / "labels"
    convert_coco(labels_dir=tmp_path, save_dir=save_dir, cls91to80=False)
    lines = (save_dir / "labels" / "train" / "image.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["0 0.05 0.05 0.1 0.1"] * 2
