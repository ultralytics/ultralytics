# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from types import SimpleNamespace

import pytest

from ultralytics.models.yolo.detect import DetectionValidator


@pytest.mark.parametrize(
    ("root_name", "lvis_layout", "annotation", "expected_lvis"),
    [
        ("pelvis_project", False, None, False),
        ("sample_data", False, None, False),
        ("sample_data", True, None, True),
        ("sample_data", False, "lvis_v1_val.json", True),
        ("pelvis_project", False, "lvis_v1_val.json", True),
        ("sample_data", False, "lvis_v1_test.json", False),
    ],
)
def test_lvis_validation_path_recognition(tmp_path, root_name, lvis_layout, annotation, expected_lvis):
    """Only explicit LVIS layouts or matching split annotations should trigger LVIS evaluation."""
    root = tmp_path / root_name
    if annotation:
        annotations = root / "annotations"
        annotations.mkdir(parents=True)
        (annotations / annotation).touch()

    val_path = root / "lvis" / "val.txt" if lvis_layout else root / "images"
    val_path.parent.mkdir(parents=True, exist_ok=True)

    validator = DetectionValidator(
        args={"split": "val", "save_json": False, "plots": False},
        save_dir=tmp_path / "runs",
    )
    validator.training = False
    validator._check_max_det = lambda *args: None
    validator.data = {"path": root, "val": str(val_path)}
    validator.dataloader = SimpleNamespace(
        dataset=SimpleNamespace(im_files=[str(root / "images" / "example.jpg")]),
        sampler=[0],
    )

    validator.init_metrics(SimpleNamespace(names={0: "object"}))

    assert validator.is_coco is False
    assert validator.is_lvis is expected_lvis
    assert validator.args.save_json is expected_lvis
