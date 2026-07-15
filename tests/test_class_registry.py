# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
import pytest
import yaml
import numpy as np
from ultralytics.data.class_registry import ClassRegistry
from ultralytics import YOLO


def test_class_registry_logic():
    """Test semantic class resolution logic and mapping arrays."""
    # 1. Identical class orders
    da1 = {"names": {0: "blood", 1: "stool"}, "nc": 2}
    da2 = {"names": {0: "blood", 1: "stool"}, "nc": 2}
    reg = ClassRegistry([da1, da2])
    assert reg.global_names == {0: "blood", 1: "stool"}
    assert reg.global_nc == 2
    assert reg.is_identity(0)
    assert reg.is_identity(1)
    assert np.array_equal(reg.get_remap(0), [0, 1])
    assert np.array_equal(reg.get_remap(1), [0, 1])

    # 2. Swapped class orders
    db1 = {"names": {0: "blood", 1: "stool"}, "nc": 2}
    db2 = {"names": {0: "stool", 1: "blood"}, "nc": 2}
    reg = ClassRegistry([db1, db2])
    assert reg.global_names == {0: "blood", 1: "stool"}
    assert reg.global_nc == 2
    assert reg.is_identity(0)
    assert not reg.is_identity(1)
    assert np.array_equal(reg.get_remap(0), [0, 1])
    assert np.array_equal(reg.get_remap(1), [1, 0])

    # 3. Partially overlapping classes
    dc1 = {"names": {0: "car", 1: "truck"}, "nc": 2}
    dc2 = {"names": {0: "car", 1: "bus"}, "nc": 2}
    reg = ClassRegistry([dc1, dc2])
    assert reg.global_names == {0: "car", 1: "truck", 2: "bus"}
    assert reg.global_nc == 3
    assert reg.is_identity(0)
    assert not reg.is_identity(1)
    assert np.array_equal(reg.get_remap(0), [0, 1])
    assert np.array_equal(reg.get_remap(1), [0, 2])

    # 4. Missing / Disjoint classes
    dd1 = {"names": {0: "cat"}, "nc": 1}
    dd2 = {"names": {0: "dog"}, "nc": 1}
    reg = ClassRegistry([dd1, dd2])
    assert reg.global_names == {0: "cat", 1: "dog"}
    assert reg.global_nc == 2
    assert reg.is_identity(0)
    assert not reg.is_identity(1)
    assert np.array_equal(reg.get_remap(0), [0])
    assert np.array_equal(reg.get_remap(1), [1])

    # 5. User-defined global names logic
    de1 = {"names": {0: "blood", 1: "stool"}, "nc": 2}
    de2 = {"names": {0: "stool", 1: "blood"}, "nc": 2}
    global_names = {0: "stool", 1: "blood", 2: "cells"}
    reg = ClassRegistry([de1, de2], global_names=global_names)
    assert reg.global_names == global_names
    assert reg.global_nc == 3
    assert np.array_equal(reg.get_remap(0), [1, 0])  # blood (0) -> 1, stool (1) -> 0
    assert np.array_equal(reg.get_remap(1), [0, 1])  # stool (0) -> 0, blood (1) -> 1


def test_class_registry_validation_errors():
    """Test validation errors in global and local class definitions."""
    # 1. Duplicate local names
    d1 = {"names": {0: "blood", 1: "blood"}, "nc": 2}
    d2 = {"names": {0: "stool"}, "nc": 1}
    with pytest.raises(ValueError, match="Duplicate class name 'blood'"):
        ClassRegistry([d1, d2])

    # 2. Duplicate global names
    d3 = {"names": {0: "blood", 1: "stool"}, "nc": 2}
    with pytest.raises(ValueError, match="Duplicate class name 'blood'"):
        ClassRegistry([d3], global_names={0: "blood", 1: "blood"})

    # 3. Class missing from user-defined global names
    d4 = {"names": {0: "blood", 1: "stool"}, "nc": 2}
    with pytest.raises(ValueError, match="not present in the user-defined global"):
        ClassRegistry([d4], global_names={0: "blood"})


def test_multi_dataset_training(tmp_path):
    """Test end-to-end training with multi-dataset configuration and class ID remapping."""
    from ultralytics.utils import ASSETS

    # Setup Dataset A
    path_a = tmp_path / "dataset_a"
    path_a.mkdir()
    (path_a / "images" / "train").mkdir(parents=True)
    (path_a / "images" / "val").mkdir(parents=True)
    (path_a / "labels" / "train").mkdir(parents=True)
    (path_a / "labels" / "val").mkdir(parents=True)

    shutil.copy(ASSETS / "bus.jpg", path_a / "images" / "train" / "img1.jpg")
    shutil.copy(ASSETS / "bus.jpg", path_a / "images" / "val" / "img1.jpg")

    with open(path_a / "labels" / "train" / "img1.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    with open(path_a / "labels" / "val" / "img1.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    yaml_a = {
        "path": str(path_a),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "blood", 1: "stool"},
        "nc": 2,
    }
    with open(path_a / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_a, f)

    # Setup Dataset B
    path_b = tmp_path / "dataset_b"
    path_b.mkdir()
    (path_b / "images" / "train").mkdir(parents=True)
    (path_b / "images" / "val").mkdir(parents=True)
    (path_b / "labels" / "train").mkdir(parents=True)
    (path_b / "labels" / "val").mkdir(parents=True)

    shutil.copy(ASSETS / "bus.jpg", path_b / "images" / "train" / "img2.jpg")
    shutil.copy(ASSETS / "bus.jpg", path_b / "images" / "val" / "img2.jpg")

    with open(path_b / "labels" / "train" / "img2.txt", "w") as f:
        f.write("0 0.4 0.4 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")
    with open(path_b / "labels" / "val" / "img2.txt", "w") as f:
        f.write("1 0.2 0.2 0.1 0.1\n")

    yaml_b = {
        "path": str(path_b),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "stool", 1: "blood"},
        "nc": 2,
    }
    with open(path_b / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_b, f)

    # Setup combined YAML
    combined_yaml = {
        "datasets": [
            str(path_a / "data.yaml"),
            str(path_b / "data.yaml"),
        ]
    }
    combined_path = tmp_path / "combined.yaml"
    with open(combined_path, "w") as f:
        yaml.safe_dump(combined_yaml, f)

    # Run training for 1 epoch using the combined dataset
    model = YOLO("yolo26n.yaml")
    model.train(
        data=str(combined_path),
        epochs=1,
        imgsz=32,
        device="cpu",
        save=False,
        plots=False,
        project=str(tmp_path / "runs"),
        name="test_multi_dataset",
    )

    # Verify trainer and resolved class mappings
    trainer = model.trainer
    assert trainer is not None
    assert model.model.names == {0: "blood", 1: "stool"}
    assert model.model.nc == 2

    # Verify labels in loaded dataset
    concat_dataset = trainer.train_loader.dataset
    assert len(concat_dataset.datasets) == 2

    ds_a, ds_b = concat_dataset.datasets
    assert ds_a.labels[0]["cls"].flatten().tolist() == [0.0, 1.0]
    assert ds_b.labels[0]["cls"].flatten().tolist() == [1.0, 0.0]


def test_multi_dataset_segmentation_training(tmp_path):
    """Test end-to-end training with multi-dataset configuration for instance segmentation."""
    from ultralytics.utils import ASSETS

    # Setup Dataset A (Segmentation)
    path_a = tmp_path / "dataset_a"
    path_a.mkdir()
    (path_a / "images" / "train").mkdir(parents=True)
    (path_a / "images" / "val").mkdir(parents=True)
    (path_a / "labels" / "train").mkdir(parents=True)
    (path_a / "labels" / "val").mkdir(parents=True)

    shutil.copy(ASSETS / "bus.jpg", path_a / "images" / "train" / "img1.jpg")
    shutil.copy(ASSETS / "bus.jpg", path_a / "images" / "val" / "img1.jpg")

    # Segment annotations (class x1 y1 x2 y2 x3 y3 x4 y4 ...)
    with open(path_a / "labels" / "train" / "img1.txt", "w") as f:
        f.write("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n1 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4\n")
    with open(path_a / "labels" / "val" / "img1.txt", "w") as f:
        f.write("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

    yaml_a = {
        "path": str(path_a),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "blood", 1: "stool"},
        "nc": 2,
    }
    with open(path_a / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_a, f)

    # Setup Dataset B (Segmentation)
    path_b = tmp_path / "dataset_b"
    path_b.mkdir()
    (path_b / "images" / "train").mkdir(parents=True)
    (path_b / "images" / "val").mkdir(parents=True)
    (path_b / "labels" / "train").mkdir(parents=True)
    (path_b / "labels" / "val").mkdir(parents=True)

    shutil.copy(ASSETS / "bus.jpg", path_b / "images" / "train" / "img2.jpg")
    shutil.copy(ASSETS / "bus.jpg", path_b / "images" / "val" / "img2.jpg")

    with open(path_b / "labels" / "train" / "img2.txt", "w") as f:
        f.write("0 0.4 0.4 0.5 0.4 0.5 0.5 0.4 0.5\n1 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.3\n")
    with open(path_b / "labels" / "val" / "img2.txt", "w") as f:
        f.write("1 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.3\n")

    yaml_b = {
        "path": str(path_b),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "stool", 1: "blood"},
        "nc": 2,
    }
    with open(path_b / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_b, f)

    # Setup combined YAML
    combined_yaml = {
        "datasets": [
            str(path_a / "data.yaml"),
            str(path_b / "data.yaml"),
        ]
    }
    combined_path = tmp_path / "combined.yaml"
    with open(combined_path, "w") as f:
        yaml.safe_dump(combined_yaml, f)

    # Train segmentation model for 1 epoch
    model = YOLO("yolo26-seg.yaml")
    model.train(
        data=str(combined_path),
        epochs=1,
        imgsz=32,
        device="cpu",
        save=False,
        plots=False,
        project=str(tmp_path / "runs"),
        name="test_multi_dataset_seg",
    )

    # Verify trainer and resolved class mappings
    trainer = model.trainer
    assert trainer is not None
    assert model.model.names == {0: "blood", 1: "stool"}
    assert model.model.nc == 2

    # Verify label classes remapped correctly in Datasets
    concat_dataset = trainer.train_loader.dataset
    assert len(concat_dataset.datasets) == 2

    ds_a, ds_b = concat_dataset.datasets
    assert ds_a.labels[0]["cls"].flatten().tolist() == [0.0, 1.0]
    assert ds_b.labels[0]["cls"].flatten().tolist() == [1.0, 0.0]


def test_multi_dataset_user_defined_names(tmp_path):
    """Test end-to-end training with multi-dataset configuration and user-defined global names."""
    from ultralytics.utils import ASSETS

    # Setup Dataset A
    path_a = tmp_path / "dataset_a"
    path_a.mkdir()
    (path_a / "images" / "train").mkdir(parents=True)
    (path_a / "images" / "val").mkdir(parents=True)
    (path_a / "labels" / "train").mkdir(parents=True)
    (path_a / "labels" / "val").mkdir(parents=True)

    shutil.copy(ASSETS / "bus.jpg", path_a / "images" / "train" / "img1.jpg")
    shutil.copy(ASSETS / "bus.jpg", path_a / "images" / "val" / "img1.jpg")

    with open(path_a / "labels" / "train" / "img1.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    with open(path_a / "labels" / "val" / "img1.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    yaml_a = {
        "path": str(path_a),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "blood", 1: "stool"},
        "nc": 2,
    }
    with open(path_a / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_a, f)

    # Setup Dataset B
    path_b = tmp_path / "dataset_b"
    path_b.mkdir()
    (path_b / "images" / "train").mkdir(parents=True)
    (path_b / "images" / "val").mkdir(parents=True)
    (path_b / "labels" / "train").mkdir(parents=True)
    (path_b / "labels" / "val").mkdir(parents=True)

    shutil.copy(ASSETS / "bus.jpg", path_b / "images" / "train" / "img2.jpg")
    shutil.copy(ASSETS / "bus.jpg", path_b / "images" / "val" / "img2.jpg")

    with open(path_b / "labels" / "train" / "img2.txt", "w") as f:
        f.write("0 0.4 0.4 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")
    with open(path_b / "labels" / "val" / "img2.txt", "w") as f:
        f.write("1 0.2 0.2 0.1 0.1\n")

    yaml_b = {
        "path": str(path_b),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "stool", 1: "blood"},
        "nc": 2,
    }
    with open(path_b / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_b, f)

    # Setup combined YAML with user-defined names ordering
    # We define a custom order: 0 -> stool, 1 -> blood
    combined_yaml = {
        "names": ["stool", "blood"],
        "datasets": [
            str(path_a / "data.yaml"),
            str(path_b / "data.yaml"),
        ],
    }
    combined_path = tmp_path / "combined.yaml"
    with open(combined_path, "w") as f:
        yaml.safe_dump(combined_yaml, f)

    model = YOLO("yolo26n.yaml")
    model.train(
        data=str(combined_path),
        epochs=1,
        imgsz=32,
        device="cpu",
        save=False,
        plots=False,
        project=str(tmp_path / "runs"),
        name="test_multi_dataset_custom_names",
    )

    # Verify trainer uses custom global ordering
    trainer = model.trainer
    assert trainer is not None
    assert model.model.names == {0: "stool", 1: "blood"}
    assert model.model.nc == 2

    # Verify labels resolved to custom global IDs
    concat_dataset = trainer.train_loader.dataset
    ds_a, ds_b = concat_dataset.datasets

    # ds_a: original 0 (blood) -> global 1, original 1 (stool) -> global 0
    assert ds_a.labels[0]["cls"].flatten().tolist() == [1.0, 0.0]

    # ds_b: original 0 (stool) -> global 0, original 1 (blood) -> global 1
    assert ds_b.labels[0]["cls"].flatten().tolist() == [0.0, 1.0]


def test_extensible_datasets_list_schema(tmp_path):
    """Test multi-dataset configuration using extensible dictionary schema in the datasets list."""
    from ultralytics.utils import ASSETS

    # Setup Dataset A
    path_a = tmp_path / "dataset_a"
    path_a.mkdir()
    (path_a / "images" / "train").mkdir(parents=True)
    (path_a / "images" / "val").mkdir(parents=True)
    (path_a / "labels" / "train").mkdir(parents=True)
    (path_a / "labels" / "val").mkdir(parents=True)

    shutil.copy(ASSETS / "bus.jpg", path_a / "images" / "train" / "img1.jpg")
    shutil.copy(ASSETS / "bus.jpg", path_a / "images" / "val" / "img1.jpg")

    with open(path_a / "labels" / "train" / "img1.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
    with open(path_a / "labels" / "val" / "img1.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    yaml_a = {
        "path": str(path_a),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "blood"},
        "nc": 1,
    }
    with open(path_a / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_a, f)

    # Setup Dataset B
    path_b = tmp_path / "dataset_b"
    path_b.mkdir()
    (path_b / "images" / "train").mkdir(parents=True)
    (path_b / "images" / "val").mkdir(parents=True)
    (path_b / "labels" / "train").mkdir(parents=True)
    (path_b / "labels" / "val").mkdir(parents=True)

    shutil.copy(ASSETS / "bus.jpg", path_b / "images" / "train" / "img2.jpg")
    shutil.copy(ASSETS / "bus.jpg", path_b / "images" / "val" / "img2.jpg")

    with open(path_b / "labels" / "train" / "img2.txt", "w") as f:
        f.write("0 0.4 0.4 0.2 0.2\n")
    with open(path_b / "labels" / "val" / "img2.txt", "w") as f:
        f.write("0 0.4 0.4 0.2 0.2\n")

    yaml_b = {
        "path": str(path_b),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "stool"},
        "nc": 1,
    }
    with open(path_b / "data.yaml", "w") as f:
        yaml.safe_dump(yaml_b, f)

    # Setup combined YAML using extensible list schema (one string, one dictionary)
    combined_yaml = {
        "datasets": [
            str(path_a / "data.yaml"),
            {"data": str(path_b / "data.yaml"), "weight": 2.0},  # dict entry
        ]
    }
    combined_path = tmp_path / "combined.yaml"
    with open(combined_path, "w") as f:
        yaml.safe_dump(combined_yaml, f)

    model = YOLO("yolo26n.yaml")
    model.train(
        data=str(combined_path),
        epochs=1,
        imgsz=32,
        device="cpu",
        save=False,
        plots=False,
        project=str(tmp_path / "runs"),
        name="test_extensible_schema",
    )

    # Verify trainer uses resolved names
    assert model.model.names == {0: "blood", 1: "stool"}
    assert model.model.nc == 2

    # Verify options are preserved in sub-dataset parsed dict
    trainer = model.trainer
    assert trainer is not None
    concat_dataset = trainer.train_loader.dataset
    ds_b = concat_dataset.datasets[1]
    assert ds_b.data.get("_ds_opt_weight") == 2.0
