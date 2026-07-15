# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import get_cfg


def test_multi_dataset_parsing(tmp_path):
    """Test multi-dataset config parsing, relative path resolution, and semantic class merging."""
    # Create child dataset directories
    child1_dir = tmp_path / "child1"
    child2_dir = tmp_path / "child2"
    child1_dir.mkdir()
    child2_dir.mkdir()

    # Write class configs
    child1_yaml = child1_dir / "child1.yaml"
    child1_yaml.write_text(
        f"""
path: {child1_dir.as_posix()}
train: train_imgs
val: val_imgs
nc: 2
names:
  0: person
  1: car
""",
        encoding="utf-8",
    )

    child2_yaml = child2_dir / "child2.yaml"
    child2_yaml.write_text(
        f"""
path: {child2_dir.as_posix()}
train: train_imgs
val: val_imgs
nc: 2
names:
  0: car
  1: dog
""",
        encoding="utf-8",
    )

    # Parent configuration specifying multiple child dataset configs
    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text(
        f"""
datasets:
  - child1/child1.yaml
  - child2/child2.yaml
""",
        encoding="utf-8",
    )

    # Run check_det_dataset on parent YAML
    data = check_det_dataset(str(parent_yaml), autodownload=False)

    # 1. Verify semantic class merging
    # Classes should be: person -> 0, car -> 1, dog -> 2
    assert data["nc"] == 3
    assert data["names"] == {0: "person", 1: "car", 2: "dog"}

    # 2. Verify class maps
    class_maps = data["class_maps"]
    c1_resolved_path = str(child1_dir.resolve())
    c2_resolved_path = str(child2_dir.resolve())
    assert c1_resolved_path in class_maps
    assert c2_resolved_path in class_maps

    # child1 class map: 0 (person) -> 0, 1 (car) -> 1
    assert class_maps[c1_resolved_path][0] == 0
    assert class_maps[c1_resolved_path][1] == 1

    # child2 class map: 0 (car) -> 1, 1 (dog) -> 2
    assert class_maps[c2_resolved_path][0] == 1
    assert class_maps[c2_resolved_path][1] == 2

    # 3. Verify path resolution (child train/val paths should be merged and absolute)
    assert len(data["train"]) == 2
    assert len(data["val"]) == 2
    assert data["train"][0] == str((child1_dir / "train_imgs").resolve())
    assert data["train"][1] == str((child2_dir / "train_imgs").resolve())


def test_multi_dataset_label_mapping(tmp_path):
    """Test that labels are correctly mapped from local class IDs to global class IDs in YOLODataset."""
    # Create child dataset directories with images and labels directories
    child1_dir = tmp_path / "child1"
    child2_dir = tmp_path / "child2"

    for d in (child1_dir, child2_dir):
        (d / "images/train").mkdir(parents=True)
        (d / "labels/train").mkdir(parents=True)

    # Create mock images (10x10 black image)
    im = Image.new("RGB", (10, 10))
    im.save(child1_dir / "images/train/img1.jpg")
    im.save(child2_dir / "images/train/img2.jpg")

    # Create mock labels
    # child1 labels: class 1 (car) -> bbox
    (child1_dir / "labels/train/img1.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    # child2 labels: class 0 (car) -> bbox
    (child2_dir / "labels/train/img2.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    # Write YAMLs
    child1_yaml = child1_dir / "child1.yaml"
    child1_yaml.write_text(
        f"""
path: {child1_dir.as_posix()}
train: images/train
val: images/train
nc: 2
names:
  0: person
  1: car
""",
        encoding="utf-8",
    )

    child2_yaml = child2_dir / "child2.yaml"
    child2_yaml.write_text(
        f"""
path: {child2_dir.as_posix()}
train: images/train
val: images/train
nc: 2
names:
  0: car
  1: dog
""",
        encoding="utf-8",
    )

    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text(
        f"""
datasets:
  - child1/child1.yaml
  - child2/child2.yaml
""",
        encoding="utf-8",
    )

    # Load parent configuration
    data = check_det_dataset(str(parent_yaml), autodownload=False)

    # Instantiate YOLODataset using the merged config
    dataset = YOLODataset(
        img_path=data["train"],
        data=data,
        task="detect",
        augment=False,
    )

    # Load labels
    labels = dataset.get_labels()
    assert len(labels) == 2

    # Verify both labels map class "car" to global class ID 1
    label1 = next(lb for lb in labels if "img1" in lb["im_file"])
    label2 = next(lb for lb in labels if "img2" in lb["im_file"])

    assert int(label1["cls"][0, 0]) == 1
    assert int(label2["cls"][0, 0]) == 1


def test_multi_dataset_relative_path_resolution(tmp_path):
    """Test that relative paths resolve correctly relative to the child YAML directory."""
    child_dir = tmp_path / "nested_dir" / "child_dataset"
    child_dir.mkdir(parents=True)
    (child_dir / "images/train").mkdir(parents=True)

    child_yaml = child_dir / "dataset.yaml"
    child_yaml.write_text(
        """
path: .
train: images/train
val: images/train
nc: 1
names: [object]
""",
        encoding="utf-8",
    )

    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text(
        """
datasets:
  - nested_dir/child_dataset/dataset.yaml
""",
        encoding="utf-8",
    )

    data = check_det_dataset(str(parent_yaml), autodownload=False)
    assert data["train"][0] == str((child_dir / "images/train").resolve())


def test_multi_dataset_ordering_and_duplicates(tmp_path):
    """Test ordering and duplicate names mapping in semantic class merging."""
    # Dataset A classes: person, car
    # Dataset B classes: car, person (swapped)
    child1_dir = tmp_path / "child1"
    child2_dir = tmp_path / "child2"
    child1_dir.mkdir()
    child2_dir.mkdir()

    child1_yaml = child1_dir / "child1.yaml"
    child1_yaml.write_text("nc: 2\nnames: [person, car]\npath: .\ntrain: .\nval: .", encoding="utf-8")

    child2_yaml = child2_dir / "child2.yaml"
    child2_yaml.write_text("nc: 2\nnames: [car, person]\npath: .\ntrain: .\nval: .", encoding="utf-8")

    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("datasets:\n  - child1/child1.yaml\n  - child2/child2.yaml", encoding="utf-8")

    data = check_det_dataset(str(parent_yaml), autodownload=False)
    # The global class IDs should be person -> 0, car -> 1
    assert data["nc"] == 2
    assert data["names"] == {0: "person", 1: "car"}

    # Child 1: person (0) -> 0, car (1) -> 1
    # Child 2: car (0) -> 1, person (1) -> 0
    c1_path = str(Path(child1_dir).resolve())
    assert data["class_maps"][c1_path][0] == 0
    assert data["class_maps"][c1_path][1] == 1


def test_multi_dataset_three_datasets(tmp_path):
    """Test recursive merging of three datasets."""
    c1_yaml = tmp_path / "c1.yaml"
    c1_yaml.write_text("nc: 1\nnames: [cat]\npath: .\ntrain: .\nval: .", encoding="utf-8")

    c2_yaml = tmp_path / "c2.yaml"
    c2_yaml.write_text("nc: 1\nnames: [dog]\npath: .\ntrain: .\nval: .", encoding="utf-8")

    c3_yaml = tmp_path / "c3.yaml"
    c3_yaml.write_text("nc: 1\nnames: [bird]\npath: .\ntrain: .\nval: .", encoding="utf-8")

    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("datasets:\n  - c1.yaml\n  - c2.yaml\n  - c3.yaml", encoding="utf-8")

    data = check_det_dataset(str(parent_yaml), autodownload=False)
    assert data["nc"] == 3
    assert list(data["names"].values()) == ["cat", "dog", "bird"]


def test_multi_dataset_missing_names_or_nc(tmp_path):
    """Test error handling when names and nc are missing in child YAML."""
    child_yaml = tmp_path / "child.yaml"
    child_yaml.write_text("path: .\ntrain: .\nval: .", encoding="utf-8")

    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("datasets:\n  - child.yaml", encoding="utf-8")

    with pytest.raises(SyntaxError):
        check_det_dataset(str(parent_yaml), autodownload=False)


def test_multi_dataset_missing_child_yaml(tmp_path):
    """Test clean failure when a child YAML is missing."""
    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("datasets:\n  - nonexistent_child.yaml", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        check_det_dataset(str(parent_yaml), autodownload=False)


def test_multi_dataset_no_names_autogen(tmp_path):
    """Test auto-generation of class names if child only specifies nc."""
    child_yaml = tmp_path / "child.yaml"
    child_yaml.write_text("nc: 3\npath: .\ntrain: .\nval: .", encoding="utf-8")

    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("datasets:\n  - child.yaml", encoding="utf-8")

    data = check_det_dataset(str(parent_yaml), autodownload=False)
    assert data["nc"] == 3
    assert data["names"] == {0: "class_0", 1: "class_1", 2: "class_2"}
