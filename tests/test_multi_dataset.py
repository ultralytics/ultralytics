# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import pytest
from PIL import Image

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset


def test_multi_dataset_parsing(tmp_path):
    """Test multi-dataset config parsing, relative path resolution, and semantic class merging."""
    # Create child dataset directories
    child1_dir = tmp_path / "child1"
    child2_dir = tmp_path / "child2"
    child1_dir.mkdir()
    child2_dir.mkdir()
    (child1_dir / "train_imgs").mkdir()
    (child1_dir / "val_imgs").mkdir()
    (child2_dir / "train_imgs").mkdir()
    (child2_dir / "val_imgs").mkdir()

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
        """
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
    c1_train_path = str(Path(child1_dir / "train_imgs").resolve())
    c2_train_path = str(Path(child2_dir / "train_imgs").resolve())
    assert c1_train_path in class_maps
    assert c2_train_path in class_maps

    # child1 class map: 0 (person) -> 0, 1 (car) -> 1
    assert class_maps[c1_train_path][0] == 0
    assert class_maps[c1_train_path][1] == 1

    # child2 class map: 0 (car) -> 1, 1 (dog) -> 2
    assert class_maps[c2_train_path][0] == 1
    assert class_maps[c2_train_path][1] == 2

    # 3. Verify path resolution (child train/val paths should be merged and absolute)
    assert len(data["train"]) == 2
    assert len(data["val"]) == 2
    assert Path(data["train"][0]).resolve() == Path(child1_dir / "train_imgs").resolve()
    assert Path(data["train"][1]).resolve() == Path(child2_dir / "train_imgs").resolve()


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
        """
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


def test_multi_dataset_overlapping_root_provenance(tmp_path):
    """Test that multiple child datasets sharing the same dataset root can coexist and resolve correctly."""
    common_dir = tmp_path / "common_root"
    child1_train = common_dir / "c1_train"
    child2_train = common_dir / "c2_train"
    child1_train.mkdir(parents=True)
    child2_train.mkdir(parents=True)

    child1_yaml = common_dir / "child1.yaml"
    child1_yaml.write_text("path: .\ntrain: c1_train\nval: c1_train\nnc: 1\nnames: [cat]", encoding="utf-8")

    child2_yaml = common_dir / "child2.yaml"
    child2_yaml.write_text("path: .\ntrain: c2_train\nval: c2_train\nnc: 1\nnames: [dog]", encoding="utf-8")

    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("datasets:\n  - common_root/child1.yaml\n  - common_root/child2.yaml", encoding="utf-8")

    data = check_det_dataset(str(parent_yaml), autodownload=False)
    assert data["nc"] == 2
    assert data["names"] == {0: "cat", 1: "dog"}

    # Assert that class maps are keyed by the resolved split paths, not the common root
    c1_resolved_train = str(child1_train.resolve())
    c2_resolved_train = str(child2_train.resolve())
    assert c1_resolved_train in data["class_maps"]
    assert c2_resolved_train in data["class_maps"]
    assert data["class_maps"][c1_resolved_train][0] == 0
    assert data["class_maps"][c2_resolved_train][0] == 1


def test_multi_dataset_out_of_bounds_cls_id(tmp_path):
    """Test that an out-of-bounds local class ID in a child dataset raises ValueError."""
    child1_dir = tmp_path / "child1"
    child2_dir = tmp_path / "child2"
    child1_dir.mkdir()
    child2_dir.mkdir()

    (child1_dir / "images").mkdir()
    (child1_dir / "labels").mkdir()
    (child2_dir / "images").mkdir()
    (child2_dir / "labels").mkdir()

    # Images and labels
    im = Image.new("RGB", (10, 10))
    im.save(child1_dir / "images/img1.jpg")
    # Child 1 only has 1 class (index 0), but label uses class 1
    (child1_dir / "labels/img1.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    im.save(child2_dir / "images/img2.jpg")
    (child2_dir / "labels/img2.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    child1_yaml = child1_dir / "child1.yaml"
    child1_yaml.write_text("path: .\ntrain: images\nval: images\nnc: 1\nnames: [cat]", encoding="utf-8")

    child2_yaml = child2_dir / "child2.yaml"
    # Child 2 introduces a second class "dog", making global nc = 2
    child2_yaml.write_text("path: .\ntrain: images\nval: images\nnc: 2\nnames: [cat, dog]", encoding="utf-8")

    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("datasets:\n  - child1/child1.yaml\n  - child2/child2.yaml", encoding="utf-8")

    data = check_det_dataset(str(parent_yaml), autodownload=False)
    with pytest.raises(ValueError, match="not defined in child dataset's names"):
        YOLODataset(img_path=data["train"], data=data, task="detect", augment=False)


def test_multi_dataset_nc_names_inconsistency(tmp_path):
    """Test that string-like and float nc are normalized, and mismatch with names raises SyntaxError."""
    # 1. Test float/string-like nc normalization
    child_yaml = tmp_path / "child.yaml"
    child_yaml.write_text("path: .\ntrain: .\nval: .\nnc: 2.0\nnames: [cat, dog]", encoding="utf-8")
    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("datasets:\n  - child.yaml", encoding="utf-8")
    data = check_det_dataset(str(parent_yaml), autodownload=False)
    assert data["nc"] == 2

    # 2. Inconsistent nc/names length
    child_yaml.write_text("path: .\ntrain: .\nval: .\nnc: 3\nnames: [cat, dog]", encoding="utf-8")
    with pytest.raises(SyntaxError, match=r"names' length .* and 'nc: .*' must match"):
        check_det_dataset(str(parent_yaml), autodownload=False)


def test_multi_dataset_missing_child_split(tmp_path):
    """Test that missing child split paths raise FileNotFoundError immediately when autodownload=False."""
    child_yaml = tmp_path / "child.yaml"
    child_yaml.write_text(
        "path: .\ntrain: non_existent_train_folder\nval: non_existent_val_folder\nnc: 1\nnames: [cat]", encoding="utf-8"
    )

    parent_yaml = tmp_path / "parent.yaml"
    parent_yaml.write_text("datasets:\n  - child.yaml", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="images not found, missing path"):
        check_det_dataset(str(parent_yaml), autodownload=False)
