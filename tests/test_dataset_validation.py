from unittest.mock import patch
from ultralytics.utils.dataset_validation import DatasetValidation, AutoFix, Table

# ---------- check_matching_files_count ----------


def test_check_matching_files_count_match(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    for i in range(3):
        (images / f"{i}.jpg").write_text("img")
        (labels / f"{i}.txt").write_text("lbl")

    dv = DatasetValidation(str(tmp_path), is_fix=False)
    assert dv.check_matching_files_count(str(images), str(labels)) is True
    assert dv.errors == []


def test_check_matching_files_count_mismatch(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    (images / "1.jpg").write_text("img")  # 1 image, 0 labels

    dv = DatasetValidation(str(tmp_path), is_fix=False)
    assert dv.check_matching_files_count(str(images), str(labels)) is False
    assert "do not match" in dv.errors[0]


# ---------- validate ----------


@patch("ultralytics.utils.dataset_validation.check_det_dataset")
@patch("ultralytics.utils.dataset_validation.verify_image_label")
@patch("ultralytics.utils.dataset_validation.Table.draw_summary_table")
def test_validation_success(mock_draw, mock_verify, mock_check, tmp_path):
    dataset_dir = tmp_path
    yaml_file = dataset_dir / "dataset.yaml"
    yaml_file.write_text("nc: 1\nnames: {0: 'car'}")

    # files structure
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_val = dataset_dir / "labels" / "val"

    for p in [images_train, labels_train, images_val, labels_val]:
        p.mkdir(parents=True)

    for i in range(2):
        (images_train / f"{i}.jpg").write_text("img")
        (labels_train / f"{i}.txt").write_text("lbl")
        (images_val / f"{i}.jpg").write_text("img")
        (labels_val / f"{i}.txt").write_text("lbl")

    # mock
    mock_check.return_value = {"train": str(images_train), "val": str(images_val), "nc": 1, "names": {0: "car"}}
    mock_verify.return_value = True

    dv = DatasetValidation(str(dataset_dir), is_fix=False)
    dv.validate()

    assert dv.errors == []
    mock_draw.assert_called_once()


# ---------- AutoFix ----------


def test_autofix_fix_missing_yaml(tmp_path):
    af = AutoFix(str(tmp_path), yaml_path=None)
    af.fix_missing_yaml()

    expected_yaml = tmp_path / "dataset.yaml"
    assert expected_yaml.exists()
    content = expected_yaml.read_text()
    assert "train" in content
    assert "names" in content


def test_autofix_fix_nc_names_mismatch(tmp_path):
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("nc: 2\nnames: {0: 'cat', 1: 'dog', 2: 'car'}")

    af = AutoFix(str(tmp_path), str(yaml_path))
    af.fix_nc_names_mismatch()

    content = yaml_path.read_text()
    assert "nc: 3" in content


def test_autofix_detect_error():
    af = AutoFix("/some/path", "/some/path/dataset.yaml")

    f1 = af.detect_error("No YAML file found in the dataset directory.")
    assert f1 == af.fix_missing_yaml

    f2 = af.detect_error("nc does not match names length")
    assert f2 == af.fix_nc_names_mismatch

    f3 = af.detect_error("This error is unknown")
    assert f3 is None


# ---------- Table ----------


@patch("ultralytics.utils.dataset_validation.LOGGER")
def test_table_draw_summary_table_minimal(mock_logger):
    table = Table(
        yaml_summary={"nc": 1, "names": {0: "car"}},
        errors=[],
        invalid_labels=[],
        dataset="/tmp/dataset",
        yaml="dataset.yaml",
    )
    table.draw_summary_table()

    log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
    assert any("DATASET VALIDATION REPORT" in msg for msg in log_calls)
    assert any("No issues found" in msg for msg in log_calls)
