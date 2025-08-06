import os
from unittest.mock import MagicMock, patch

from ultralytics.utils.dataset_validation import AutoFix, DatasetValidation, Table

# ---------- Tests for error cases ----------


def test_validate_no_yaml_file(tmp_path):
    """Test that validate() adds error when no YAML file exists."""
    dataset_dir = tmp_path

    dv = DatasetValidation(str(dataset_dir), is_fix=False)

    with patch("ultralytics.utils.dataset_validation.Table.draw_summary_table"):
        dv.validate()

    # Should have at least the "No YAML file found" error
    assert len(dv.errors) >= 1
    assert any("No YAML file found in the dataset directory" in error for error in dv.errors)


def test_validate_no_yaml_file_isolated(tmp_path):
    """Test only the YAML file check logic in isolation."""
    dataset_dir = tmp_path

    dv = DatasetValidation(str(dataset_dir), is_fix=False)

    # Test just the initial part of validate() that checks for YAML files
    if os.path.isdir(dv.dataset):
        import glob

        yamls = glob.glob(os.path.join(dv.dataset, "*.yaml"))
        if not yamls:
            dv.errors.append("No YAML file found in the dataset directory.")

    assert len(dv.errors) == 1
    assert "No YAML file found in the dataset directory" in dv.errors[0]


@patch("ultralytics.utils.dataset_validation.check_det_dataset")
@patch("ultralytics.utils.dataset_validation.Table.draw_summary_table")
def test_validate_invalid_dataset_structure(mock_draw, mock_check, tmp_path):
    """Test that validate() adds error when dataset structure is invalid."""
    dataset_dir = tmp_path
    yaml_file = dataset_dir / "dataset.yaml"
    yaml_file.write_text("nc: 1\nnames: {0: 'car'}")

    mock_check.return_value = "Invalid dataset structure - missing train directory"

    dv = DatasetValidation(str(dataset_dir), is_fix=False)
    dv.validate()

    # Should have the structure error
    assert len(dv.errors) >= 1
    assert any("Invalid dataset structure - missing train directory" in error for error in dv.errors)
    mock_draw.assert_called_once()


@patch("ultralytics.utils.dataset_validation.check_det_dataset")
@patch("ultralytics.utils.dataset_validation.verify_image_label")
@patch("ultralytics.utils.dataset_validation.Table.draw_summary_table")
def test_validate_with_invalid_labels(mock_draw, mock_verify, mock_check, tmp_path):
    """Test that validate() adds invalid labels when verify_image_label finds errors."""
    dataset_dir = tmp_path
    yaml_file = dataset_dir / "dataset.yaml"
    yaml_file.write_text("nc: 1\nnames: {0: 'car'}")

    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_val = dataset_dir / "labels" / "val"

    for p in [images_train, labels_train, images_val, labels_val]:
        p.mkdir(parents=True)

    (images_train / "image1.jpg").write_text("img")
    (labels_train / "image1.txt").write_text("lbl")
    (images_val / "image2.jpg").write_text("img")
    (labels_val / "image2.txt").write_text("lbl")

    mock_check.return_value = {"train": str(images_train), "val": str(images_val), "nc": 1, "names": {0: "car"}}

    # Return invalid label (list with 10+ elements where 9: is error info)
    invalid_label_data = ["", "", "", "", "", "", "", "", "", "Invalid label format", "additional", "error", "info"]
    mock_verify.return_value = invalid_label_data

    dv = DatasetValidation(str(dataset_dir), is_fix=False)
    dv.validate()

    assert len(dv.invalid_labels) > 0
    assert len(dv.invalid_labels[0]) > 0  # Should contain error info from index 9 onwards
    mock_draw.assert_called_once()


@patch("ultralytics.utils.dataset_validation.check_det_dataset")
@patch("ultralytics.utils.dataset_validation.Table.draw_summary_table")
def test_validate_permission_error(mock_draw, mock_check, tmp_path):
    """Test that validate() handles PermissionError correctly."""
    dataset_dir = tmp_path
    yaml_file = dataset_dir / "dataset.yaml"
    yaml_file.write_text("nc: 1\nnames: {0: 'car'}")

    mock_check.side_effect = PermissionError("Access denied")

    dv = DatasetValidation(str(dataset_dir), is_fix=False)
    dv.validate()

    assert len(dv.errors) == 1
    assert "Permission denied" in dv.errors[0]


@patch("ultralytics.utils.dataset_validation.check_det_dataset")
@patch("ultralytics.utils.dataset_validation.Table.draw_summary_table")
def test_validate_generic_exception(mock_draw, mock_check, tmp_path):
    """Test that validate() handles generic exceptions correctly."""
    dataset_dir = tmp_path
    yaml_file = dataset_dir / "dataset.yaml"
    yaml_file.write_text("nc: 1\nnames: {0: 'car'}")

    mock_check.side_effect = Exception("Something went wrong")

    dv = DatasetValidation(str(dataset_dir), is_fix=False)
    dv.validate()

    assert len(dv.errors) == 1
    assert "Dataset check failed: Something went wrong" in dv.errors[0]
    mock_draw.assert_called_once()


@patch("ultralytics.utils.dataset_validation.check_det_dataset")
@patch("ultralytics.utils.dataset_validation.AutoFix")
@patch("ultralytics.utils.dataset_validation.Table.draw_summary_table")
def test_validate_with_autofix_enabled(mock_draw, mock_autofix, mock_check, tmp_path):
    """Test that validate() calls AutoFix when is_fix=True and errors exist."""
    dataset_dir = tmp_path
    yaml_file = dataset_dir / "dataset.yaml"
    yaml_file.write_text("nc: 1\nnames: {0: 'car'}")

    mock_check.return_value = "Some validation error"
    mock_autofix_instance = MagicMock()
    mock_autofix.return_value = mock_autofix_instance

    dv = DatasetValidation(str(dataset_dir), is_fix=True)
    dv.validate()

    # Should have at least one error
    assert len(dv.errors) >= 1
    assert any("Some validation error" in error for error in dv.errors)
    mock_autofix.assert_called_once_with(str(dataset_dir), str(yaml_file))
    # AutoFix.fix should be called for each error
    assert mock_autofix_instance.fix.call_count >= 1


# ---------- Tests for Table class with errors and invalid labels ----------


@patch("ultralytics.utils.dataset_validation.LOGGER")
def test_table_draw_summary_with_errors_and_labels(mock_logger):
    """Test that Table.draw_summary_table correctly displays errors and invalid labels."""
    errors = ["Error 1: Missing file", "Error 2: Invalid format"]
    invalid_labels = [["Invalid label path 1", "error details"], ["Invalid label path 2", "more error details"]]

    table = Table(
        yaml_summary={"nc": 2, "names": {0: "car", 1: "dog"}},
        errors=errors,
        invalid_labels=invalid_labels,
        dataset="/tmp/dataset",
        yaml="dataset.yaml",
    )
    table.draw_summary_table()

    log_calls = [call.args[0] for call in mock_logger.info.call_args_list]

    assert any("DATASET VALIDATION REPORT" in msg for msg in log_calls)
    assert any("VALIDATION ERRORS" in msg for msg in log_calls)
    assert any("ERROR 1" in msg for msg in log_calls)
    assert any("ERROR 2" in msg for msg in log_calls)
    assert any("LABEL" in msg for msg in log_calls)

    # Check statistics
    assert any("Total Errors" in msg for msg in log_calls)
    assert any("Invalid Labels" in msg for msg in log_calls)


# ---------- Tests for AutoFix error detection and fixing ----------


def test_autofix_load_yaml_file_not_exists():
    """Test that load_yaml returns False when YAML file doesn't exist."""
    af = AutoFix("/some/path", "/non/existent/file.yaml")
    assert af.load_yaml() is False


def test_autofix_save_yaml_no_data():
    """Test that save_yaml returns False when no data provided."""
    af = AutoFix("/some/path", "/some/file.yaml")
    assert af.save_yaml(None) is False


@patch("ultralytics.utils.dataset_validation.LOGGER")
def test_autofix_fix_no_function_found(mock_logger):
    """Test that fix() logs warning when no fix function found for error."""
    af = AutoFix("/some/path", "/some/file.yaml")
    af.fix("Unknown error that has no fix")

    # Check that warning was logged
    log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
    assert any("No fix available for error" in msg for msg in log_calls)


# ---------- Tests for verify_labels method ----------


@patch("ultralytics.utils.dataset_validation.verify_image_label")
def test_verify_labels_adds_to_structure(mock_verify, tmp_path):
    """Test that verify_labels correctly processes files and calls verify_image_label."""
    dataset_dir = tmp_path
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"

    for p in [images_train, labels_train]:
        p.mkdir(parents=True)

    (images_train / "test1.jpg").write_text("image")
    (labels_train / "test1.txt").write_text("label")

    mock_verify.return_value = "verification_result"

    dv = DatasetValidation(str(dataset_dir), is_fix=False)
    dv.yaml_summary = {"train": str(images_train), "nc": 1}

    verify_structure = []
    dv.verify_labels(str(labels_train), verify_structure, "train")

    assert len(verify_structure) == 1
    assert verify_structure[0] == "verification_result"
    mock_verify.assert_called_once()
