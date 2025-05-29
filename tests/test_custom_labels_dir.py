import os

from ultralytics.data.utils import img2label_paths

IMAGE_PATHS = [
    os.path.join("data", "images", "sample1.jpg"),
    os.path.join("data", "images", "sample2.jpg"),
    os.path.join("data", "images", "sample3.jpg"),
]

LABELS_PATHS_DEFAULT = [
    os.path.join("data", "labels", "sample1.txt"),
    os.path.join("data", "labels", "sample2.txt"),
    os.path.join("data", "labels", "sample3.txt"),
]

LABELS_PATHS_CUSTOM = [
    os.path.join("data", "labels-custom", "sample1.txt"),
    os.path.join("data", "labels-custom", "sample2.txt"),
    os.path.join("data", "labels-custom", "sample3.txt"),
]

LABELS_PATHS_NESTED = [
    os.path.join("data", "labels", "version3", "sample1.txt"),
    os.path.join("data", "labels", "version3", "sample2.txt"),
    os.path.join("data", "labels", "version3", "sample3.txt"),
]


def test_img2label_paths_default():
    """Test the img2label_paths with default labels directory."""
    label_paths = img2label_paths(IMAGE_PATHS)

    assert label_paths == LABELS_PATHS_DEFAULT, "Label paths do not match expected output"
    return


def test_img2label_paths_custom():
    """Test the img2label_paths function with a custom labels directory."""
    label_paths = img2label_paths(IMAGE_PATHS, labels_dirname="labels-custom")
    assert label_paths == LABELS_PATHS_CUSTOM, "Label paths with custom directory do not match expected output"

    label_paths_nested = img2label_paths(
        IMAGE_PATHS,
        labels_dirname=os.path.join("labels", "version3"),
    )
    assert label_paths_nested == LABELS_PATHS_NESTED, (
        "Label paths with complex custom directory do not match expected output"
    )
    return
