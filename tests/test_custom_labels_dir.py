from ultralytics.data.utils import img2label_paths

def test_img2label_paths_default():
    """
    Test the img2label_paths with default labels directory.
    """
    img_paths = [
        "data/images/sample1.jpg",
        "data/images/sample2.jpg",
        "data/images/sample3.jpg"
    ]
    label_paths = img2label_paths(img_paths)

    assert label_paths == [
        "data/labels/sample1.txt",
        "data/labels/sample2.txt",
        "data/labels/sample3.txt"
    ], "Label paths do not match expected output"
    return

def test_img2label_paths_custom():
    """
    Test the img2label_paths function with a custom labels directory.
    """
    img_paths = [
        "data/images/sample1.jpg",
        "data/images/sample2.jpg",
        "data/images/sample3.jpg"
    ]
    label_paths = img2label_paths(img_paths, labels_dirname="labels-custom")
    assert label_paths == [
        "data/labels-custom/sample1.txt",
        "data/labels-custom/sample2.txt",
        "data/labels-custom/sample3.txt"
    ], "Label paths with custom directory do not match expected output"

    label_paths_complex = img2label_paths(
        img_paths,
        labels_dirname="labels/version3",
    )
    assert label_paths_complex == [
        "data/labels/version3/sample1.txt",
        "data/labels/version3/sample2.txt",
        "data/labels/version3/sample3.txt"
    ], "Label paths with complex custom directory do not match expected output"
    return