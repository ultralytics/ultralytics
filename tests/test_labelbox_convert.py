import math
from pathlib import Path

from ultralytics.data.labelbox import convert_labelbox, load_class_map, load_labelbox_mapping


def test_load_class_map_and_mapping(tmp_path: Path):
    yaml_text = """
    classes:
      class0:
        labelbox: ["labelbox.class0"]
      class1:
        labelbox: ["labelbox.class1"]
      class2:
        labelbox: ["labelbox.class2a", "labelbox.class2b"]
    """
    p = tmp_path / "class_map.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    name_to_id = load_class_map(p)
    lb_to_id = load_labelbox_mapping(p)

    assert name_to_id == {"class0": 0, "class1": 1, "class2": 2}
    assert lb_to_id["labelbox.class0"] == 0
    assert lb_to_id["labelbox.class1"] == 1
    assert lb_to_id["labelbox.class2a"] == 2
    assert lb_to_id["labelbox.class2b"] == 2


def test_convert_labelbox_creates_yolo_dataset(tmp_path: Path):
    # Class map
    yaml_text = """
    classes:
      class0:
        labelbox: ["labelbox.class0"]
    """
    class_map_path = tmp_path / "class_map.yaml"
    class_map_path.write_text(yaml_text, encoding="utf-8")

    # Images
    images_dir = tmp_path / "images_src"
    images_dir.mkdir()
    img = images_dir / "img1.jpg"
    img.write_bytes(b"dummy")

    # NDJSON export with a single bbox
    ndjson_path = tmp_path / "export.ndjson"
    line = {
        "externalId": "img1.jpg",
        "imageWidth": 100,
        "imageHeight": 50,
        "objects": [
            {
                "className": "labelbox.class0",
                "bbox": {"left": 10, "top": 20, "width": 20, "height": 10},
            }
        ],
    }
    ndjson_path.write_text(f"{line}\n".replace("'", '"'), encoding="utf-8")

    # Run conversion
    save_dir = tmp_path / "out"
    yaml_out = convert_labelbox(ndjson_path, images_dir, save_dir, class_map_path)

    # Check dataset YAML
    assert yaml_out.exists()

    # Check labels file
    labels_file = save_dir / "labels" / "train" / "img1.txt"
    assert labels_file.exists()
    text = labels_file.read_text(encoding="utf-8").strip()
    parts = text.split()
    assert parts[0] == "0"  # class id
    x_c, y_c, w, h = map(float, parts[1:])

    # Expected normalized values
    assert math.isclose(x_c, 0.2, rel_tol=1e-6)
    assert math.isclose(y_c, 0.5, rel_tol=1e-6)
    assert math.isclose(w, 0.2, rel_tol=1e-6)
    assert math.isclose(h, 0.2, rel_tol=1e-6)

    # Check image copied
    copied_img = save_dir / "images" / "train" / "img1.jpg"
    assert copied_img.exists()


