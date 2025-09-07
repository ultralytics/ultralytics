from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.prune import prune_detection_model

ROOT = Path(__file__).resolve().parents[1]  # repo root
IMG = ROOT / "ultralytics/assets/bus.jpg"
CFG = ROOT / "ultralytics/cfg/models/v8/sample_prune.yaml"


def test_prune_roundtrip(tmp_path):
    model = YOLO("yolov8n.yaml")
    pruned_model = prune_detection_model(model, prune_ratio=0.25)

    # save and reload
    save_path = tmp_path / "pruned.pt"
    pruned_model.save(save_path)
    loaded_model = YOLO(str(save_path))

    # dummy inference
    results = loaded_model(str(IMG))
    assert results is not None


def test_prune_roundtrip_with_config(tmp_path):
    model = YOLO("yolov8n.yaml")
    pruned_model = prune_detection_model(model, prune_yaml=str(CFG))

    # save and reload
    save_path = tmp_path / "pruned_cfg.pt"
    pruned_model.save(save_path)
    loaded_model = YOLO(str(save_path))

    # dummy inference
    results = loaded_model(str(IMG))
    assert results is not None


def test_prune_train(tmp_path):
    model = YOLO("yolov8n.yaml")
    pruned_model = prune_detection_model(model, prune_ratio=0.1)

    # save and reload
    save_path = tmp_path / "pruned_train.pt"
    pruned_model.save(save_path)
    loaded_model = YOLO(str(save_path))

    # train briefly
    loaded_model.train(data="coco8.yaml", epochs=2, imgsz=32)

    # dummy inference
    results = loaded_model(str(IMG))
    assert results is not None


def test_prune_reduces_size(tmp_path):
    model = YOLO("yolov8n.yaml")
    original_path = tmp_path / "original.pt"
    pruned_path = tmp_path / "pruned.pt"

    # save original model
    model.save(original_path)
    orig_size = original_path.stat().st_size

    # prune and save
    pruned_model = prune_detection_model(model, prune_ratio=0.1)
    pruned_model.save(pruned_path)
    pruned_size = pruned_path.stat().st_size

    # check pruned model is smaller
    assert pruned_size < orig_size, f"Pruned model size {pruned_size} >= original {orig_size}"
