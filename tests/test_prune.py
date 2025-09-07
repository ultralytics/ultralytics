from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils.prune import prune_detection_model

ROOT = Path(__file__).resolve().parents[1]  # repo root
IMG = ROOT / "ultralytics/assets/bus.jpg"
CFG = ROOT / "ultralytics/cfg/pruning/sample_prune.yaml"


def test_prune_roundtrip(tmp_path):
    """Test that pruning with a global ratio saves/loads correctly and inference still runs."""
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
    """Test that pruning with a YAML config works and the pruned model saves/loads correctly and inference still runs"""
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
    """Test that a pruned model can still be trained after pruning, and that the trained model's inference still works"""
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
    """Test that pruning reduces the number of parameters and usually reduces saved model size."""
    model = YOLO("yolov8n.yaml")

    # prune and save
    prune_ratio = 0.2
    pruned_model = prune_detection_model(model, prune_ratio=prune_ratio)

    # check that number of parameters is reduced
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    assert pruned_params < orig_params, f"Pruned params {pruned_params} >= original {orig_params}"


def test_zero_prune(tmp_path):
    """Test that pruning with zero ratio doesnt change model weights."""
    model = YOLO("yolov8n.yaml")
    pruned_model = prune_detection_model(model, prune_ratio=0)

    # Check number of parameters
    assert sum(p.numel() for p in pruned_model.parameters()) == sum(p.numel() for p in model.parameters())

    # Check that parameter values are identical
    assert all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), pruned_model.parameters()))


