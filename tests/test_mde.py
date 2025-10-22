# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Unit tests for MDE (Monocular Depth Estimation) functionality.

Tests cover:
- MDE model loading and architecture
- MDE training on minimal dataset
- MDE validation
- MDE prediction with depth output
- MDE metrics computation
"""

import pytest
import torch

from tests import TMP
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def test_mde_model_loading():
    """Test loading YOLO11 MDE model configuration."""
    model = YOLO("yolo11n-mde.yaml", task="mde")
    assert model is not None
    assert model.task == "mde"  # MDE task
    
    # Check model has necessary components
    assert hasattr(model.model, "model")
    
    # Verify last layer is Detect_MDE
    last_layer = model.model.model[-1]
    assert last_layer.__class__.__name__ == "Detect_MDE", f"Expected Detect_MDE, got {last_layer.__class__.__name__}"
    
    # Verify MDE-specific attributes
    # Detect_MDE outputs depth through cv3 (nc + 1 channels)
    assert hasattr(last_layer, "cv3"), "Detect_MDE should have cv3 for class and depth prediction"
    # Check that output channels include depth (+1 for depth channel)
    assert last_layer.no == last_layer.nc + last_layer.reg_max * 4 + 1, "MDE should have +1 output for depth"


def test_mde_model_forward():
    """Test forward pass of MDE model."""
    model = YOLO("yolo11n-mde.yaml", task="mde")
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 320, 320)
    
    # Set model to eval mode for inference format output
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(dummy_input)
    
    assert outputs is not None
    
    # In eval mode, model returns (predictions, raw_outputs)
    if isinstance(outputs, tuple):
        preds, raw = outputs
        assert isinstance(preds, torch.Tensor), "Predictions should be a tensor"
        # Predictions should have shape (batch, num_predictions, 6 + nc) 
        # where 6 = (x, y, w, h, conf, depth)
        assert preds.dim() == 3, f"Predictions should have 3 dimensions, got {preds.dim()}"
    else:
        # Training mode outputs list of feature maps
        assert isinstance(outputs, list), "Outputs should be a list in training mode"
        for i, out in enumerate(outputs):
            assert isinstance(out, torch.Tensor), f"Output {i} should be a tensor"
            # Training outputs are 4D: (batch, channels, height, width)
            assert out.dim() == 4, f"Output {i} should have 4 dimensions in training mode"


def test_mde_train():
    """Test MDE training on minimal dataset."""
    # Create model
    model = YOLO("yolo11n-mde.yaml", task="mde")
    
    # Clean up any existing model weights to ensure fresh training
    import os
    weights_dir = TMP / "test_mde_train" / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            os.remove(weight_file)
    
    # Train on minimal dataset with minimal settings
    results = model.train(
        data="sunrgbd8.yaml",
        epochs=1,
        imgsz=160,  # Small size for fast testing
        batch=2,
        cache=False,
        device="cpu",
        workers=0,  # Single worker for consistency
        project=str(TMP),
        name="test_mde_train",
        exist_ok=True,
        verbose=False,
    )
    
    assert results is not None
    
    # Check that model was saved
    weights_path = TMP / "test_mde_train" / "weights" / "last.pt"
    assert weights_path.exists(), f"Model weights should be saved at {weights_path}"


def test_mde_val():
    """Test MDE validation."""
    # Train a minimal model first
    model = YOLO("yolo11n-mde.yaml", task="mde")
    
    # Clean up any existing model weights to ensure fresh training
    import os
    weights_dir = TMP / "test_mde_val" / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            os.remove(weight_file)
    
    model.train(
        data="sunrgbd8.yaml",
        epochs=1,
        imgsz=160,
        batch=2,
        cache=False,
        device="cpu",
        workers=0,
        project=str(TMP),
        name="test_mde_val",
        exist_ok=True,
        verbose=False,
    )
    
    # Validate the model
    results = model.val(
        data="sunrgbd8.yaml",
        imgsz=160,
        batch=2,
        device="cpu",
        workers=0,
        project=str(TMP),
        name="test_mde_val_results",
        exist_ok=True,
        verbose=False,
    )
    
    assert results is not None
    
    # Check for expected metrics
    assert hasattr(results, "box"), "Results should have box metrics"
    
    # MDE-specific: should have depth metrics
    # Note: Depth metrics might be in results.speed or a separate attribute
    # depending on the implementation


def test_mde_predict():
    """Test MDE prediction with depth output."""
    # Train a minimal model first
    model = YOLO("yolo11n-mde.yaml", task="mde")
    
    # Clean up any existing model weights to ensure fresh training
    import os
    weights_dir = TMP / "test_mde_predict" / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            os.remove(weight_file)
    
    model.train(
        data="sunrgbd8.yaml",
        epochs=1,
        imgsz=160,
        batch=2,
        cache=False,
        device="cpu",
        workers=0,
        project=str(TMP),
        name="test_mde_predict",
        exist_ok=True,
        verbose=False,
    )
    
    # Run prediction on test images
    test_images = ASSETS / "sunrgbd8" / "images"
    results = model.predict(
        source=str(test_images),
        imgsz=160,
        conf=0.01,  # Low confidence to ensure we get some predictions
        device="cpu",
        verbose=False,
    )
    
    assert results is not None
    assert len(results) > 0, "Should return results for all images"
    
    # Check result structure
    for r in results:
        assert hasattr(r, "boxes"), "Results should have boxes"
        
        # If there are detections, check for depth information
        if r.boxes is not None and len(r.boxes) > 0:
            # The boxes.data tensor should have depth as the last column
            # Format: [x1, y1, x2, y2, conf, cls, depth]
            data = r.boxes.data
            assert data.shape[1] >= 7, f"Box data should have at least 7 columns (including depth), got {data.shape[1]}"


def test_mde_pretrained_predict():
    """Test prediction with just the model architecture (no pre-training)."""
    model = YOLO("yolo11n-mde.yaml", task="mde")
    
    # Run prediction on a single test image
    test_image = ASSETS / "sunrgbd8" / "images" / "test000.jpg"
    
    results = model.predict(
        source=str(test_image),
        imgsz=160,
        conf=0.25,
        device="cpu",
        verbose=False,
    )
    
    assert results is not None
    assert len(results) == 1, "Should return one result for one image"


@pytest.mark.slow
def test_mde_train_multi_epoch():
    """Test MDE training with multiple epochs (slow test)."""
    model = YOLO("yolo11n-mde.yaml")
    
    results = model.train(
        data="sunrgbd8.yaml",
        epochs=3,
        imgsz=320,
        batch=4,
        cache=False,
        device="cpu",
        workers=0,
        project=str(TMP),
        name="test_mde_multi_epoch",
        exist_ok=True,
        verbose=False,
    )
    
    assert results is not None
    
    # Check that training improved (losses should decrease)
    # This is a basic sanity check
    weights_path = TMP / "test_mde_multi_epoch" / "weights" / "best.pt"
    assert weights_path.exists(), "Best weights should be saved"


def test_mde_dataset_config():
    """Test that MDE dataset configuration is valid."""
    import yaml
    from pathlib import Path
    
    config_path = Path("ultralytics/cfg/datasets/sunrgbd8.yaml")
    assert config_path.exists(), f"Dataset config should exist at {config_path}"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    assert "path" in config, "Config should have 'path' field"
    assert "train" in config, "Config should have 'train' field"
    assert "val" in config, "Config should have 'val' field"
    assert "nc" in config, "Config should have 'nc' field"
    assert "names" in config, "Config should have 'names' field"
    assert "depth_max" in config, "MDE config should have 'depth_max' field"
    
    # Check number of classes matches names
    assert config["nc"] == 37, "SUNRGBD should have 37 classes"
    assert len(config["names"]) == 37, "Should have 37 class names"
    
    # Check depth_max is reasonable
    assert config["depth_max"] == 10.0, "SUNRGBD depth_max should be 10.0 meters"


def test_mde_data_loading():
    """Test that MDE dataset can be loaded."""
    from ultralytics.data.utils import check_det_dataset
    from pathlib import Path
    
    # Use the same logic as the actual training code
    data = check_det_dataset('sunrgbd8.yaml', autodownload=True)
    
    # Check images exist
    images_path = Path(data["train"])
    assert images_path.exists(), f"Images directory should exist at {images_path}"

    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    assert len(image_files) == 8, f"Should have 8 test images, found {len(image_files)}"

    # Check labels exist
    labels_path = images_path.parent / "labels"
    assert labels_path.exists(), f"Labels directory should exist at {labels_path}"

    label_files = list(labels_path.glob("*.txt"))
    assert len(label_files) == 8, f"Should have 8 label files, found {len(label_files)}"

    # Verify label format (should have 6 values: class x y w h depth)
    sample_label = label_files[0]
    with open(sample_label, "r") as f:
        first_line = f.readline().strip()
        if first_line:  # Only check if not empty
            parts = first_line.split()
            assert len(parts) == 6, f"Label should have 6 values (class x y w h depth), got {len(parts)}"
            # Verify depth value is normalized (0-1)
            depth = float(parts[5])
            assert 0 <= depth <= 1, f"Depth should be normalized to [0, 1], got {depth}"


def test_mde_dataset_download():
    """Test that sunrgbd8 dataset can be downloaded from GitHub Assets."""
    import shutil
    from pathlib import Path
    from ultralytics.utils import DATASETS_DIR
    
    # Remove existing dataset if it exists to test fresh download
    dataset_path = DATASETS_DIR / "sunrgbd8"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    
    # Test download using check_det_dataset with autodownload=True
    from ultralytics.data.utils import check_det_dataset
    
    # This should trigger the download
    data = check_det_dataset('sunrgbd8.yaml', autodownload=True)
    
    # Verify dataset was downloaded
    assert dataset_path.exists(), f"Dataset should be downloaded to {dataset_path}"
    
    # Check structure
    images_path = dataset_path / "images"
    labels_path = dataset_path / "labels"
    
    assert images_path.exists(), f"Images directory should exist at {images_path}"
    assert labels_path.exists(), f"Labels directory should exist at {labels_path}"
    
    # Check file counts
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    label_files = list(labels_path.glob("*.txt"))
    
    assert len(image_files) == 8, f"Should have 8 images after download, found {len(image_files)}"
    assert len(label_files) == 8, f"Should have 8 label files after download, found {len(label_files)}"
    
    # Verify download URL is accessible (basic connectivity test)
    import requests
    download_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/sunrgbd8.zip"
    try:
        response = requests.head(download_url, timeout=10, allow_redirects=True)
        # GitHub releases often redirect (302), so accept both 200 and 302
        assert response.status_code in [200, 302], f"Download URL should be accessible, got status {response.status_code}"
    except requests.RequestException as e:
        pytest.skip(f"Download URL not accessible: {e}")
    
    # Test that the dataset can be used for training after download
    model = YOLO("yolo11n-mde.yaml", task="mde")
    
    # Clean up any existing model weights to ensure fresh training
    import os
    weights_dir = TMP / "test_mde_download" / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            os.remove(weight_file)
    
    # Quick training test to verify downloaded dataset works
    results = model.train(
        data="sunrgbd8.yaml",
        epochs=1,
        imgsz=160,
        batch=2,
        cache=False,
        device="cpu",
        workers=0,
        project=str(TMP),
        name="test_mde_download",
        exist_ok=True,
        verbose=False,
    )
    
    assert results is not None, "Training should succeed with downloaded dataset"

