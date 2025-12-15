#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Validation script for stereo 3D detection training process.

This script validates that the training infrastructure is properly set up and
can be initialized without errors.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Stereo 3D Detection Training Process Validation")
print("=" * 70)
print()

# Test 1: Configuration loading
print("Test 1: Loading stereo configuration...")
try:
    from ultralytics.cfg.models.stereo import load_stereo_config

    config = load_stereo_config("ultralytics/cfg/models/stereo/stereo-centernet-s.yaml")
    assert config["stereo"] is True, "Stereo flag not set"
    assert config["input_channels"] == 6, "Input channels should be 6"
    assert config["nc"] == 3, "Number of classes should be 3"
    print("✓ Configuration loaded successfully")
    print(f"  - Classes: {config['nc']}")
    print(f"  - Input channels: {config['input_channels']}")
    print(f"  - Stereo mode: {config['stereo']}")
except Exception as e:
    print(f"✗ Configuration loading failed: {e}")
    sys.exit(1)

# Test 2: Task detection
print("\nTest 2: Task detection from config...")
try:
    from ultralytics.nn.tasks import guess_model_task, yaml_model_load

    config_dict = yaml_model_load("ultralytics/cfg/models/stereo/stereo-centernet-s.yaml")
    task = guess_model_task(config_dict)
    assert task == "stereo3ddet", f"Expected 'stereo3ddet', got '{task}'"
    print("✓ Task detection works")
    print(f"  - Detected task: {task}")
except Exception as e:
    print(f"✗ Task detection failed: {e}")
    sys.exit(1)

# Test 3: Detection head output shapes
print("\nTest 3: Detection head output shapes...")
try:
    import torch
    from ultralytics.nn.modules.stereo.head import StereoCenterNetHead

    batch_size = 2
    in_channels = 256
    num_classes = 3
    height, width = 96, 320

    head = StereoCenterNetHead(in_channels=in_channels, num_classes=num_classes)
    head.eval()

    x = torch.randn(batch_size, in_channels, height, width)
    outputs = head(x)

    expected_branches = [
        "heatmap",
        "offset",
        "bbox_size",
        "lr_distance",
        "right_width",
        "dimensions",
        "orientation",
        "vertices",
        "vertex_offset",
        "vertex_dist",
    ]

    assert len(outputs) == 10, f"Expected 10 branches, got {len(outputs)}"
    for branch_name in expected_branches:
        assert branch_name in outputs, f"Missing branch: {branch_name}"

    # Verify shapes
    assert outputs["heatmap"].shape == (batch_size, num_classes, height, width)
    assert outputs["offset"].shape == (batch_size, 2, height, width)
    assert outputs["dimensions"].shape == (batch_size, 3, height, width)

    print("✓ Detection head produces correct output shapes")
    print(f"  - All 10 branches present")
    print(f"  - Heatmap shape: {outputs['heatmap'].shape}")
    print(f"  - Dimensions shape: {outputs['dimensions'].shape}")
except Exception as e:
    print(f"✗ Detection head test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Trainer initialization
print("\nTest 4: Trainer initialization...")
try:
    from ultralytics.engine.stereo.trainer import StereoTrainer
    from ultralytics.utils import DEFAULT_CFG

    overrides = {
        "task": "stereo3ddet",
        "model": "ultralytics/cfg/models/stereo/stereo-centernet-s.yaml",
        "data": "coco8.yaml",  # Placeholder
        "epochs": 1,
        "imgsz": 384,
        "batch": 2,
        "workers": 0,
        "save": False,
        "plots": False,
        "val": False,
    }

    trainer = StereoTrainer(overrides=overrides)
    assert trainer is not None
    assert hasattr(trainer, "loss_names")
    assert len(trainer.loss_names) == 10

    print("✓ Trainer initialized successfully")
    print(f"  - Loss names: {len(trainer.loss_names)} branches")
    print(f"  - Loss names: {', '.join(trainer.loss_names[:3])}...")
except Exception as e:
    print(f"✗ Trainer initialization failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Loss functions
print("\nTest 5: Loss functions...")
try:
    from ultralytics.nn.modules.stereo.loss import FocalLoss, L1Loss, StereoLoss

    # Test FocalLoss
    focal_loss = FocalLoss(alpha=2.0, beta=4.0)
    pred = torch.sigmoid(torch.randn(2, 3, 96, 320))
    target = torch.zeros(2, 3, 96, 320)
    target[0, 0, 48, 160] = 1.0  # One positive
    loss = focal_loss(pred, target)
    assert loss.item() > 0, "Focal loss should be positive"

    # Test L1Loss
    l1_loss = L1Loss()
    pred_reg = torch.randn(2, 3, 96, 320)
    target_reg = torch.randn(2, 3, 96, 320)
    loss_reg = l1_loss(pred_reg, target_reg)
    assert loss_reg.item() >= 0, "L1 loss should be non-negative"

    print("✓ Loss functions work correctly")
    print(f"  - Focal loss: {loss.item():.4f}")
    print(f"  - L1 loss: {loss_reg.item():.4f}")
except Exception as e:
    print(f"✗ Loss function test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 6: Checkpoint methods
print("\nTest 6: Checkpoint save/load methods...")
try:
    from ultralytics.engine.stereo.trainer import StereoTrainer

    overrides = {
        "task": "stereo3ddet",
        "model": "yolo11n.yaml",
        "data": "coco8.yaml",
        "epochs": 1,
        "save": False,
        "val": False,
    }
    trainer = StereoTrainer(overrides=overrides)

    assert hasattr(trainer, "save_model"), "save_model method missing"
    assert hasattr(trainer, "resume_training"), "resume_training method missing"
    assert callable(trainer.save_model), "save_model not callable"
    assert callable(trainer.resume_training), "resume_training not callable"

    print("✓ Checkpoint methods exist and are callable")
except Exception as e:
    print(f"✗ Checkpoint methods test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("Validation Summary")
print("=" * 70)
print("✓ All training components validated successfully!")
print()
print("Training process is ready for use:")
print("  - Configuration loading: ✓")
print("  - Task detection: ✓")
print("  - Detection head: ✓")
print("  - Trainer initialization: ✓")
print("  - Loss functions: ✓")
print("  - Checkpoint support: ✓")
print()
print("Next steps:")
print("  1. Prepare KITTI-format dataset")
print("  2. Run: model = YOLO('cfg/models/stereo/stereo-centernet-s.yaml')")
print("  3. Run: model.train(data='kitti.yaml', epochs=100)")
print("=" * 70)

