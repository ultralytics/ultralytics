#!/usr/bin/env python3
"""
Test script to validate all P23456 models can train successfully.

Tests:
1. yolov8-p23456.yaml - Standard 5-layer detection
2. yolov8-seg-p23456.yaml - Standard 5-layer segmentation
3. yolov8-p23456-moe.yaml - Head-level MoE detection
4. yolov8-seg-p23456-moe.yaml - Head-level MoE segmentation
5. yolov8-p23456-neck-moe.yaml - Neck-level MoE detection
6. yolov8-seg-p23456-neck-moe.yaml - Neck-level MoE segmentation
"""

from ultralytics import YOLO


def test_model_training(config_path, data_yaml, model_name, epochs=3, imgsz=640):
    """Test if a model can train without errors."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name}")
    print(f"Config: {config_path}")
    print(f"Data: {data_yaml}")
    print(f"{'=' * 80}")

    try:
        # Load model
        print("📦 Loading model...")
        model = YOLO(config_path)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Check for MoE components
        has_moe = False
        if hasattr(model.model, "model"):
            for m in model.model.model:
                if "MoE" in m.__class__.__name__:
                    has_moe = True
                    print(f"   Found MoE module: {m.__class__.__name__}")
                    if hasattr(m, "top_k"):
                        print(f"   - top_k: {m.top_k}")
                    if hasattr(m, "nl"):
                        print(f"   - num_layers: {m.nl}")

        if not has_moe and "moe" in config_path.lower():
            print("   ⚠️  Warning: Expected MoE module but none found!")

        # Start training
        print(f"\n🚀 Starting training (epochs={epochs}, imgsz={imgsz})...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=2,  # Small batch for testing
            device="cpu",  # Use CPU for compatibility
            verbose=False,  # Reduce output
            plots=False,  # Disable plots
            save=False,  # Don't save checkpoints
        )

        print("✅ Training completed successfully!")
        print(f"   Final metrics available: {results is not None}")

        return True

    except Exception as e:
        print("❌ Training failed!")
        print(f"   Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Test all model configurations."""
    print("=" * 80)
    print("YOLOv8-P23456 Models Training Test Suite")
    print("=" * 80)

    # Test configurations
    tests = [
        # Standard models
        {
            "name": "YOLOv8-P23456 (Detection)",
            "config": "ultralytics/cfg/models/v8/yolov8-p23456.yaml",
            "data": "coco8.yaml",
        },
        {
            "name": "YOLOv8-SEG-P23456 (Segmentation)",
            "config": "ultralytics/cfg/models/v8/yolov8-seg-p23456.yaml",
            "data": "coco8-seg.yaml",
        },
        # Head-level MoE models
        {
            "name": "YOLOv8-P23456-MoE (Detection, Head-level routing)",
            "config": "ultralytics/cfg/models/v8/yolov8-p23456-moe.yaml",
            "data": "coco8.yaml",
        },
        {
            "name": "YOLOv8-SEG-P23456-MoE (Segmentation, Head-level routing)",
            "config": "ultralytics/cfg/models/v8/yolov8-seg-p23456-moe.yaml",
            "data": "coco8-seg.yaml",
        },
        # Neck-level MoE models
        {
            "name": "YOLOv8-P23456-Neck-MoE (Detection, Neck-level routing)",
            "config": "ultralytics/cfg/models/v8/yolov8-p23456-neck-moe.yaml",
            "data": "coco8.yaml",
        },
        {
            "name": "YOLOv8-SEG-P23456-Neck-MoE (Segmentation, Neck-level routing)",
            "config": "ultralytics/cfg/models/v8/yolov8-seg-p23456-neck-moe.yaml",
            "data": "coco8-seg.yaml",
        },
    ]

    # Run tests
    results = {}
    for test in tests:
        success = test_model_training(
            config_path=test["config"],
            data_yaml=test["data"],
            model_name=test["name"],
            epochs=3,  # Just 3 epochs for quick validation
            imgsz=640,
        )
        results[test["name"]] = success

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n{'=' * 80}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'=' * 80}")

    if passed == total:
        print("\n🎉 All models can train successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} model(s) failed to train")
        return 1


if __name__ == "__main__":
    exit(main())
