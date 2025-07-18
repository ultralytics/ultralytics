"""
Example usage of YOLO class balancing features.

This script demonstrates how to use the new class balancing capabilities including pos_weight for BCEWithLogitsLoss and
WeightedRandomSampler for DataLoader.
"""

from pathlib import Path

import yaml

from ultralytics import YOLO


def example_auto_class_weights():
    """Example: Training YOLO with automatic class weight calculation."""
    print("=== Example 1: Automatic Class Weight Calculation ===")

    model = YOLO("yolo11n.yaml")

    results = model.train(data="coco8.yaml", epochs=2, cls_weights=True, imgsz=640, batch=4, verbose=True)

    print("Training completed with automatic class balancing!")
    return results


def example_manual_class_weights():
    """Example: Training YOLO with manual class weights."""
    print("\n=== Example 2: Manual Class Weight Specification ===")

    model = YOLO("yolo11n.yaml")

    custom_weights = [1.0, 2.0, 1.5, 0.8, 1.2, 1.8, 0.9, 1.1]

    results = model.train(data="coco8.yaml", epochs=2, cls_weights=custom_weights, imgsz=640, batch=4, verbose=True)

    print("Training completed with manual class weights!")
    return results


def example_config_file():
    """Example: Using class weights through configuration file."""
    print("\n=== Example 3: Configuration File Usage ===")

    config_content = {
        "task": "detect",
        "mode": "train",
        "data": "coco8.yaml",
        "epochs": 2,
        "cls_weights": [1.0, 2.0, 1.5, 0.8, 1.2, 1.8, 0.9, 1.1],
        "imgsz": 640,
        "batch": 4,
        "verbose": True,
    }

    config_path = Path("custom_balanced_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    try:
        model = YOLO("yolo11n.yaml")
        results = model.train(cfg=str(config_path))
        print("Training completed with config file!")
        return results
    finally:
        if config_path.exists():
            config_path.unlink()


def analyze_class_distribution():
    """Example: Analyzing class distribution before and after balancing."""
    print("\n=== Example 4: Class Distribution Analysis ===")

    from ultralytics.data.utils import calculate_class_weights

    try:
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset

        cfg = get_cfg()
        cfg.imgsz = 640
        cfg.task = "detect"

        dataset = build_yolo_dataset(
            cfg, "coco8/images/train", batch=4, data={"train": "coco8/images/train"}, mode="train"
        )

        nc = 8
        class_weights = calculate_class_weights(dataset, nc)

        print(f"Number of classes: {nc}")
        print(f"Calculated class weights: {class_weights.tolist()}")
        print(f"Weight range: {class_weights.min():.3f} - {class_weights.max():.3f}")

        imbalance_ratio = class_weights.max() / class_weights.min()
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")

        if imbalance_ratio > 2.0:
            print("⚠️  Significant class imbalance detected - class balancing recommended!")
        else:
            print("✅ Classes are relatively balanced")

    except Exception as e:
        print(f"Could not analyze dataset: {e}")
        print("This is expected if coco8 dataset is not available")


def demonstrate_weighted_sampler():
    """Example: Demonstrating WeightedRandomSampler effect."""
    print("\n=== Example 5: WeightedRandomSampler Demonstration ===")

    print("WeightedRandomSampler is automatically used when cls_weights is specified.")
    print("It ensures more balanced class distribution in each training batch.")
    print("This is particularly useful for datasets with severe class imbalance.")

    print("\nTo verify the effect:")
    print("1. Train without class balancing and observe class distribution in batches")
    print("2. Train with cls_weights=True and compare batch distributions")
    print("3. Minority classes should appear more frequently in balanced training")


def main():
    """Run all examples."""
    print("YOLO Class Balancing Examples")
    print("=" * 50)

    try:
        example_auto_class_weights()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_manual_class_weights()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_config_file()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        analyze_class_distribution()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    demonstrate_weighted_sampler()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nKey Features Demonstrated:")
    print("✅ Automatic class weight calculation (cls_weights=True)")
    print("✅ Manual class weight specification (cls_weights=[...])")
    print("✅ Configuration file usage")
    print("✅ Class distribution analysis")
    print("✅ WeightedRandomSampler integration")
    print("\nFor production use:")
    print("- Use cls_weights=True for automatic balancing")
    print("- Monitor training metrics for minority classes")
    print("- Adjust weights based on validation performance")


if __name__ == "__main__":
    main()
