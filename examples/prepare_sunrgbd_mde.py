"""
Example script to prepare the SUN RGB-D dataset for object-level depth estimation.

This script demonstrates how to use the prepare_sunrgbd_mde_dataset function to:
1. Download the SUN RGB-D dataset (optional)
2. Convert annotations to YOLO MDE format
3. Create train/val splits
4. Generate dataset statistics

Usage:
    python examples/prepare_sunrgbd_mde.py --download --dataset-dir datasets/sunrgbd_mde
"""

import argparse
from pathlib import Path

from ultralytics.data.prepare_sunrgbd_mde import prepare_sunrgbd_mde_dataset
from ultralytics.utils import LOGGER


def main():
    """Main function to prepare SUN RGB-D dataset."""
    parser = argparse.ArgumentParser(description="Prepare SUN RGB-D dataset for MDE training")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/sunrgbd_mde",
        help="Output directory for the prepared dataset",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=10.0,
        help="Maximum depth value in meters for normalization",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (ignored if using existing splits)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download SUN RGB-D dataset if not present",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared dataset",
    )
    parser.add_argument(
        "--no-existing-splits",
        action="store_true",
        help="Create new splits instead of using official train/test splits",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=10,
        help="Minimum bounding box size (width or height) in pixels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for splitting (if not using existing splits)",
    )

    args = parser.parse_args()

    LOGGER.info("🚀 Preparing SUN RGB-D dataset for object-level depth estimation...")
    LOGGER.info(f"📁 Output directory: {args.dataset_dir}")
    LOGGER.info(f"📏 Depth max: {args.depth_max} meters")
    LOGGER.info(f"📊 Train ratio: {args.train_ratio}")
    LOGGER.info(f"⬇️  Download: {args.download}")
    LOGGER.info(f"🔄 Overwrite: {args.overwrite}")
    LOGGER.info(f"📝 Use existing splits: {not args.no_existing_splits}")

    try:
        summary = prepare_sunrgbd_mde_dataset(
            dataset_dir=args.dataset_dir,
            depth_max=args.depth_max,
            train_ratio=args.train_ratio,
            download_assets=args.download,
            overwrite=args.overwrite,
            shuffle=True,
            seed=args.seed,
            use_existing_splits=not args.no_existing_splits,
            min_box_size=args.min_box_size,
        )

        # Print summary statistics
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("✅ Dataset Preparation Complete!")
        LOGGER.info("=" * 60)
        LOGGER.info(f"📂 Dataset location: {summary['dataset_dir']}")
        LOGGER.info(f"📏 Depth max: {summary['depth_max']} meters")
        LOGGER.info(f"\n📊 Training Set:")
        LOGGER.info(f"   Images: {summary['train']['images']}")
        LOGGER.info(f"   Objects: {summary['train']['objects']}")
        if summary["train"]["class_counts"]:
            LOGGER.info(f"   Classes: {len(summary['train']['class_counts'])}")
            LOGGER.info(f"   Top classes: {dict(list(summary['train']['class_counts'].items())[:5])}")

        LOGGER.info(f"\n📊 Validation Set:")
        LOGGER.info(f"   Images: {summary['val']['images']}")
        LOGGER.info(f"   Objects: {summary['val']['objects']}")
        if summary["val"]["class_counts"]:
            LOGGER.info(f"   Classes: {len(summary['val']['class_counts'])}")
            LOGGER.info(f"   Top classes: {dict(list(summary['val']['class_counts'].items())[:5])}")

        LOGGER.info("\n🎯 Next Steps:")
        LOGGER.info("   1. Review the prepared dataset")
        LOGGER.info(f"   2. Use the config: ultralytics/cfg/datasets/sunrgbd-mde.yaml")
        LOGGER.info("   3. Train with: yolo train model=yolo11n.pt data=sunrgbd-mde.yaml task=mde")

    except Exception as e:
        LOGGER.error(f"❌ Error preparing dataset: {e}")
        raise


if __name__ == "__main__":
    main()

