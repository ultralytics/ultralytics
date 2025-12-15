import torch
import argparse
from ultralytics import RTDETR


def parse_args():
    """
    Parse command line arguments for RT-DETR training
    """
    parser = argparse.ArgumentParser(description='RT-DETR Training Pipeline')

    # Model and config arguments
    parser.add_argument(
        '--model',
        type=str,
        default='ultralytics/cfg/models/11/yolo11-rtdetr-res50.yaml',
        help='Path to model configuration YAML file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='working_dir/rtdetr_train_config.yaml',
        help='Path to training configuration YAML file'
    )

    parser.add_argument(
        '--name',
        type=str,
        default='rtdetr_res50_pretrained',
        help='Name for the training run'
    )

    parser.add_argument(
        '--project',
        type=str,
        default='/Users/esat/workspace/detr_trainings',
        help='Project directory to save training results'
    )

    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='Path to pretrained weights file'
    )

    return parser.parse_args()


def main():
    """
    RT-DETR Training Pipeline using YAML Configuration
    """
    # Parse command line arguments
    args = parse_args()

    print(f"\nModel: {args.model}")
    print(f"Config: {args.config}")

    # Check available devices
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load RT-DETR model
    model = RTDETR(args.model)
    if args.pretrained:
        model.load(args.pretrained)

    # Prepare training kwargs
    train_kwargs = {'cfg': args.config, 'name': args.name, 'project': args.project}

    # Train using YAML configuration
    results = model.train(**train_kwargs)

    if results is not None:
        # Print training results
        print("\n" + "="*50)
        print("Training completed!")
        print("="*50)
        print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")

        # Validate the trained model
        print("\n" + "="*50)
        print("Running validation...")
        print("="*50)
        metrics = model.val()

        # Print validation metrics
        print(f"Validation mAP50: {metrics.box.map50:.4f}")
        print(f"Validation mAP50-95: {metrics.box.map:.4f}")

        print("\nTraining pipeline completed successfully!")
        print(f"Results saved to: {results.save_dir}")


if __name__ == '__main__':
    main()
