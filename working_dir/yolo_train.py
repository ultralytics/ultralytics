import torch
import argparse
from ultralytics import RTDETR
import yaml


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
        default='rtdetr_res50_train',
        help='Name for the training run'
    )

    parser.add_argument(
        '--train', nargs='*', default=[], 
        help='Additional Ultralytics train args, e.g. freeze=10 imgsz=1024 pretrained=/path/to/weights.pt')

    return parser.parse_args()


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs:
        key, value = pair.split('=', 1)
        overrides[key] = yaml.safe_load(value)  # auto-casts ints/bools/lists
    return overrides


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

    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            print(f"Layer: {name} | Requires Grad: {param.requires_grad}")

    # Prepare training kwargs
    train_kwargs = {'cfg': args.config, 'name': args.name}
    train_kwargs.update(parse_overrides(args.train))

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
