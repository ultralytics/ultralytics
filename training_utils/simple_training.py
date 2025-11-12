from ultralytics import YOLO
from training_utils import (
    PrepareDataset,
    GetModelYaml,
    GetLatestWeightsDir,
)
from training_utils import (
    dataset_yaml_path,
    coco_classes_file,
    training_task,
    experiment_name,
)
import os
import argparse
from export import Export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("-l", "--load", type=str, default=None, help="Path to the model weights to load. Load the pretrained model")
    parser.add_argument("-r", "--learning-rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("-p", "--patience", type=int, default=50, help="Number of epochs triggering early stopping when no improvement")
    parser.add_argument("-s", "--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("-b", "--disable-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    print(f"Model weights initialized from: {args.load if args.load else 'scratch'}")
    print(f"Learning rate: {args.learning_rate}, Epochs: {args.epochs}, Patience: {args.patience}, Batch size: {args.batch_size}")

    PrepareDataset(coco_classes_file, dataset_yaml_path, training_task)

    if args.load is not None:
        if os.path.exists(args.load):
            model = YOLO(args.load)
        else:
            print(f"[ERROR] : Model {args.load} does not exists")
            exit(1)
    else:
        model = YOLO(GetModelYaml(training_task))  # Initialize model

    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    model.train(
        task=training_task,
        data="verdant.yaml",
        optimizer='SGD',
        lr0=args.learning_rate,
        lrf=0.01,
        epochs=args.epochs,
        flipud=0.5,
        fliplr=0.5,
        scale=0.2,
        mosaic=0.0, # Please set this to 0.0 TODO: Fix the issue with mosaic and keypoint detection
        imgsz=768,
        seed=1,
        batch=args.batch_size,
        name=experiment_name,
        device=[0, 1, 2, 3, 4, 5, 6, 7],
        patience=args.patience,
    )

    print("Training completed. Exporting the model by converting checkpoints to ONNX format...")
    latest_weights_dir = GetLatestWeightsDir()
    Export(f"{latest_weights_dir}/best.pt")
    Export(f"{latest_weights_dir}/last.pt")