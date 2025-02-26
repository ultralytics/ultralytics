import numpy as np

from ultralytics import YOLO
from ultralytics import settings
from pathlib import Path

root = Path()
datasets_root = root.parent.resolve()

settings.update({
    "datasets_dir": str(datasets_root / "datasets"),
    'tensorboard': False, 'comet': False, "mlflow": False, "clearml": False, "neptune": False,
    'dvc': False, 'hub': False, 'sync': False,
    'wandb': False
})
if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
    model.val(data="coco8-pose.yaml",)
    # Train the model
    #results = model.train(data="coco8-pose.yaml", epochs=5, imgsz=640)
