from ultralytics import YOLO
from training_utils import (
    PrepareDataset,
    GetModelYaml,
    LoadBestModel,
    GetLatestRunDir,
)
from training_utils import (
    dataset_yaml_path,
    coco_classes_file,
    training_task,
    experiment_name,
)
import os


PrepareDataset(coco_classes_file, dataset_yaml_path, training_task)
model = YOLO(GetModelYaml(training_task))  # Initialize model

model.train(
    task=training_task,
    data="verdant.yaml",
    epochs=300,
    flipud=0.5,
    fliplr=0.5,
    scale=0.2,
    mosaic=0.0,  # Please set this to 0.0 TODO: Fix the issue with mosaic and keypoint detection
    imgsz=768,
    seed=1,
    batch=128,
    name=experiment_name,
    device=[0, 1, 2, 3, 4, 5, 6, 7],
)


model = LoadBestModel()  # To load the best model
path = model.export(format="onnx", imgsz=[768, 768], opset=12)


latest_run_dir = GetLatestRunDir()
os.system(f"mv {path} {latest_run_dir}/weights/best.onnx")
path = model.export(format="onnx", imgsz=[2144, 768], opset=12)
os.system(f"mv {path} {latest_run_dir}/weights/best_full_height.onnx")
