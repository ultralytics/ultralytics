from ultralytics import YOLO

from ultralytics.utils import SETTINGS
SETTINGS['comet'] = False  # set True to log using Comet.ml

# Only for tune method because it uses all GPUs by default
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize the YOLO model
model = YOLO('yolov8n.pt')

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data='coco8.yaml',
    fraction=0.9,
    epochs=3,
    plots=False,
    save=False,
    val=False,
    grace_period=2,
    iterations=2,
    #max_samples=2,
    #gpu_per_trial=1 # number of GPUs per trial, DDP not supported
)