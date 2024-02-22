import os
from ultralytics import YOLO
import comet_ml
from ultralytics.utils import SETTINGS

# Only for tune method because it uses all GPUs by default
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

from ultralytics.utils import SETTINGS
SETTINGS['comet'] = True  # set True to log using Comet.ml
comet_ml.init()

# Initialize model and load matching weights
model = YOLO('yolov8s.yaml', task='detect').load('./../models/yolov8s.pt')
results = model.train(
    save=True,
    verbose=True,
    plots=True,
    project='debug',
    name='8s',
    data='coco8.yaml',
    epochs=3,
    batch=4,
    imgsz=320,
)