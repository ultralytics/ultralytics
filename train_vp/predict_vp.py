

import sys,os
sys.path.append("/root/ultra_louis_work/ultralytics")
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # change to the directory of the current script


import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor,YOLOEVPDetectPredictor

# Initialize a YOLOE model
# model = YOLOE("yoloe-11l-seg.pt")

model = YOLOE("/root/ultra_louis_work/ultralytics/runs/detect/train/weights/last.pt")

# Define visual prompts using bounding boxes and their corresponding class IDs.
# Each box highlights an example of the object you want the model to detect.
visual_prompts = dict(
    bboxes=np.array(
        [
            [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
            [120, 425, 160, 445],  # Box enclosing glasses
        ],
    ),
    cls=np.array(
        [
            0,  # ID to be assigned for person
            1,  # ID to be assigned for glassses
        ]
    ),
)

# Run inference on an image, using the provided visual prompts as guidance
results = model.predict(
    "/root/ultra_louis_work/ultralytics/ultralytics/assets/bus.jpg",
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor, plots=True
)

# Show results
results[0].save("bus_vp.jpg")  # save results to file