import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

model = YOLOE("pretrain/yoloe-v8l-seg.pt")

# Handcrafted shape can also be passed, please refer to app.py
# Multiple boxes or handcrafted shapes can also be passed as visual prompt in an image
visuals = dict(
    bboxes=np.array(
        [
            [221.52, 405.8, 344.98, 857.54],  # For person
            [120, 425, 160, 445],  # For glasses
        ],
    ),
    cls=np.array(
        [
            0,  # For person
            1,  # For glasses
        ]
    ),
)

source_image = "ultralytics/assets/bus.jpg"

model.predict(source_image, save=True, prompts=visuals, predictor=YOLOEVPSegPredictor)

# Prompts in different images can be passed
# Please set a smaller conf for cross-image prompts
# model.predictor = None  # remove VPPredictor
target_image = "ultralytics/assets/zidane.jpg"
model.predict(source_image, prompts=visuals, predictor=YOLOEVPSegPredictor, return_vpe=True)
model.set_classes(["object0", "object1"], model.predictor.vpe)
model.predictor = None  # remove VPPredictor
model.predict(target_image, save=True)
