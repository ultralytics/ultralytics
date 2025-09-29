
import os ,ultralytics
from ultralytics.engine import results
os.chdir(os.path.dirname(os.path.dirname(ultralytics.__file__)))


import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor,YOLOEVPDetectPredictor



# Define visual prompts using bounding boxes and their corresponding class IDs.
# Each box highlights an example of the object you want the model to detect.
# visual_prompts = dict(
#     bboxes=np.array(
#         [
#             [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
#             [120, 425, 160, 445],  # Box enclosing glasses
#         ],
#     ),
#     cls=np.array(
#         [
#             0,  # ID to be assigned for person
#             1,  # ID to be assigned for glassses
#         ]
#     ),
# )

# # Run inference on an image, using the provided visual prompts as guidance
# results = model.predict(
#     "ultralytics/assets/bus.jpg",
#     visual_prompts=visual_prompts,
#     predictor=YOLOEVPDetectPredictor,
# )

# # Show results
# results[0].save(filename="yoloe_vp_detect.jpg")


# Initialize a YOLOE model
model = YOLOE("yoloe-v8s-seg.pt")

results1 = model.predict(
    "ultralytics/assets/bus.jpg",
    visual_prompts = dict(bboxes=np.array([[221.52, 405.8, 344.98, 857.54]]),
    cls=np.array([0])),
    predictor=YOLOEVPDetectPredictor,
)

results2 = model.predict(
    "ultralytics/assets/bus.jpg",
    visual_prompts = dict(bboxes=np.array([[120, 425, 160, 445]]),
    cls=["glasses"]),
)

results3 = model.predict(
    "ultralytics/assets/zidane.jpg",conf=0.1)


results1[0].save(filename="yoloe_vp_detect1.jpg")
results2[0].save(filename="yoloe_vp_detect2.jpg")
results3[0].save(filename="yoloe_vp_detect3.jpg")


img1 = results1[0].plot()
img2 = results2[0].plot()
img3 = results3[0].plot()

# lettoxbox to 640x640

from ultralytics.data.augment import LetterBox
import numpy as np

# Create the LetterBox transform
letterbox = LetterBox(new_shape=(640, 480))

# Apply letterbox to get the padded image
img1 = letterbox(image=img1)
img2 = letterbox(image=img2)
img3 = letterbox(image=img3)

# concat the results and save to disk

import cv2
import numpy as np
final_img = np.concatenate((img1, img2, img3), axis=1)
cv2.imwrite("yoloe_vp_detect_combined.jpg", final_img)
