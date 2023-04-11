import os
from pathlib import Path
import albumentations as A

import torch.cuda
import torchvision.transforms as transforms

parent_dir = Path(__file__).parent.parent
ROOT_DIR = os.path.join(parent_dir, "datasets", "copper")

# if no yaml file, this must be manually inserted
# nc is number of classes (int)
nc = None
# list containing the labels of classes: i.e. ["cat", "dog"]
labels = None

FIRST_OUT = 48

CLS_PW = 1.0
OBJ_PW = 1.0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 640

CONF_THRESHOLD = 0.01  # to get all possible bboxes, trade-off metrics/speed --> we choose metrics
NMS_IOU_THRESH = 0.6
# for map 50
MAP_IOU_THRESH = 0.5

# triple check what anchors REALLY are

ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32#
]


TRAIN_TRANSFORMS = A.Compose(
    [
# v8
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_lower=75, p=0.0)
# # v5
#         # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.4),
#         # A.Transpose(p=1),
#         # A.HorizontalFlip(p=0.5),
#         # A.VerticalFlip(p=0.5),
#         # A.Rotate(limit=(-20, 20), p=0.7),
#         A.Blur(p=0.05),
#         A.CLAHE(p=0.1),
#         # A.Posterize(p=0.1),
#         # A.ChannelShuffle(p=0.05),
    ],
    bbox_params=A.BboxParams("yolo", min_visibility=0.0, label_fields=[]),
)

FLIR = [
    'car',
    'person'
]

COCO = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

nc = len(COCO)
labels = COCO
