# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.yolo.data.build import load_inference_source
from ultralytics.yolo.utils import LINUX, ONLINE, ROOT, SETTINGS

from multiprocessing import freeze_support


def config01():
    model = YOLO('yolov8n.yaml')
    model.train(
        data='D:\MyProjects\yolov8-custom\datasets\defect.yaml',
        epochs=50,
        batch=16,
        workers=4,
        optimizer='Adam',
        cos_lr=True
    )


if __name__ == '__main__':
    freeze_support()
    config01()