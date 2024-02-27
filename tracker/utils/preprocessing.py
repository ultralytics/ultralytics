import subprocess

import numpy as np
import torch

from ultralytics.utils import is_online, emojis, LOGGER, TQDM
from ultralytics.utils.downloads import unzip_file
from urllib import parse, request

chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}


def extract_image_patches(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int32)
    patches = [image[box[1]:box[3], box[0]:box[2], :] for box in bboxes]
    # bboxes = clip_boxes(bboxes, image.shape)
    return patches