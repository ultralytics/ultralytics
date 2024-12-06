from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch import Tensor

def read_lidarmap(img_path) -> Tensor:
    path = Path(img_path)
    path = path.parents[1]/'map'/path.name
    return Tensor(Image.open(path)[2:])

def read_lidarpoint(img_path) -> Tensor:
    path = Path(img_path)
    path = path.parents[1]/'point'/path.name
    return Tensor(Image.open(path))

def read_combo(img_path) -> Tensor:
    path = Path(img_path)
    im = Tensor(Image.open(path))
    map = read_lidarmap(path)
    point = read_lidarpoint(img_path)
    return torch.concat([im, map]), point

if __name__ == '__main__':
    path = ''