import itertools
import os
import subprocess
from gs_train import train_script
import sys

from ultralytics import YOLO

# Set number of threads
os.environ['OMP_NUM_THREADS'] = '4'

gs_config = {
    'batch': [66, 126, 192],
    'cos_lr': [False, True],
    'lr0': [0.03, 0.01, 0.005],
    'lrf': [0.01, 0.005]
}
gs_config = {
    'batch': [4, 8],
    'cos_lr': [False],
    'lr0': [0.01],
    'lrf': [0.01]
}

# Get the keys and lists from the dictionary
keys = list(gs_config.keys())
lists = list(gs_config.values())

# Generate all possible combinations
combinations = list(itertools.product(*lists))

# Iterate over the combinations and print the output
for combination in combinations:
    # Unpack the combination
    batch, cos_lr, lr0, lrf = combination
    epochs = 3
    command = f"yolo detect train data=coco8.yaml model=yolov8s.yaml pretrained=../models/yolov8s.pt epochs={epochs} batch={batch} fraction=0.9 project=grid-search-cdv1 name=8s_3{epochs}_{batch}b_coslr-{cos_lr}_lr0-{lr0}_lrf-{lrf} device=[4,5] patience=10 lr0={lr0} lrf={lrf} cos_lr={cos_lr}"
    #os.system('source /home-net/ierregue/.virtualenvs/small-fast-detector/bin/activate; ' + command)
    subprocess.run('pwd', executable='/home-net/ierregue/.virtualenvs/small-fast-detector/bin/activate', shell=True)