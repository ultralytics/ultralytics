

import sys,os
import torch
import torch.nn as nn
sys.path.append("/root/ultra_louis_work/ultralytics")
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # change to the directory of the current script



from ultralytics.utils.patches import torch_load
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPTrainer

# model init 
model = YOLOE("yoloe-v8s.yaml").load("yoloe-v8s-seg.pt")




from ultralytics import YOLOE
from ultralytics.utils.patches import torch_load

det_model = YOLOE("yoloe-v8s.yaml")
state = torch_load("yoloe-v8s-seg.pt")
det_model.load(state["model"])
det_model.save("yoloe-11l-seg-det.pt")