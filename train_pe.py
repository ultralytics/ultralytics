from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPETrainer, YOLOEPESegTrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER
import torch

os.environ["PYTHONHASHSEED"] = "0"

data = "ultralytics/cfg/datasets/coco.yaml"

model_path = "yoloe-v8l-seg.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/coco_{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

model = YOLOE("yoloe-v8l-seg.pt")

# Ensure pe is set for classes
names = list(yaml_load(data)['names'].values())
tpe = model.get_text_pe(names)
pe_path = "coco-pe.pt"
torch.save({"names": names, "pe": tpe}, pe_path)

head_index = len(model.model.model) - 1
freeze = [str(f) for f in range(0, head_index)]
for name, child in model.model.model[-1].named_children():
    if 'cv3' not in name:
        freeze.append(f"{head_index}.{name}")

freeze.extend([f"{head_index}.cv3.0.0", f"{head_index}.cv3.0.1", f"{head_index}.cv3.1.0", f"{head_index}.cv3.1.1", f"{head_index}.cv3.2.0", f"{head_index}.cv3.2.1"])
        
model.train(data=data, epochs=10, close_mosaic=5, batch=128, 
            optimizer='AdamW', lr0=1e-3, warmup_bias_lr=0.0, \
            weight_decay=0.025, momentum=0.9, workers=4, \
            device="0,1,2,3,4,5,6,7", **extends, \
            trainer=YOLOEPESegTrainer, freeze=freeze, train_pe_path=pe_path)