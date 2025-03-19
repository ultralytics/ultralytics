import os

import torch

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPESegTrainer
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import LOGGER, yaml_load

os.environ["PYTHONHASHSEED"] = "0"

data = "ultralytics/cfg/datasets/coco.yaml"

model_path = "yoloe-v8s-seg.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/coco_{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert all(k in defaults for k in extends)
LOGGER.info(f"Extends: {extends}")

model = YOLOE("yoloe-v8s-seg.pt")

# Ensure pe is set for classes
names = list(yaml_load(data)["names"].values())
tpe = model.get_text_pe(names)
pe_path = "coco-pe.pt"
torch.save({"names": names, "pe": tpe}, pe_path)

model.train(
    data=data,
    epochs=80,
    close_mosaic=10,
    batch=128,
    optimizer="AdamW",
    lr0=1e-3,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=4,
    device="0,1,2,3,4,5,6,7",
    **extends,
    trainer=YOLOEPESegTrainer,
    train_pe_path=pe_path,
)
