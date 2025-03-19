import os

import torch

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPEFreeTrainer
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import LOGGER, yaml_load

os.environ["PYTHONHASHSEED"] = "0"

data = dict(
    train=dict(
        yolo_data=["Objects365v1.yaml"],
        grounding_data=[
            dict(
                img_path="../datasets/flickr/full_images/",
                json_file="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json",
            ),
            dict(
                img_path="../datasets/mixed_grounding/gqa/images",
                json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
            ),
        ],
    ),
    val=dict(yolo_data=["lvis.yaml"]),
)

model_path = "yoloe-v8l.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert all(k in defaults for k in extends)
LOGGER.info(f"Extends: {extends}")

model = YOLOE("yoloe-v8l-seg-det.pt")

# Ensure pe is set for classes
names = ["object"]
tpe = model.get_text_pe(names)
pe_path = "free-pe.pt"
torch.save({"names": names, "pe": tpe}, pe_path)

head_index = len(model.model.model) - 1
freeze = [str(f) for f in range(0, head_index)]
for name, child in model.model.model[-1].named_children():
    if "cv3" not in name:
        freeze.append(f"{head_index}.{name}")

freeze.extend(
    [
        f"{head_index}.cv3.0.0",
        f"{head_index}.cv3.0.1",
        f"{head_index}.cv3.1.0",
        f"{head_index}.cv3.1.1",
        f"{head_index}.cv3.2.0",
        f"{head_index}.cv3.2.1",
    ]
)

model.train(
    data=data,
    batch=128,
    epochs=1,
    **extends,
    close_mosaic=1,
    optimizer="AdamW",
    lr0=2e-3,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=4,
    trainer=YOLOEPEFreeTrainer,
    device="0,1,2,3,4,5,6,7",
    freeze=freeze,
    single_cls=True,
    train_pe_path=pe_path,
)
