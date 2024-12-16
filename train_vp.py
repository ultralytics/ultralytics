from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_seg_vp import YOLOESegVPTrainer
from ultralytics.models.yolo.yoloe.train_vp import YOLOEVPTrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER

os.environ["PYTHONHASHSEED"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

model_path = "yoloe-v8s.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

model = YOLOE("yoloe-v8s-seg-det.pt")
# YOLOESegVPTrainer => YOLOEVPTrainer

head_index = len(model.model.model) - 1
freeze = list(range(0, head_index))
for name, child in model.model.model[-1].named_children():
    if 'savpe' not in name:
        freeze.append(f"{head_index}.{name}")

# For s/m, please set lr0=8e-3
model.train(data=data, batch=128, epochs=2, **extends, close_mosaic=2, \
    optimizer='AdamW', lr0=16e-3, warmup_bias_lr=0.0, \
        weight_decay=0.025, momentum=0.9, workers=4, \
        trainer=YOLOEVPTrainer, device='0,1,2,3,4,5,6,7', freeze=freeze, load_vp=True)