from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_yoloe import YOLOETrainerFromScratch
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER

os.environ["PYTHONHASHSEED"] = "0"

data = dict(
    train=dict(
        yolo_data=["Objects365v1.yaml"],
        grounding_data=[
            dict(
                img_path="../datasets/flickr/full_images/",
                json_file="../datasets/flickr/annotations/final_flickr_separateGT_train.json",
            ),
            dict(
                img_path="../datasets/mixed_grounding/gqa/images",
                json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco.json",
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
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

model = YOLOE(model_path)

model.train(data=data, batch=128, epochs=30, **extends, close_mosaic=2, \
    optimizer='AdamW', lr0=2e-3, warmup_bias_lr=0.0, \
        weight_decay=0.025, momentum=0.9, \
        trainer=YOLOETrainerFromScratch, device='0,1,2,3,4,5,6,7')