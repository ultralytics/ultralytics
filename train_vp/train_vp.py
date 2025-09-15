
import sys,os
sys.path.append("/root/ultra_louis_work/ultralytics")
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # change to the directory of the current script

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPTrainer




DATA_DIR="/root/autodl-tmp/datasets/"

Objects365v1="./Objects365v1.yaml"

data = dict(
    train=dict(
        yolo_data=[Objects365v1],
        grounding_data=[
            dict(
                img_path=DATA_DIR+"flickr/full_images/",
                json_file=DATA_DIR+"flickr/annotations/final_flickr_separateGT_train_segm.json",
            ),
            dict(
                img_path=DATA_DIR+"mixed_grounding/gqa/images",
                json_file=DATA_DIR+"mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
            ),
        ],
    ),
    val=dict(yolo_data=["./lvis.yaml"]),
)

# model = YOLOE("yoloe-11s-seg.pt")
# replace to yoloe-11l-seg-det.pt if converted to detection model
# model = YOLOE("yoloe-v8s.yaml")
# # model_path="/home/user/shuailyu/ultralytics/runs/segment/train23/weights/best.pt"
# model_path="yoloe-v8s-seg.pt"
# from ultralytics.utils.patches import torch_load
# state = torch_load(model_path)


# # Load model weights but skip savpe module weights
# model_state_dict = state["model"].state_dict()

# # Filter out savpe weights from the loaded state dict
# filtered_state_dict = {}
# for key, value in model_state_dict.items():
#     # if "savpe" not in key:
#     filtered_state_dict[key] = value

# # Load the filtered state dict
# model.model.load_state_dict(filtered_state_dict, strict=False)





# model.load("./tempmodel1.pt")


# metrics = model.val(batch=1,data="./lvis.yaml", load_vp=True, split='minival',save_json=True,
#                     refer_data="./lvis_train_vps.yaml",max_det=1000)


model = YOLOE("/root/ultra_louis_work/yoloe/yoloe-v8s-seg-det.pt")


# reinit the model.model.savpe.
model.model.model[-1].savpe.init_weights()

# freeze every layer except of the savpe module.
head_index = len(model.model.model) - 1
freeze = list(range(0, head_index))
for name, child in model.model.model[-1].named_children():
    if "savpe" not in name:
        freeze.append(f"{head_index}.{name}")





model.train(
    data=data,
    batch=64,
    epochs=2,
    close_mosaic=2,
    optimizer="AdamW",
    lr0=8e-3, # for s/m, please set lr0=8e-3
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    # workers=4,
    trainer=YOLOEVPTrainer,  # use YOLOEVPTrainer if converted to detection model
    device="0",
    freeze=freeze, 
    val=False
)


# model.save("./tempmodel2.pt")


# model=YOLOE("./tempmodel2.pt")


# metrics = model.val(batch=1,data="./lvis.yaml", load_vp=True, split='minival',save_json=True,
#                     refer_data="./lvis_train_vps.yaml",max_det=1000)
