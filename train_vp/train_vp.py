
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPTrainer




DATA_DIR="../datasets/"

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



# model = YOLOE("/root/ultra_louis_work/yoloe/yoloe-v8s-seg-det.pt")
model = YOLOE("yoloe-v8s.yaml").load("./yoloe-v8s-seg.pt")

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
    batch=128,
    epochs=2,
    close_mosaic=0,
    optimizer="AdamW",
    lr0=8e-3, # for s/m, please set lr0=8e-3
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=8,
    trainer=YOLOEVPTrainer,  # use YOLOEVPTrainer if converted to detection model
    device="0,1,2,3",
    freeze=freeze, 
    val=False
)


# model.save("./tempmodel2.pt")


# model=YOLOE("./tempmodel2.pt")


# metrics = model.val(batch=1,data="./lvis.yaml", load_vp=True, split='minival',save_json=True,
#                     refer_data="./lvis_train_vps.yaml",max_det=1000)
