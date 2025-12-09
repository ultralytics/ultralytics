import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



from ultralytics import YOLOE,YOLO
from ultralytics.models.yolo.yoloe import YOLOETrainerFromScratch,YOLOEVPTrainer,YOLOEPEFreeTrainer
from ultralytics.models.yolo.yoloe import YOLOESegTrainerFromScratch





# model = YOLOE("/root/ultra_louis_work/yoloe/yoloe-v8s-seg-det.pt")

import argparse



parser=argparse.ArgumentParser(description="train yoloe with visual prompt")
parser.add_argument("--model_version", type=str, default="26s")
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--close_mosaic", type=int, default=0)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--project", type=str, default="../runs/train_tp")
#device
parser.add_argument("--device", type=str, default="7")
# val
parser.add_argument("--val", type=bool, default=True)
parser.add_argument("--name", type=str, default="yoloe_vp")
parser.add_argument("--clip_weight_name", type=str, default="mobileclip:blt") 
parser.add_argument("--scale", type=float, default=0.9)

parser.add_argument("--ag", type=bool, default=False) # all grounding

parser.add_argument("--weight_path", type=str,default="yolo26s-objv1.pt")  # model weight path
parser.add_argument("--trainer", type=str,default="YOLOETrainerFromScratch")  # model weight path
parser.add_argument("--max_det", type=int,default=1000)  # max det
parser.add_argument("--workers", type=int,default=8)  # workers
parser.add_argument("--save_period", type=int,default=5)  # save period

parser.add_argument("--data", type=str, default=None) # fast_verify or objv1_only


parser.add_argument("--optimizer",type=str, default="MuSGD") # "MuSGD"
parser.add_argument("--momentum",type=float, default=0.9)
parser.add_argument("--weight_decay",type=float, default=0.0007)
parser.add_argument("--lrf", type=float, default=0.5)
parser.add_argument("--lr0", type=float, default=0.00125)  # initial learning rate
parser.add_argument("--o2m", type=float, default=0.1)

 # mobileclip2b
#
args = parser.parse_args()
assert args.trainer in ["YOLOETrainerFromScratch","YOLOEVPTrainer","YOLOEPEFreeTrainer","YOLOESegTrainerFromScratch"], \
    "trainer must be YOLOETrainerFromScratch, YOLOEVPTrainer, YOLOEPEFreeTrainer, or YOLOESegTrainerFromScratch"





if args.ag:
    data = dict(
        train=dict(
            grounding_data=[
                dict(
                    img_path="../datasets/flickr/full_images/",
                    json_file="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json",
                ),
                dict(
                    img_path="../datasets/mixed_grounding/gqa/images",
                    json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
                ),
                dict(
                    img_path="../datasets/Objects365v1/images/train",
                    json_file="../datasets/Objects365v1/annotations/objects365_train_segm.json",
                ),
            ],
        ),
        val=dict(yolo_data=["../datasets/lvis.yaml"]),
    )

else:
    Objects365v1="../datasets/Objects365v1.yaml"
    data = dict(
        train=dict(
            yolo_data=[Objects365v1],
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
        val=dict(yolo_data=["../datasets/lvis.yaml"]),
    )

if args.data=="fast_verify":



    data = dict(
        train=dict(
            yolo_data=["coco128.yaml"]
        ),
        val=dict(yolo_data=[os.path.abspath("../datasets/lvis.yaml")]),
    )
elif args.data=="objv1_only":
    Objects365v1="../datasets/Objects365v1.yaml"
    data = dict(
        train=dict(
            yolo_data=[Objects365v1],
        ),
        val=dict(yolo_data=["../datasets/lvis.yaml"]),
    )
elif args.data is None or args.data=="":
    pass
else:
    raise ValueError("data argument must be fast_verify or objv1_only")


###################################################################
# args.trainer="YOLOESegTrainerFromScratch"
# args.project="runs/quick_verfiy"
# args.model_version="26s-seg"
# data = dict(
#     train=dict(
#         yolo_data=["coco128-seg.yaml"]
#     ),
#     val=dict(yolo_data=[os.path.abspath("../datasets/lvis.yaml")]),
# )

###################################################################
model = YOLO("yoloe-{}.yaml".format(args.model_version))
model=model.load(args.weight_path)


if args.trainer == "YOLOETrainerFromScratch" or args.trainer== "YOLOESegTrainerFromScratch":
    print("Using YOLOETrainerFromScratch for training.")
    freeze = []
    refer_data=None
    single_cls=False
elif args.trainer == "YOLOEVPTrainer":
    print("Using YOLOEVPTrainer for training.")
    # reinit the model.model.savpe.
    model.model.model[-1].savpe.init_weights()

    # freeze every layer except of the savpe module.
    head_index = len(model.model.model) - 1
    freeze = list(range(0, head_index))
    for name, child in model.model.model[-1].named_children():

        if "savpe" not in name:
            freeze.append(f"{head_index}.{name}")

    refer_data=os.path.abspath("../datasets/lvis_train_vps.yaml")
    single_cls=False
elif args.trainer == "YOLOEPEFreeTrainer":

    # freeze layers.
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
    freeze.extend(
        [
            f"{head_index}.one2one_cv3.0.0",
            f"{head_index}.one2one_cv3.0.1",
            f"{head_index}.one2one_cv3.1.0",
            f"{head_index}.one2one_cv3.1.1",
            f"{head_index}.one2one_cv3.2.0",
            f"{head_index}.one2one_cv3.2.1",
        ]
    )

    refer_data=None
    single_cls=True
else:
    print("trainer_class:", args.trainer)
    raise ValueError("trainer_class must be YOLOETrainerFromScratch or YOLOEVPTrainer")

trainer_class =eval( args.trainer)

model.train(
    data=data,
    batch=args.batch,
    epochs=args.epochs,
    close_mosaic=args.close_mosaic,

    workers=args.workers,
    max_det=args.max_det,
    trainer=trainer_class,  # use YOLOEVPTrainer if converted to detection model
    clip_weight_name=args.clip_weight_name,
    device=args.device,
    save_period=args.save_period,
    val=args.val,
    project=args.project,
    name=args.name,

    cache="disk",
    scale=args.scale, # sensitive.  [0.1,  1.9] 
    copy_paste=0.15, 
    mixup=0.05,
    dfl=6.0,
    o2m=args.o2m,
    warmup_epochs=1,
    warmup_bias_lr=0.0,
    lr0=args.lr0,
    lrf=args.lrf,
    optimizer=args.optimizer,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    single_cls=single_cls, # for YOLOEPEFreeTrainer
    freeze=freeze, # for YOLOEVPTrainer
    refer_data=refer_data, # for YOLOEVPTrainer

)
