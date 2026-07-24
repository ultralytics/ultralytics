






OVERRIDES=dict()




# for yoloe-26x-tp (provided by Jing with better performance than default hyperparameters)
# modify close_mosaic to 2
or260105=dict(lr0=0.00038, lrf=0.88219, momentum=0.94751, weight_decay=0.00027, warmup_bias_lr=0.05684, warmup_epochs=0.98745, warmup_momentum=0.54064, box=9.83241, cls=0.64896, dfl=0.95824, hsv_h=0.01315, hsv_s=0.35348, hsv_v=0.19383, degrees=0.00012, translate=0.27484, scale=0.95, shear=0.00136, perspective=0.00074, flipud=0.00653, fliplr=0.30393, bgr=0.0, mosaic=0.99182, mixup=0.42713, cutmix=0.00082, copy_paste=0.40413, close_mosaic=2, o2m=0.70518, muon_w=0.4355, sgd_w=0.47908, cls_w=3.48357, epochs=15)
OVERRIDES["or260105"]=or260105

# for yoloe-26l-tp (provided by Jing with better performance than default hyperparameters)
# modify close_mosaic to 2
or260107_forl=dict(lr0=0.00038, lrf=0.88219, momentum=0.94751, weight_decay=0.00027, warmup_bias_lr=0.05684, warmup_epochs=0.98745, warmup_momentum=0.54064, box=9.83241, cls=0.64896, dfl=0.95824,hsv_h=0.01315, hsv_s=0.35348, hsv_v=0.19383, degrees=0.00012, translate=0.27484, scale=0.95, shear=0.00136, perspective=0.00074,  flipud=0.00653,fliplr=0.30393,  bgr=0.0, mosaic=0.99182,  mixup=0.42713, cutmix=0.00082, copy_paste=0.40413, close_mosaic=2, o2m=0.70518, muon_w=0.4355, sgd_w=0.47908, cls_w=3.48357) # batch=128 


OVERRIDES["or260107_forl"]=or260107_forl



########################## data config  ##########################
import os 


# train_data_root="../datasets"
train_data_root="/data/shared-datasets/yoloe26_data"
flickr_v4_json="flickr/pipeline_outputs/v4/merged.json"
mixed_grounding_v4_json="mixed_grounding/pipeline_outputs/v4/merged.json"
obj365_v4_json="Objects365v1/pipeline_outputs/train/v4/merged.json"

flickr_v5_json="flickr/pipeline_outputs/v5/merged.json"
mixed_grounding_v5_json="mixed_grounding/pipeline_outputs/v5/merged.json"
obj365_v5_json="Objects365v1/pipeline_outputs/train/v5/merged.json"

ye_v4_json="yolo-enterprise/pipeline_outputs/train/v4/merged.json"
ye_v5_json="yolo-enterprise/pipeline_outputs/train/v5/merged_simplified.json"


refer_data_yaml=os.path.abspath(f"/data/shared-datasets/louis_data/lvis_train_vps.yaml")



DATA_CONFIG=dict()

# old-engine data: absolute paths on the shared NFS mount /data/shared-datasets
# (portable across ultra6 users; formerly relative ../datasets/...)
lvis_data=os.path.abspath("/data/shared-datasets/louis_data/lvis.yaml")


old_flickr_data= dict(
                img_path="/data/shared-datasets/louis_data/flickr/full_images/",
                json_file="/data/shared-datasets/louis_data/flickr/annotations/final_flickr_separateGT_train_segm.json",
            )
old_mixed_data= dict(
                img_path="/data/shared-datasets/louis_data/mixed_grounding/gqa/images",
                json_file="/data/shared-datasets/louis_data/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
            )
old_obj365_data= dict(
                img_path="/data/shared-datasets/louis_data/Objects365v1/images/train",
                json_file="/data/shared-datasets/louis_data/Objects365v1/annotations/objects365_train_segm.json",
            )  




new_flickr_v4= dict(
                img_path=f"{train_data_root}/flickr/full_images/",
                json_file=f"{train_data_root}/{flickr_v4_json}",
            )
new_mixed_v4= dict(
                img_path=f"{train_data_root}/mixed_grounding/gqa/images",
                json_file=f"{train_data_root}/{mixed_grounding_v4_json}",
            )
new_obj365_v4= dict(
                img_path=f"{train_data_root}/Objects365v1/images/train",
                json_file=f"{train_data_root}/{obj365_v4_json}",
            )


new_flickr_v5= dict(
                img_path=f"{train_data_root}/flickr/full_images/",
                json_file=f"{train_data_root}/{flickr_v5_json}",
            )
new_mixed_v5= dict(
                img_path=f"{train_data_root}/mixed_grounding/gqa/images",
                json_file=f"{train_data_root}/{mixed_grounding_v5_json}",
            )
new_obj365_v5= dict(
                img_path=f"{train_data_root}/Objects365v1/images/train",
                json_file=f"{train_data_root}/{obj365_v5_json}",
            )


ye_v4= dict(img_path=f"{train_data_root}/yolo-enterprise/images/train",
                json_file=f"{train_data_root}/{ye_v4_json}",) 


ye_v5= dict(img_path=f"{train_data_root}/yolo-enterprise/images/train",
                json_file=f"{train_data_root}/{ye_v5_json}",) # todo 



# ================= Version A — old-engine data (made by Louis) =================
# DATA_CONFIG["old_engine_data"] = dict(
#     train=dict(grounding_data=[ old_flickr_data, old_mixed_data, old_obj365_data],),
#     val=dict(yolo_data=[lvis_data]),
# )


DATA_CONFIG["old_engine_data"] = dict(
    train=dict(
        grounding_data=[
            dict(
                img_path=f"{train_data_root}/Objects365v1/images/train",
                json_file=f"{train_data_root}/Objects365v1/annotations/objects365_train_segm.engine.cache",
            ) ,
            dict(
                img_path=f"{train_data_root}/flickr/full_images/",
                json_file=f"{train_data_root}/flickr/annotations/final_flickr_separateGT_train_segm.engine.cache"
            ),
            dict(
                img_path=f"{train_data_root}/mixed_grounding/gqa/images",
                json_file=f"{train_data_root}/mixed_grounding/annotations/final_mixed_train_no_coco_segm.engine.cache"
            ),
        ]
    ),
    val=dict(
        yolo_data=[lvis_data]
    )
)





# ========== Version B — enterprise data (Fatih's pipeline, v4 / v5) ==========


DATA_CONFIG["yedata"]=dict(
    train=dict(
        grounding_data=[ new_flickr_v4, new_mixed_v4, new_obj365_v4,ye_v5],
    ),
    val=dict(yolo_data=[lvis_data]),    
    ) # enterprise data v4





DATA_CONFIG["coco128"]=dict(
    train=dict(
        yolo_data=["coco128.yaml"],
    ),
    val=dict(yolo_data=["coco128.yaml"]),
    ) # local smoke-test with coco128


DATA_CONFIG["coco128_seg"]=dict(
    train=dict(
        yolo_data=["coco128-seg.yaml"],
    ),
    val=dict(yolo_data=["coco128-seg.yaml"]),
    ) # local smoke-test for the SEG stage (needs mask labels, not bbox-only coco128)








 
import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



from ultralytics import YOLOE,YOLO
from ultralytics.models.yolo.yoloe import YOLOETrainerFromScratch,YOLOEVPTrainer,YOLOEPEFreeTrainer
from ultralytics.models.yolo.yoloe import YOLOESegTrainerFromScratch, YOLOESegTrainerSegHead
# YOLOESegTrainerSegHead





# model = YOLOE("/root/ultra_louis_work/yoloe/yoloe-v8s-seg-det.pt")

import argparse
parser=argparse.ArgumentParser(description="train yoloe with visual prompt")
parser.add_argument("--model_version", type=str, default="26s")
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--close_mosaic", type=int, default=0)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--project", type=str, default="../runs/train_tp")
#device
parser.add_argument("--device", type=str, default="0,1")
# val
parser.add_argument("--val", type=str, default="True", choices=["True", "False"])
parser.add_argument("--name", type=str, default="yoloe_vp")
# parser.add_argument("--clip_weight_name", type=str, default="mobileclip2:b") 
parser.add_argument("--scale", type=float, default=0.9)

parser.add_argument("--ag", type=str, default="False", choices=["True", "False"]) # all grounding

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
# parser.add_argument("--o2m", type=float, default=0.1)
parser.add_argument("--copy_paste", type=float, default=0.15)
parser.add_argument("--mixup", type=float, default=0.05)
parser.add_argument("--semseg_loss", type=str, default="False", choices=["True", "False"]) # if use segseg_loss 

parser.add_argument("--override", type=str, default=None) 

# freeze all but one2one_cv4
parser.add_argument("--freeze_all_but_one2one_cv4", action="store_true", help="freeze all layers except one2one_cv4")

parser.add_argument("--save_json",type=str,default="True", choices=["True", "False"]) # whether to save results in json format for evaluation



 # mobileclip2b
#
args = parser.parse_args()


if args.freeze_all_but_one2one_cv4:
    assert args.trainer == "YOLOETrainerFromScratch", "freeze_all_but_one2one_cv4 is only supported for YOLOETrainerFromScratch trainer"


# Convert string to bool
args.val = args.val == "True"
args.ag = args.ag == "True"
args.save_json = args.save_json == "True"
assert args.save_json == True, "Currently we only support saving json format results for evaluation. Please set --save_json to True."

# args.semseg_loss = args.semseg_loss == "True"
assert args.trainer in ["YOLOETrainerFromScratch","YOLOEVPTrainer","YOLOEPEFreeTrainer","YOLOESegTrainerFromScratch","YOLOESegTrainerSegHead"], \
    "trainer must be YOLOETrainerFromScratch, YOLOEVPTrainer, YOLOEPEFreeTrainer, YOLOESegTrainerFromScratch, or YOLOESegTrainerSegHead"





assert args.data is not None, "data argument must be provided"
assert args.data in DATA_CONFIG.keys(), f"data {args.data} not found in DATA_CONFIG"
data=DATA_CONFIG[args.data]
 

assert args.trainer in ["YOLOETrainerFromScratch","YOLOEVPTrainer","YOLOEPEFreeTrainer","YOLOESegTrainerFromScratch","YOLOESegTrainerSegHead"], \
    "trainer must be YOLOETrainerFromScratch, YOLOEVPTrainer, YOLOEPEFreeTrainer, YOLOESegTrainerFromScratch, or YOLOESegTrainerSegHead"





###################################################################
model = YOLO("yoloe-{}.yaml".format(args.model_version))
model=model.load(args.weight_path)


if args.trainer == "YOLOETrainerFromScratch" or args.trainer== "YOLOESegTrainerFromScratch":
    print("Using YOLOETrainerFromScratch for training.")
    freeze = []
    if args.freeze_all_but_one2one_cv4: 
        head_index = len(model.model.model) - 1
        freeze = list(range(0, head_index))
        for name, child in model.model.model[-1].named_children():
            if "one2one_cv4" not in name:
                freeze.append(f"{head_index}.{name}")


    refer_data=None
    single_cls=False
elif args.trainer == "YOLOEVPTrainer":
    print("Using YOLOEVPTrainer for training.")
    # savpe is freshly initialized when the model is built from the yaml; the older
    # savpe.init_weights() API was removed from ultralytics, so no manual reinit here.

    # freeze every layer except of the savpe module.
    head_index = len(model.model.model) - 1
    freeze = list(range(0, head_index))
    train_layers=[]

    for name, child in model.model.model[-1].named_children():

        print("name:", name)

        if "savpe"  in name: # unfreeze the whole savpe module
            continue

        elif "cv4" in name: # unfreeze the whole cv4 and one2one_cv4 modules, which is the visual prompt module
            # freeze.extend(
            #     [
            #         f"{head_index}.{name}.0.norm",
            #         f"{head_index}.{name}.1.norm",
            #         f"{head_index}.{name}.2.norm",
            #     ]  )
            continue
        else:
            freeze.append(f"{head_index}.{name}")


    refer_data=refer_data_yaml
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
        ])
    refer_data=None
    single_cls=True


elif args.trainer=="YOLOESegTrainerSegHead":

    print("Using YOLOESegTrainerSegHead for training.")
    # reinit the model.model.savpe.
    # freeze every layer except of the savpe module.
    head_index = len(model.model.model) - 1
    train_layers=[]
    freeze = list(range(0, head_index))
    for name, child in model.model.model[-1].named_children():
        print("name:", name)

        if ("cv5" not in name) and ("one2one_cv5" not in name) and ("proto" not in name):
            freeze.append(f"{head_index}.{name}")
        else:
            train_layers.append(f"{head_index}.{name}")

    refer_data=None
    single_cls=False


else:
    print("trainer_class:", args.trainer)
    raise ValueError("trainer_class must be YOLOETrainerFromScratch, YOLOEVPTrainer, YOLOEPEFreeTrainer, YOLOESegTrainerFromScratch, or YOLOESegTrainerSegHead")

trainer_class =eval( args.trainer)


train_args=dict( data=data,
    batch=args.batch,
    epochs=args.epochs,
    close_mosaic=args.close_mosaic,
    workers=args.workers,
    max_det=args.max_det,
    trainer=trainer_class,  # use YOLOEVPTrainer if converted to detection model
    # clip_weight_name=args.clip_weight_name,
    device=args.device,
    save_period=args.save_period,
    val=args.val,
    project=args.project,
    name=args.name,
    cache=False,
    scale=args.scale, # sensitive.  [0.1,  1.9] 
    copy_paste=args.copy_paste, 
    mixup=args.mixup,
    dfl=6.0,
    # o2m=args.o2m,
    warmup_epochs=1,
    warmup_bias_lr=0.0,
    lr0=args.lr0,
    lrf=args.lrf,
    optimizer=args.optimizer,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    single_cls=single_cls, # for YOLOEPEFreeTrainer
    freeze=freeze, # for YOLOEVPTrainer
    # refer_data=refer_data, # for YOLOEVPTrainer)
    save_json=args.save_json,
    )



if args.override is not None:
    assert args.override in OVERRIDES, f"override {args.override} not found"
    override=OVERRIDES[args.override]
    for k,v in override.items():
        print(f"Overriding {k} from {train_args.get(k, 'N/A')} to {v}")
        train_args[k]=v
            
for k, v in train_args.items():
    print(f"{k}: {v}")

model.train(**train_args)
