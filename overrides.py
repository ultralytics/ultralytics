






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


refer_data_yaml=os.path.abspath(f"../datasets/lvis_train_vps.yaml")



DATA_CONFIG=dict()

lvis_data=os.path.abspath("../datasets/lvis.yaml")


old_flickr_data= dict(
                img_path="../datasets/flickr/full_images/",
                json_file="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json",
            )
old_mixed_data= dict(
                img_path="../datasets/mixed_grounding/gqa/images",
                json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
            )
old_obj365_data= dict(
                img_path="../datasets/Objects365v1/images/train",
                json_file="../datasets/Objects365v1/annotations/objects365_train_segm.json",
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


yoloep_v4=None # to do 

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


yoloep_v5= dict( img_path=f"{train_data_root}/yolo-enterprise/images/train",
                json_file=f"{train_data_root}/yolo-enterprise/merged/train/original_plus_florence_phrases_train.json",) # todo 



DATA_CONFIG["old_engine_data"] = dict(
    train=dict(grounding_data=[ old_flickr_data, old_mixed_data, old_obj365_data],),
    val=dict(yolo_data=[lvis_data]),
)



DATA_CONFIG["newdatav4"]=dict(
    train=dict(
        grounding_data=[ new_flickr_v4, new_mixed_v4, new_obj365_v4],
    ),
    val=dict(yolo_data=[lvis_data]),
)


DATA_CONFIG["newdatav4_obj365v5"]=dict(
    train=dict(
        grounding_data=[ new_flickr_v4, new_mixed_v4, new_obj365_v5],
    ),
    val=dict(yolo_data=[lvis_data]),
)

DATA_CONFIG["newdatav5"]=dict(
    train=dict(
        grounding_data=[ new_flickr_v5, new_mixed_v5, new_obj365_v5],
    ),
    val=dict(yolo_data=[lvis_data]),    
    )

DATA_CONFIG["epdatav5"]=dict(
    train=dict(
        grounding_data=[ new_flickr_v5, new_mixed_v5, new_obj365_v5,yoloep_v5],
    ),
    val=dict(yolo_data=[lvis_data]),    
    )  # enterprise data v5

DATA_CONFIG["epdatav4"]=dict(
    train=dict(
        grounding_data=[ new_flickr_v4, new_mixed_v4, new_obj365_v4,yoloep_v4],
    ),
    val=dict(yolo_data=[lvis_data]),    
    ) # enterprise data v4



DATA_CONFIG["old_objv1_only"]=dict(
    train=dict(
        grounding_data=[ old_obj365_data],
    ),
    val=dict(yolo_data=[lvis_data]),
)
DATA_CONFIG["new_objv5_only"]=dict(
    train=dict(
        grounding_data=[ new_obj365_v5],
    ),
    val=dict(yolo_data=[lvis_data]),
)


