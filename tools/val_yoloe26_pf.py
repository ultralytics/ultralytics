import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator


from pathlib import Path
import re,yaml

def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data



def read_pf_det_from_seg_unfused(model_path,yaml_name,unfused_model_weight,clip_weight_name="mobileclip2:b"):

    # load the most model weights
    det_model = YOLOE(yaml_name).load(model_path)
    det_model.model.args['clip_weight_name']=clip_weight_name
    # set vocab from the unfused model
    unfused_model=YOLOE(unfused_model_weight)
    unfused_model.model.args['clip_weight_name']=clip_weight_name
    unfused_model.eval()
    unfused_model.cuda()
    unfused_model.args['clip_model_weight']=clip_weight_name
    with open('../buffer/ram_tag_list.txt', 'r') as f:
        names = [x.strip() for x in f.readlines()]
    # categories = yaml_load("ultralytics/cfg/datasets/lvis.yaml")["names"].values()
    # names = [c.split("/")[0] for c in categories]     
    vocab = unfused_model.get_vocab(names)
    


    det_model.eval()
    det_model.cuda()
    det_model.set_vocab(vocab, names=names, set_open_ended_te=True)
    det_model.model.model[-1].is_fused = True
    det_model.model.model[-1].conf = 0.001
    det_model.model.model[-1].max_det = 1000
    
    return det_model




def read_pf_det_from_seg_fused(model_path,yaml_name):
    """
        read pd_det from a fused seg model
    """

    # load the most model weights
    det_model = YOLOE(yaml_name).load(model_path)
    det_model.model.args['clip_weight_name']="mobileclip:blt"
    det_model.eval()
    det_model.cuda() 
    import copy
    seg_model=YOLOE(model_path)
    seg_model.model.args['clip_weight_name']="mobileclip:blt"
    seg_model.eval()
    seg_model.cuda()
    # copy the lrpc model ()
    det_model.model.model[-1].lrpc =copy.deepcopy(seg_model.model.model[-1].lrpc)
    det_model.model.model[-1].is_fused = True
    det_model.model.model[-1].conf = 0.001
    det_model.model.model[-1].max_det = 1000

    # del the last layer of loc and cls head (which is copied from the set_vocab function)
    import torch.nn as nn
    for loc_head, cls_head in zip(det_model.model.model[-1].cv2, det_model.model.model[-1].cv3):
        assert isinstance(loc_head, nn.Sequential)
        assert isinstance(cls_head, nn.Sequential)
        del loc_head[-1]
        del cls_head[-1]

    with open('../buffer/ram_tag_list.txt', 'r') as f:
        names = [x.strip() for x in f.readlines()]
    tpe = det_model.model.get_text_pe(names)
    det_model.model.set_classes(names, tpe)

    return det_model






# version='26s'
# weight_path_tp="./runs/yoloe26s_tp_ultra6/mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf0.5_bn_exp/weights/best.pt"
# model_weight="/home/louis/ultra_louis_work/ultralytics/runs/yoloe26s_pf_ultra6/mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf_bn_o2m0.1_pfA/weights/best.pt"


# # set the open_ended_te for YOLOEDetect
# with open('../buffer/ram_tag_list.txt', 'r') as f:
#     names = [x.strip() for x in f.readlines()]
#     model.model.set_open_ended_te(names)
# metrics = model.val(data="lvis.yaml",split="minival", single_cls=single_cls ,max_det=1000,save_json= (not single_cls)) # map






# if single_cls:
#     model=YOLOE(model_weight)
#     head=model.model.model[-1]
#     head.set_fixed_nc(1)  # stop the dynamic update of YOLOEDetect.nc

#     metrics = model.val(data="lvis.yaml",split="minival", single_cls=single_cls ,max_det=1000,save_json= False) # map 0

# else:
#     # model_weight="runs/yoloe26s_pf_ultra6/mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf0.5_bn_o2m0.1_pf2/weights/best.pt"
#     model=read_pf_det_from_seg_unfused(model_weight,f"yoloe-{version}.yaml",weight_path_tp)
#     metrics = model.val(data="lvis.yaml",split="minival", single_cls=single_cls ,max_det=1000,save_json= (not single_cls)) # map 0


# version='11s'
# model_weight="yoloe-11s-seg-pf.pt"
# model=read_pf_det_from_seg_fused(model_weight,f"yoloe-{version}.yaml")


# metrics = model.val(data="lvis.yaml",split="minival", single_cls=False ,max_det=1000,save_json=True, plots=False,
#                     project="runs/prompt_free_test",name="yoloe-11s-seg-pf") # map


version='v8l'
model_weight=f"/home/louis/repos/yoloe/pretrain/yoloe-{version}-seg-pf.pt"
weight_path_tp=f"/home/louis/repos/yoloe/pretrain/yoloe-{version}-seg.pt"
model=read_pf_det_from_seg_unfused(model_weight,f"yoloe-{version}.yaml",weight_path_tp, clip_weight_name="mobileclip:blt")

# del model.model.model[-1].one2one_cv2
# del model.model.model[-1].one2one_cv3
# del model.model.model[-1].one2one_cv4
model.model.end2end = False


metrics = model.val(data="lvis.yaml",split="minival",batch=1, single_cls=False ,max_det=1000,save_json=True, plots=False,
                    project="runs/prompt_free_test",name=f"yoloe-{version}-seg-pf") # map
