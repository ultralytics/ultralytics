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



def read_pf_det_from_seg_unfused(model_path,yaml_name,unfused_model_weight):

    # load the most model weights
    det_model = YOLOE(yaml_name).load(model_path)
    det_model.model.args['clip_weight_name']="mobileclip2:b"
    # set vocab from the unfused model
    unfused_model=YOLOE(unfused_model_weight)
    unfused_model.model.args['clip_weight_name']="mobileclip2:b"
    unfused_model.eval()
    unfused_model.cuda()
    unfused_model.args['clip_model_weight']="mobileclip2:b"
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

    
    names = list(yaml_load("ultralytics/cfg/datasets/lvis.yaml")["names"].values())
    tpe = det_model.model.get_text_pe(names)
    det_model.model.set_classes(names, tpe)
    return det_model



version='26s'
weight_path_tp="./runs/yoloe26s_tp_ultra6/mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf0.5_bn_exp/weights/best.pt"
model_weight="/home/louis/ultra_louis_work/ultralytics/runs/yoloe26s_pf_ultra6/mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf_bn_o2m0.1_pfA/weights/best.pt"


single_cls=False

if single_cls:
    model=YOLOE(model_weight)
    head=model.model.model[-1]
    head.set_fixed_nc(1)  # stop the dynamic update of YOLOEDetect.nc

    metrics = model.val(data="lvis.yaml",split="minival", single_cls=single_cls ,max_det=1000,save_json= False) # map 0

else:
    # model_weight="runs/yoloe26s_pf_ultra6/mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf0.5_bn_o2m0.1_pf2/weights/best.pt"
    model=read_pf_det_from_seg_unfused(model_weight,f"yoloe-{version}.yaml",weight_path_tp)


    metrics = model.val(data="lvis.yaml",split="minival", single_cls=single_cls ,max_det=1000,save_json= (not single_cls)) # map 0
