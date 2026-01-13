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



def read_pf_det_from_seg_unfused(model_path,yaml_name,unfused_model_weight,clip_weight_name="mobileclip2:b",end2end=False):


    # set vocab from the unfused model
    unfused_model=YOLOE(unfused_model_weight)
    unfused_model.model.args['clip_weight_name']=clip_weight_name
    unfused_model.eval()
    unfused_model.cuda()

    if not end2end:
        # del unfused_model.model.model[-1].one2one_cv2
        # del unfused_model.model.model[-1].one2one_cv3
        # del unfused_model.model.model[-1].one2one_cv4
        unfused_model.model.end2end = False
        # unfused_model.model.model[-1].end2end = False

    with open('./tools/ram_tag_list.txt', 'r') as f:
        names = [x.strip() for x in f.readlines()]
    # categories = yaml_load("ultralytics/cfg/datasets/lvis.yaml")["names"].values()
    # names = [c.split("/")[0] for c in categories]     
    vocab = unfused_model.get_vocab(names)
    
    # load the most model weights
    det_model = YOLOE(yaml_name).load(model_path)
    # det_model=YOLOE(model_path)
    det_model.model.args['clip_weight_name']=clip_weight_name
    if not end2end:
        # del det_model.model.model[-1].one2one_cv2
        # del det_model.model.model[-1].one2one_cv3
        # del det_model.model.model[-1].one2one_cv4
        det_model.model.end2end = False
        # det_model.model.model[-1].end2end = False


    det_model.eval()
    det_model.cuda()
    det_model.set_vocab(vocab, names=names)
    det_model.model.model[-1].is_fused = True
    det_model.model.model[-1].conf = 0.001
    det_model.model.model[-1].max_det = 1000
    
    return det_model







yoloe26m_tp="runs/yoloe26_tp/26m_ptwobjv1_bs256_epo25_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt"
yoloe26m_pf="runs/yoloe26_pf/26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
default_tp_weight=yoloe26m_tp
default_pf_weight=yoloe26m_pf
default_version="26m"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='cuda device(   s) to use')
parser.add_argument('--tp_weight', type=str, default=default_tp_weight, help='path to text prompt model weight')
parser.add_argument('--pf_weight', type=str, default=default_pf_weight, help='path to visual prompt model weight')
parser.add_argument('--single_cls', type=str, default="False", help='whether to eval as single class')
parser.add_argument('--version', type=str, default=default_version, help='model version')
parser.add_argument('--not_end2end', action='store_true', help='whether to use end2end mode')




args= parser.parse_args()

assert args.single_cls in ["True","False"]
single_cls={"True":True, "False":False}[args.single_cls]

# ['clip_weight_name']="mobileclip2:b"

if single_cls:
    end2end=not args.not_end2end

    model_weight=args.pf_weight
    model=YOLOE(model_weight)
    head=model.model.model[-1]
    if not end2end:
        del model.model.model[-1].one2one_cv2
        del model.model.model[-1].one2one_cv3
        del model.model.model[-1].one2one_cv4
        model.model.end2end = False

    head.set_fixed_nc(1)  # stop the dynamic update of YOLOEDetect.nc

    metrics = model.val(data="lvis.yaml",split="minival", single_cls=single_cls ,max_det=1000,save_json= False) # map 0

else:

    assert args.tp_weight is not None, "Please provide text prompt model weight for unfused det model."
    model_weight=args.pf_weight
    model_weight_tp=args.tp_weight
    version=args.version

    # model_weight="runs/yoloe26s_pf_ultra6/mobileclip2:b_26s_bs128_ptwobject365v1_close2_agdata2_lrf0.5_bn_o2m0.1_pf2/weights/best.pt"
    end2end=not args.not_end2end

    model=read_pf_det_from_seg_unfused(model_weight,f"yoloe-{version}.yaml",model_weight_tp,end2end=end2end)



    import time
    time_stamp=time.strftime("%Y%m%d-%H%M%S")

    # get the time_stamp with ms 

    project="runs/yoloe26_pf_eval"
    run_name="val_"+time_stamp
    metrics = model.val(data="lvis.yaml",split="minival", single_cls=single_cls ,max_det=1000,save_json= (not single_cls),clip_weight_name="mobileclip2:b",project=project,name=run_name) # map 0
    pred_file= f"{project}/{run_name}/predictions.json"

    import os 
    abs_path=os.path.abspath("./tools/eval_open_ended.py")
    lvis_json="/home/louis/ultra_louis_work/datasets/lvis/annotations/lvis_v1_minival.json"
    os.system(f"python {abs_path} --json {lvis_json} --pred {pred_file} --fixed")