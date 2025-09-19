import sys,os
sys.path.append("/home/louis/ultra_louis_work/ultralytics")
os.chdir("/home/louis/ultra_louis_work/ultralytics")
from ultralytics import YOLOE
from ultralytics.utils.patches import torch_load







def run_val(**kwargs):


    model = YOLOE(kwargs["model_weight"])




    data=kwargs["data"]
    vp_weight=kwargs["vp_weight"]
    refer_data=kwargs["refer_data"]
    batch=kwargs["batch"]
    load_vp=kwargs["load_vp"]
    split=kwargs["split"]
    plots=kwargs["plots"]
    save_json=kwargs["save_json"]
    max_det=kwargs["max_det"]
    

    metrics = model.val(data=data, load_vp=load_vp, split=split, save_json=save_json, vp_weight=vp_weight, batch=batch,
                        refer_data=refer_data, max_det=max_det, plots=plots)

    states=dict(**kwargs)

    if isinstance(metrics, dict):
        for k, v in metrics.items():
            states[k] = v
    else:
        states["metrics"]=metrics
    return states

############################################################################
# get vp_weights from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vp_weight', type=float, default=1, help='weight for visual prompt embeddings when combining with text embeddings')
parser.add_argument('--model_weight', type=str, 
default="/home/louis/ultra_louis_work/ultralytics/runs/detect/train13/weights/best.pt", help='model weight path')
args = parser.parse_args()
vp_weight = args.vp_weight
model_weight=args.model_weight
print(f"vp_weight: {vp_weight}")
############################################################################

kwargs= dict(model_weight=model_weight,
              max_det=1000,
              vp_weight=vp_weight,
              refer_data="./train_vp/lvis_train_vps.yaml",
              batch=64,
              data="./train_vp/lvis.yaml",
              load_vp=True,
              split='minival',
              plots=False,
              save_json=True,

)
############################################################################

states=run_val(**kwargs)
import pandas as pd
df=pd.DataFrame([states])
############################################################################

exp_res_csv="./runs/vp_result/results.csv"
if not os.path.exists("./runs/vp_result"):
    os.makedirs("./runs/vp_result")

# check if file exists, write it ,  append if exists
if not os.path.exists(exp_res_csv):
    df.to_csv(exp_res_csv, index=False)
else:
    df.to_csv(exp_res_csv, mode='a', header=False, index=False)