

import sys,os
sys.path.append("/root/ultra_louis_work/ultralytics")
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # change to the directory of the current script


from ultralytics import YOLOE
from ultralytics.utils.patches import torch_load





#############################################################################
# eval "yoloe-11s-seg.pt" on lvis dataset with visual prompt
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.264
#############################################################################
# model = YOLOE("yoloe-11s.yaml")
# model_path="yoloe-11s-seg.pt"
# state = torch_load(model_path)
# model.load(state["model"])

# metrics = model.val(data="./lvis.yaml", load_vp=True, split='minival',
#                     refer_data="./lvis_train_vps.yaml",max_det=1000)



#############################################################################
# eval best.pt on lvis dataset with visual prompt
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.264
#############################################################################

# model = YOLOE("yoloe-11s.yaml").load("yoloe-11s-seg.pt")
# model_path="yoloe-11s-seg.pt"
# state = torch_load(model_path)
# model.load(state["model"])


model = YOLOE("/root/ultra_louis_work/ultralytics/runs/detect/train17/weights/last.pt")


# model=YOLOE("/root/ultra_louis_work/ultralytics/runs/detect/train/weights/best.pt") 

# If you need to update savpe from another model:
# pretrain_model = YOLOE("yoloe-11s-seg.pt")

# from copy import deepcopy

# model.model.model[-1].savpe = deepcopy(best.model.model[-1].savpe)
# model.eval()



# Conduct model validation on the COCO128-seg example dataset
# metrics = model.val(data="lvis.yaml", load_vp=True)
metrics = model.val(data="./lvis.yaml", load_vp=True, split='minival',save_json=True,
                    refer_data="./lvis_train_vps.yaml",max_det=1000)






# model_path="yoloe-11s-seg.pt"
# from ultralytics.utils.patches import torch_load
# model = YOLOE("yoloe-11s.yaml")
# state = torch_load(model_path)
# model.load(state["model"])
# metrics = model.val(data="./lvis.yaml", load_vp=True, split='minival',save_json=True,
#                     refer_data="./lvis_train_vps.yaml",max_det=1000)
