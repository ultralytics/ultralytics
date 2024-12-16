from ultralytics import YOLOE
from copy import deepcopy

model_name = "v8s"
model = YOLOE(f"yoloe-{model_name}-seg.yaml")
model.load(f"yoloe-{model_name}-seg.pt")

vp_model = YOLOE(f"yoloe-{model_name}-vp.pt")
model.model.model[-1].savpe = deepcopy(vp_model.model.model[-1].savpe)
model.eval()

model.save(f"yoloe-{model_name}-seg.pt")