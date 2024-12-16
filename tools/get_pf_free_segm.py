from ultralytics import YOLOE
from copy import deepcopy

model_name = "v8s"
model = YOLOE(f"yoloe-{model_name}-seg.pt")
model.eval()

pf_model = YOLOE(f"yoloe-{model_name}-pf.pt")

names = ["object"]
tpe = model.get_text_pe(names)
model.set_classes(names, tpe)
model.model.model[-1].fuse(model.model.pe)

model.model.model[-1].cv3[0][2] = deepcopy(pf_model.model.model[-1].cv3[0][2]).requires_grad_(True)
model.model.model[-1].cv3[1][2] = deepcopy(pf_model.model.model[-1].cv3[1][2]).requires_grad_(True)
model.model.model[-1].cv3[2][2] = deepcopy(pf_model.model.model[-1].cv3[2][2]).requires_grad_(True)
del model.model.pe

model.save(f"yoloe-{model_name}-seg-pf.pt")

