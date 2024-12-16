from ultralytics import YOLOE
import torch

det_model = YOLOE("yoloe-v8l.yaml")

state = torch.load("yoloe-v8l-seg.pt")

det_model.load(state["model"])
det_model.save("yoloe-v8l-seg-det.pt")

model = YOLOE("yoloe-v8l-seg-det.pt")
print(model.args)
