from ultralytics import YOLO
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

model = YOLO("ultralytics/cfg/models/ext/cad_yolo12.yaml")
print("Model built successfully")

x = torch.randn(1, 3, 640, 640)
y = model.model(x)
print([t.shape for t in y])

model = YOLO("/home/ralampay/workspace/ultralytics/custom_yolo12.yaml")
print(model.model)
