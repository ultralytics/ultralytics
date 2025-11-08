from ultralytics import YOLO
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

model = YOLO("ultralytics/cfg/models/ext/cad_yolo12.yaml")
print("Model built successfully")

x = torch.randn(1, 3, 640, 640)
y = model.model(x)
if isinstance(y, torch.Tensor):
    print(y.shape)
elif isinstance(y, (list, tuple)):
    print([getattr(t, "shape", type(t).__name__) for t in y])
else:
    print(type(y).__name__)
print(model.model)
