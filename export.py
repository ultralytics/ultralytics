import os
from pathlib import Path

from ultralytics import YOLOE
from ultralytics.utils import yaml_load

model_name = "pretrain/yoloe-v8s-seg.pt"
file_name = "ultralytics/cfg/datasets/lvis.yaml"

model = YOLOE(model_name).cuda()
model.eval()

# Please replace names with yours
data = yaml_load(file_name)
names = [n.split("/")[0] for n in data["names"].values()]

model.set_classes(names, model.get_text_pe(names))

onnx_path = model.export(format="onnx", half=True, opset=13, simplify=True, device="0")
coreml_path = model.export(format="coreml", half=True, nms=False, device="0")

save_name = f"{Path(model_name).stem}"
os.rename(onnx_path, os.path.join(f"{save_name}.onnx"))
os.rename(coreml_path, os.path.join(f"{save_name}.mlpackage"))
