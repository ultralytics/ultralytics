from collections import OrderedDict
from ultralytics import YOLO

model = YOLO("yolo26m-objv1-150.pt")
sd = model.model.state_dict()

new_sd = OrderedDict()

new_layer_idxs = sorted([23, 24, 25, 26, 27])
new_layer_idxs = [idx - i for i, idx in enumerate(new_layer_idxs)]

layer_map = {}
offset = 0
for i in range(len(model.model.model)):
    while new_layer_idxs and i >= new_layer_idxs[0]:  # <-- while, not if
        offset += 1
        new_layer_idxs.pop(0)
    layer_map[i] = i + offset

skip_head = True
for k, v in sd.items():
    ln = int(k.split(".")[1])
    if skip_head and ln == len(model.model.model) - 1:
        continue
    k = k.replace(f"model.{ln}", f"model.{layer_map[ln]}")
    new_sd[k] = v

model = YOLO("yolo26m-p3-fusion.yaml")
model.model.load_state_dict(new_sd, strict=False)
model.save("yolo26m-p3-fusion.pt")