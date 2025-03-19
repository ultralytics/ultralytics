from ultralytics import YOLOE

model = YOLOE("yoloe-v8l.yaml")
model.load("pretrain/yoloe-v8l-seg.pt")
model.eval()

filename = "ultralytics/cfg/datasets/lvis.yaml"

model.val(data=filename, batch=1, split="minival", rect=False, load_vp=True)

# Fixed AP
# model.val(data=filename, batch=1, split='minival', rect=False, load_vp=True, max_det=1000)
# python tools/eval_fixed_ap.py ../datasets/lvis/annotations/lvis_v1_minival.json runs/detect/val2/predictions.json
