from ultralytics import YOLOE

model = YOLOE("pretrain/yoloe-v8s-seg.pt")

filename = "ultralytics/cfg/datasets/lvis.yaml"

model.val(data=filename, batch=1, split="minival", rect=False)

# Fixed AP
# model.val(data=filename, batch=1, split='minival', rect=False, max_det=1000)
# python tools/eval_fixed_ap.py ../datasets/lvis/annotations/lvis_v1_minival.json runs/detect/val2/predictions.json
