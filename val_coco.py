from ultralytics import YOLOE

model = YOLOE("pretrain/yoloe-v8s-seg-coco-pe.pt")

filename = "ultralytics/cfg/datasets/coco.yaml"

model.val(data=filename, batch=1)