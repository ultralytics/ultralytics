"""Short-recipe COCO-seg arm: python train.py <name> <seg_assign> <seed> <device>."""
import sys
from ultralytics import YOLO

name, mode, seed, device = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
YOLO("yolo26n-seg.yaml").train(
    data="/data/rick/segassign/coco-seg.yaml", pretrained="/data/rick/segassign/yolo26n.pt", epochs=100, batch=64,
    imgsz=640, device=device, seed=seed, seg_assign=mode, project="/data/rick/segassign/runs", name=name,
    exist_ok=True, workers=8,
)
