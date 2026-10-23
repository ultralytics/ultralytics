"""H11 Stage 1 arms, F_clip's recipe exactly; only the initialisation differs."""
import argparse, sys
sys.path.insert(0, "/home/rick/ultralytics_pose3d")
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("--device", default="0")
a = ap.parse_args()
pre = False if a.arm == "S_scratch_full" else f"/home/rick/h11/init_{a.arm}.pt"
YOLO("/home/rick/ultralytics_pose3d/ultralytics/cfg/models/26/yolo26n-pose3d.yaml").train(
    data="coco-pose3d-refocal.yaml", epochs=100, batch=64, imgsz=640, device=a.device, workers=8, seed=0,
    deterministic=False, project="/home/rick/runs/pose3d", name=a.arm, exist_ok=True, plots=True, pretrained=pre)
