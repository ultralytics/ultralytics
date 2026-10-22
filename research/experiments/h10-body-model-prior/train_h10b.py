"""H10 addendum arms: real-GT 3DPW training with and without the body-structure prior (gain from env)."""
import argparse, sys
sys.path.insert(0, "/home/rick/ultralytics_pose3d")
from ultralytics import YOLO

INIT = {"pw": ("/home/rick/ultralytics_pose3d/ultralytics/cfg/models/26/yolo26n-pose3d.yaml", "/home/rick/assets/yolo26n-pose.pt"),
        "pwft": ("/home/rick/runs/pose3d/F_clip/weights/best.pt", None)}
ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("init", choices=INIT); ap.add_argument("--device", default="0")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
model, pre = INIT[a.init]
YOLO(model).train(data="3dpw-pose3d-trainval.yaml", epochs=a.epochs, batch=64, imgsz=640, device=a.device,
                  workers=8, seed=a.seed, deterministic=False, project="/home/rick/runs/pose3d", name=a.arm,
                  exist_ok=True, plots=False, **({"pretrained": pre} if pre else {}))
