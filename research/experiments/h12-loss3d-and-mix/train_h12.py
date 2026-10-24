"""H12 arms. #3: MPJPE-space loss as a F_clip fine-tune. #1: COCO + BEDLAM mixed training, step-matched.

The MPJPE gain is read from the environment by Pose3DLoss (POSE3D_MPJPE_GAIN); launch with it set.
  X_ftctrl     F_clip + 20 ep COCO, gain 0                 <- control for both X arms and for M_mixft10
  X_ftmpjpe1   F_clip + 20 ep COCO, gain 1
  X_ftmpjpe4   F_clip + 20 ep COCO, gain 4
  M_mix50      yolo26n-pose init, 50 ep COCO+BEDLAM (113k imgs/ep) = F_clip's 100 ep x 56.6k steps
  M_mixft10    F_clip + 10 ep COCO+BEDLAM = X_ftctrl's steps
"""
import argparse, sys
sys.path.insert(0, "/home/rick/ultralytics_pose3d")
from ultralytics import YOLO

FCLIP, POSE = "/home/rick/runs/pose3d/F_clip/weights/best.pt", "/home/rick/assets/yolo26n-pose.pt"
YAML = "/home/rick/ultralytics_pose3d/ultralytics/cfg/models/26/yolo26n-pose3d.yaml"
ARMS = {
    "X_ftctrl": (FCLIP, None, "coco-pose3d-refocal.yaml", 20),
    "X_ftmpjpe1": (FCLIP, None, "coco-pose3d-refocal.yaml", 20),
    "X_ftmpjpe4": (FCLIP, None, "coco-pose3d-refocal.yaml", 20),
    "M_mix50": (YAML, POSE, "coco-bedlam-pose3d.yaml", 50),
    "M_mixft10": (FCLIP, None, "coco-bedlam-pose3d.yaml", 10),
}
ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("--device", default="0")
ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--fraction", type=float, default=1.0)
a = ap.parse_args()
model, pre, data, ep = ARMS[a.arm]
YOLO(model).train(data=data, epochs=a.epochs or ep, batch=64, imgsz=640, device=a.device, workers=8, seed=0,
                  deterministic=False, project="/home/rick/runs/pose3d", name=a.arm, exist_ok=True, plots=False,
                  fraction=a.fraction, **({"pretrained": pre} if pre else {}))
