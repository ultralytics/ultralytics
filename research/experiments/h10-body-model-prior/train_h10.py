"""H10 fine-tune arms: F_clip + 20 epochs, with and without the body-structure prior.

  I_ftctrl   F_clip best.pt fine-tuned 20 epochs, POSE3D_BONE_GAIN=0   <- control (same extra compute)
  J_ftbone1  same, POSE3D_BONE_GAIN=1                                  <- prior
  K_ftbone4  same, POSE3D_BONE_GAIN=4                                  <- strong prior
The gain is read from the environment by Pose3DLoss; launch with it set.
"""
import argparse, sys
sys.path.insert(0, "/home/rick/ultralytics_pose3d")
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("--device", default="0"); ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--fraction", type=float, default=1.0)
a = ap.parse_args()
YOLO("/home/rick/runs/pose3d/F_clip/weights/best.pt").train(
    data="coco-pose3d-refocal.yaml", epochs=a.epochs, batch=64, imgsz=640, device=a.device, workers=8, seed=0,
    deterministic=False, project="/home/rick/runs/pose3d", name=a.arm, exist_ok=True, plots=False,
    fraction=a.fraction)
