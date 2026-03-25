"""
4_inference.py
--------------
ตัวอย่าง Inference — ทดสอบ model บนรูป / folder / video
ต้องรัน 2_train.py ก่อน

Usage:
    python 4_inference.py --source path/to/image.jpg
    python 4_inference.py --source path/to/folder/
    python 4_inference.py --source path/to/video.mp4
"""

import argparse
import os
from ultralytics import YOLO
from config import CONF_THRESH, IOU_THRESH

# --- รับ argument ---
parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help="Path to image / folder / video")
parser.add_argument("--model",  default=None,  help="Path to .pt file (optional, ใช้ best model ถ้าไม่ระบุ)")
args = parser.parse_args()

# --- โหลด model ---
if args.model:
    model_path = args.model
else:
    if not os.path.exists("best_model_path.txt"):
        raise FileNotFoundError("ไม่พบ best_model_path.txt — รัน 2_train.py ก่อน หรือระบุ --model")
    with open("best_model_path.txt") as f:
        model_path = f.read().strip()

print(f"[INFO] Model  : {model_path}")
print(f"[INFO] Source : {args.source}")
print(f"[INFO] Conf   : {CONF_THRESH}  |  IoU: {IOU_THRESH}")
print("-" * 40)

model       = YOLO(model_path)
predictions = model.predict(
    source=args.source,
    conf=CONF_THRESH,
    iou=IOU_THRESH,
    save=True,    # บันทึกรูปผลลัพธ์
    show=False,
)

# --- แสดงผล ---
total = sum(len(r.boxes) for r in predictions)
print(f"\n[OK] Detected {total} object(s) total\n")

for i, result in enumerate(predictions):
    print(f"  [{i+1}] {os.path.basename(str(result.path))}  —  {len(result.boxes)} object(s)")
    for box in result.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf     = float(box.conf[0])
        xyxy     = [round(v, 1) for v in box.xyxy[0].tolist()]
        print(f"       • {cls_name:20s}  conf={conf:.2f}  bbox={xyxy}")

print(f"\n[OK] Results saved to: {predictions[0].save_dir}")
