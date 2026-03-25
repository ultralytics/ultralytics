"""
3_evaluate.py
-------------
รายงาน Performance Metrics หลังเทรนเสร็จ
ต้องรัน 2_train.py ก่อน

Usage:
    python 3_evaluate.py
"""

import os
from ultralytics import YOLO

# อ่าน best model path
if not os.path.exists("best_model_path.txt"):
    raise FileNotFoundError("ไม่พบ best_model_path.txt — รัน 2_train.py ก่อน")

with open("best_model_path.txt") as f:
    model_path = f.read().strip()

print(f"[INFO] Loading model: {model_path}")
model   = YOLO(model_path)
metrics = model.val()

print("\n" + "=" * 45)
print("        PERFORMANCE METRICS (Validation Set)")
print("=" * 45)
print(f"  mAP50        : {metrics.box.map50:.4f}")
print(f"  mAP50-95     : {metrics.box.map:.4f}")
print(f"  Precision    : {metrics.box.mp:.4f}")
print(f"  Recall       : {metrics.box.mr:.4f}")
print("=" * 45)

print("\n--- Per-Class AP50 ---")
for i, cls_name in enumerate(metrics.names.values()):
    ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0.0
    bar = "#" * int(ap * 20)
    print(f"  {cls_name:20s}: {ap:.4f}  |{bar:<20}|")

print("\n[OK] Evaluation complete!")
