"""
2_train.py
----------
เทรน YOLO model บน dataset ที่ดาวน์โหลดมาจาก Roboflow
ต้องรัน 1_download_dataset.py ก่อน

Usage:
    python 2_train.py
"""

import os
from ultralytics import YOLO
from config import MODEL_NAME, EPOCHS, IMGSZ, BATCH, DEVICE, RUN_NAME

# อ่าน data.yaml path ที่บันทึกไว้
if not os.path.exists("dataset_path.txt"):
    raise FileNotFoundError("ไม่พบ dataset_path.txt — รัน 1_download_dataset.py ก่อน")

with open("dataset_path.txt") as f:
    data_yaml = f.read().strip()

print(f"[INFO] Data YAML : {data_yaml}")
print(f"[INFO] Model     : {MODEL_NAME}")
print(f"[INFO] Epochs    : {EPOCHS}")
print(f"[INFO] Image size: {IMGSZ}")
print(f"[INFO] Batch     : {BATCH}")
print(f"[INFO] Device    : {DEVICE}")
print("-" * 40)

model   = YOLO(MODEL_NAME)
results = model.train(
    data=data_yaml,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    name=RUN_NAME,
    patience=10,
    plots=True,
)

best_weights = os.path.join(str(results.save_dir), "weights", "best.pt")

# บันทึก path ของ best model ไว้ให้ script อื่นอ่าน
with open("best_model_path.txt", "w") as f:
    f.write(best_weights)

print("\n[OK] Training complete!")
print(f"[OK] Results saved to : {results.save_dir}")
print(f"[OK] Best weights     : {best_weights}")
print("[OK] Path saved to best_model_path.txt")
