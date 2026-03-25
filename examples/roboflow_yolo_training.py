"""
========================================================
  AI Assignment: Train YOLO with Roboflow Dataset
========================================================

Steps:
  1. Install dependencies
  2. Download dataset from Roboflow
  3. Train YOLO model
  4. Report Performance Metrics
  5. Run Inference (example code)

Requirements:
  pip install ultralytics roboflow
"""

# ============================================================
# STEP 1: Install & Import
# ============================================================
# Run in terminal:
#   pip install ultralytics roboflow

from ultralytics import YOLO
from roboflow import Roboflow
import os

# ============================================================
# STEP 2: Download Dataset from Roboflow
# ============================================================
# Fill in your Roboflow API key and project details

RF_API_KEY    = "YOUR_ROBOFLOW_API_KEY"   # <-- แก้ตรงนี้
RF_WORKSPACE  = "YOUR_WORKSPACE_NAME"     # <-- แก้ตรงนี้
RF_PROJECT    = "YOUR_PROJECT_NAME"       # <-- แก้ตรงนี้
RF_VERSION    = 1                         # <-- เวอร์ชัน dataset ที่ต้องการ

rf = Roboflow(api_key=RF_API_KEY)
project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
version = project.version(RF_VERSION)
dataset = version.download("yolov8")     # download ในรูปแบบ YOLOv8

DATA_YAML = os.path.join(dataset.location, "data.yaml")
print(f"Dataset downloaded to: {dataset.location}")
print(f"Data YAML: {DATA_YAML}")


# ============================================================
# STEP 3: Train YOLO Model
# ============================================================
# เลือก model ขนาดที่ต้องการ:
#   yolo11n.pt  (nano  - เร็วที่สุด, แม่นน้อยสุด)
#   yolo11s.pt  (small)
#   yolo11m.pt  (medium)
#   yolo11l.pt  (large)
#   yolo11x.pt  (extra-large - ช้าที่สุด, แม่นที่สุด)

model = YOLO("yolo11n.pt")  # โหลด pretrained model

results = model.train(
    data=DATA_YAML,
    epochs=50,          # จำนวนรอบการเทรน (ปรับได้ 50-300)
    imgsz=640,          # ขนาดรูป input
    batch=16,           # batch size (ลดลงถ้า GPU memory ไม่พอ)
    name="roboflow_assignment",   # ชื่อ run
    patience=10,        # หยุดเร็วถ้าไม่ดีขึ้นใน 10 epoch
    device=0,           # 0 = GPU, 'cpu' = CPU
    plots=True,         # สร้างกราฟ metrics
)

print("Training complete!")
print(f"Results saved to: {results.save_dir}")


# ============================================================
# STEP 4: Report Performance Metrics
# ============================================================
# หลังเทรนเสร็จ รัน validation บน validation set

metrics = model.val()

print("\n========== PERFORMANCE METRICS ==========")
print(f"mAP50        : {metrics.box.map50:.4f}")   # mAP @ IoU=0.50
print(f"mAP50-95     : {metrics.box.map:.4f}")     # mAP @ IoU=0.50:0.95
print(f"Precision    : {metrics.box.mp:.4f}")
print(f"Recall       : {metrics.box.mr:.4f}")
print("=========================================")

# ดู per-class metrics
print("\n--- Per-Class AP50 ---")
for i, cls_name in enumerate(metrics.names.values()):
    ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
    print(f"  {cls_name:20s}: {ap:.4f}")


# ============================================================
# STEP 5: Inference Example
# ============================================================
# โหลด best model ที่เทรนได้

best_model_path = os.path.join(str(results.save_dir), "weights", "best.pt")
best_model = YOLO(best_model_path)

# --- Inference บนรูปเดียว ---
IMAGE_PATH = "path/to/your/test/image.jpg"   # <-- แก้ตรงนี้

predictions = best_model.predict(
    source=IMAGE_PATH,
    conf=0.25,     # confidence threshold
    iou=0.45,      # NMS IoU threshold
    save=True,     # บันทึกรูปที่มี bounding box
    show=False,    # แสดงผลแบบ popup (ตั้งเป็น True ถ้าต้องการ)
)

# แสดงผล prediction
for result in predictions:
    boxes = result.boxes
    print(f"\nDetected {len(boxes)} object(s):")
    for box in boxes:
        cls_id  = int(box.cls[0])
        cls_name = best_model.names[cls_id]
        conf    = float(box.conf[0])
        xyxy    = box.xyxy[0].tolist()
        print(f"  Class: {cls_name:20s} | Confidence: {conf:.2f} | BBox: {[round(v,1) for v in xyxy]}")

# --- Inference บน folder รูปทั้งหมด ---
# predictions = best_model.predict(source="path/to/folder/", save=True, conf=0.25)

# --- Inference บน video ---
# predictions = best_model.predict(source="path/to/video.mp4", save=True, conf=0.25)

print(f"\nInference results saved to: {predictions[0].save_dir}")
