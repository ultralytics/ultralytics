# ============================================================
#  config.py — แก้ค่าตรงนี้ก่อนรัน script อื่น ๆ
# ============================================================

# --- Roboflow ---
RF_API_KEY   = "YOUR_ROBOFLOW_API_KEY"   # API Key จาก Roboflow > Settings
RF_WORKSPACE = "YOUR_WORKSPACE_NAME"     # ชื่อ workspace ใน URL
RF_PROJECT   = "YOUR_PROJECT_NAME"       # ชื่อ project ใน URL
RF_VERSION   = 1                         # เวอร์ชัน dataset

# --- Model ---
MODEL_NAME   = "yolo11n.pt"   # nano=เร็ว | yolo11s/m/l/x=แม่นขึ้น
EPOCHS       = 50
IMGSZ        = 640
BATCH        = 16             # ลดเป็น 8 ถ้า GPU memory ไม่พอ
DEVICE       = 0              # 0 = GPU แรก, "cpu" = ใช้ CPU
RUN_NAME     = "roboflow_assignment"

# --- Inference ---
CONF_THRESH  = 0.25           # confidence threshold
IOU_THRESH   = 0.45           # NMS IoU threshold
