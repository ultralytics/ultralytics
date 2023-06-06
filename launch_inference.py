from ultralytics import YOLO
from pathlib import Path

NAME="dummy"
MODEL_PATH = Path("/home/proios/ultralytics/runs/detect/single_cls_n/weights/best.pt")
DATA_SOURCE = Path("/home/proios/data-lake/datav2/datasets/yolo_database/images/test")
CONF=0.25
IOU=0.7
HALF=True
DEVICE=0
SHOW = False
SAVE_IMGS=True
SAVE_TXT=False
SAVE_CONF=False
SAVE_CROP=False
SHOW_LABELS=False
MAX_DET=300
AGNOSTIC_NMS=True
SHOW_BOXES=True

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Can't find the model {MODEL_PATH}")

if not DATA_SOURCE.exists():
    raise FileNotFoundError(f"Can't find the source {DATA_SOURCE}")

model = YOLO(str(MODEL_PATH))
results = model.predict(source=DATA_SOURCE,
                        conf=CONF,
                        iou=IOU,
                        half=HALF,
                        device=DEVICE,
                        show=SHOW,
                        save=SAVE_IMGS,
                        save_txt=SAVE_TXT,
                        save_conf=SAVE_CONF,
                        save_crop=SAVE_CROP,
                        show_labels=SHOW_LABELS,
                        max_det=MAX_DET,
                        agnostic_nms=AGNOSTIC_NMS,
                        boxes=SHOW_BOXES,
                        name=NAME)


