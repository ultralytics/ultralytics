from ultralytics import YOLO

PRETRAINED=False
SINGLE_CLS=True
DATA="/home/proios/ultralytics/ultralytics/datasets/polyperception.yaml"
MODEL="yolov8m"
EPOCHS=40
PATIENCE=10
DEFAULT_BATCH=16
BATCH=8
LR=0.008 * (BATCH / DEFAULT_BATCH)
DECAY=0.0005
IMG_SIZE=640
SMOOTHING=0.0
CLS=0.5
MIXUP=0.2
NAME="single_cls_m"
DEVICE=1

# Load a model
if PRETRAINED:
    model = YOLO(f'{MODEL}.yaml').load(F'{MODEL}.pt')  # build from YAML and transfer weights
else:
    model = YOLO(f'{MODEL}.yaml')

# Train the model
model.train(data=DATA,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH,
            imgsz=IMG_SIZE,
            device=DEVICE,
            workers=8,
            project=None,
            name=NAME,
            pretrained=PRETRAINED,
            single_cls=SINGLE_CLS,
            cos_lr=False,
            lr0=LR,
            lrf=0.005,
            momentum=0.937,
            weight_decay=DECAY,
            warmup_epochs=8.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=5.0,
            cls=CLS,
            dfl=1.5,
            label_smoothing=SMOOTHING,
            hsv_h=0.01,
            hsv_s=0.05,
            hsv_v=0.05,
            mixup=MIXUP)