import comet_ml
from ultralytics import YOLO

# Initialize COMET logger, first log done in notebook where API key was asked, now seems to be saved in .comet.config
comet_ml.init()

# Initialize model and load matching weights
model = YOLO('yolov8n-p2.yaml').load('./../models/yolov8n.pt')

# Train the model
model.train(
    data='coco128.yaml',
    epochs=5,
    batch=64,
    imgsz=640,
    save=True,
    device=[2,3],
    project='training_baselines',
    name='8n2_620',
    verbose=True
)


#x=0