import comet_ml
from ultralytics import YOLO

# Initialize COMET logger, first log done in notebook where API key was asked, now seems to be saved in .comet.config
comet_ml.init()

# Initialize model and load matching weights
model = YOLO('./evaluation_tools/models/8s.pt')

metrics = model.val(
    data='custom_dataset.yaml',
    imgsz=640,
    batch=16,
    device=[3],
    verbose=True,
    save=True,
    save_json=True,
    plots=True,
    # save results to project/name relative to script directory or absolute path
    project='validations',
    name='8s-sd',
)