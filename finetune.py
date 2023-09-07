import comet_ml
from ultralytics import YOLO


# Initialize COMET logger, first log done in notebook where API key was asked, now seems to be saved in .comet.config
comet_ml.init()

# Initialize model and load matching weights
model = YOLO('yolov8n.yaml', task='detect').load('./../models/yolov8n.pt')

epochs = 3
batch = 64

model.train(
    resume=False,
    data='custom_dataset.yaml',
    epochs=epochs,
    batch=batch,
    save=True,
    device=[1,3],
    project='fine-tune-cdv1',
    name=f'8n-{epochs}e-{batch}b',
    verbose=True,
    save_period=1,
    patience=25,

)