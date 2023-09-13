import os
import comet_ml
from ultralytics import YOLO

# Set number of threads
os.environ['OMP_NUM_THREADS'] = '4'

# Initialize COMET logger, first log done in notebook where API key was asked, now seems to be saved in .comet.config
comet_ml.init()

# Initialize model and load matching weights
model = YOLO('yolov8s.yaml', task='detect').load('./../models/yolov8s.pt')

epochs = 100
batch = 256

model.train(
    resume=False,
    data='custom_dataset.yaml',
    epochs=epochs,
    batch=batch,
    save=True,
    device=[3,4,5,6],
    project='fine-tune-cdv1',
    name=f'8s-{epochs}e-{batch}b_4w',
    verbose=True,
    save_period=20,
    patience=25,

)