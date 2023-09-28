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
batch = 128
lr0 = 0.001
optimizer = 'Adam'

model.train(
    resume=False,
    data='custom_dataset.yaml',
    epochs=epochs,
    batch=batch,
    optimizer=optimizer,
    lr0=lr0,
    save=False,
    device=[0,1,2,3],
    project='fine-tune-cdv1',
    name=f'8s-{epochs}e-{batch}b_{optimizer}_{lr0}lr0_4w',
    verbose=True,
    patience=10,

)