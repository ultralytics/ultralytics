import comet_ml
import os
from ultralytics import YOLO

# Only for tune method because it uses all GPUs by default
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

# Initialize COMET logger, first log done in notebook where API key was asked, now seems to be saved in .comet.config
comet_ml.init()

# Initialize model and load matching weights
model = YOLO('yolov8s.yaml', task='detect')#.load('./../models/yolov8n.pt')
#model = YOLO('/home-net/ierregue/project/detector/small-fast-detector/training_baselines/8np2-300e-64b2/weights/last.pt', task='detect')

epochs = 300
batch = 64

model.train(
    resume=False,
    data='coco.yaml',
    epochs=epochs,
    batch=batch,
    cache=True,
    save=False,
    device=[1,2,3,7],
    project='training_baselines',
    name=f'8s-{epochs}e-{batch}b',
    verbose=True,
    save_period=50,
    patience=25,

)

"""
# Resume training from 'training/weights/last.pt'
model = YOLO('last.pt')
model.train(resume=True)
"""

"""
# Fine-tune the model
model.tune(
    data='coco.yaml',
    fraction=0.25,
    batch=48,
    epochs=20,
    patience=3,
    gpu_per_trial=1, # number of GPUs per trial, DDP not supported
    max_samples=20,
)
"""