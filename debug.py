import os
from ultralytics import YOLO
import comet_ml
from ultralytics.utils import SETTINGS

# Only for tune method because it uses all GPUs by default
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

from ultralytics.utils import SETTINGS
SETTINGS['comet'] = True  # set True to log using Comet.ml
comet_ml.init()

# Initialize model and load matching weights
model = YOLO('yolov8s.yaml', task='detect').load('./../models/yolov8s.pt')
results = model.train(
    save=True,
    verbose=True,
    plots=True,
    project='debug',
    name='8s',
    data='coco8.yaml',
    epochs=3,
    batch=4,
    imgsz=320,
)



#epochs = 4
#batch = 258

#metrics = model.val(data='custom_dataset.yaml', verbose=True, plots=True, save_json=True, device=[1])
"""
model.train(
    resume=False,
    data='custom_dataset.yaml',
    epochs=epochs,
    batch=batch,
    save=True,
    lr0=0.001,
    device=[0,1,2],
    project='debug',
    name=f'8s-{epochs}e-{batch}b',
    verbose=True,
    patience=100,
    close_mosaic=20.7,
    cos_lr=0.75

)
"""
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