import os
from ultralytics import YOLO

# Only for tune method because it uses all GPUs by default
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

#from ultralytics.utils import SETTINGS
#SETTINGS['comet'] = False  # set True to log using Comet.ml

# Initialize model and load matching weights
model = YOLO('yolov8n.pt')
epochs = 2
batch = 4

#metrics = model.val(data='custom_dataset.yaml', verbose=True, plots=True, save_json=True, device=[1])

model.train(
    resume=False,
    data='coco8.yaml',
    epochs=epochs,
    batch=batch,
    save=True,
    device=[4],
    project='fine-tune-cdv1',
    name=f'8n-{epochs}e-{batch}b',
    verbose=True,
    save_period=1,
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