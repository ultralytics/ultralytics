from ultralytics import YOLO

#from ultralytics.utils import SETTINGS
#SETTINGS['comet'] = False  # set True to log using Comet.ml

# Only for tune method because it uses all GPUs by default
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datetime import datetime

# Initialize the YOLO model
model = YOLO('yolov8s.yaml', task='detect').load('./../models/yolov8s.pt')

# Tune hyperparameters on COCO8 for 3 epochs with default Tuner
model.tune(
    use_ray=False,
    iterations=20,
    # Fixed training parameters
    device=[0,1,2],
    data='custom_dataset.yaml',
    project=f'grid-search-cdv1/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    fraction=0.5,
    epochs=50,
    patience=10,
    batch=258,
    optimizer='AdamW', # MUST BE FIXED
    plots=False,
    save=False,
    val=False,
)


"""

# Tune hyperparameters on COCO128 for 3 epochs with Ray Tune
model.tune(
    # Ray Tune parameters, default search space
    use_ray=True,
    iterations=2,
    grace_period=10,
    #gpu_per_trial=1,
    # Fixed training parameters
    #device=[0],
    data='coco8.yaml',
    project='grid-search-cdv1',
    fraction=0.8,
    epochs=25,
    batch=4,
    plots=False,
    save=False,
    val=False,
)
"""