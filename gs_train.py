from ultralytics import YOLO

#from ultralytics.utils import SETTINGS
#SETTINGS['comet'] = False  # set True to log using Comet.ml

# Only for tune method because it uses all GPUs by default
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datetime import datetime

# Initialize the YOLO model
model = YOLO('yolov8n.pt')

# Tune hyperparameters on COCO8 for 3 epochs with default Tuner
model.tune(
    use_ray=False,
    iterations=2,
    # Fixed training parameters
    device=[1],
    data='coco128.yaml',
    project=f'grid-search-cdv1/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    fraction=0.5,
    epochs=3,
    batch=4,
    optimizer='SGD', # MUST BE FIXED, try also Adam
    cos_lr=False, # MUST BE FIXED, try also True
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