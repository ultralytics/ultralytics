import time
from ultralytics import YOLO


# SETTING UP PARAMETERS
# Better not to change these parameters
dataset_root = './data/client_test/'
model_path = './models/detector_best.pt'
outputs_root = './outputs'
experiment_name = time.strftime("%Y%m%d-%H%M%S")
# Can be changed
imgsz = 640
batch = 8
device = 'cpu'


#  START OF EVALUATION
print("🚀...WELCOME TO EVALUATION DETECTOR MODEL...")

print("🚀...Initializing model...")
model = YOLO(model_path, task='detect')

print("🚀...INFERENCE MODE...🚀")
print("📦...GETTING PREDICTIONS...📦")
metrics = model.val(
    data=dataset_root+'data.yaml',
    imgsz=imgsz,
    batch=batch,
    device=device,
    save=True,
    save_json=True,
    plots=True,
    save_txt=True,      # Text files
    save_conf=True,     # Save confidences
    # save results to project/name relative to script directory or absolute path
    project=outputs_root,
    name=experiment_name,
)
