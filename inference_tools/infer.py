import json
import time

from inference_utils import inference_time_csv_writer
from ultralytics import YOLO
# For more info visit: https://docs.ultralytics.com/modes/predict/


# Load config.json
with open("./inference_config.json", "r") as f:
    config = json.load(f)
print("Loaded config: ", config)

# Source
source = config['input_data_dir']

# Output directory
root = config['output_dir']
experiment_name = time.strftime("%Y%m%d-%H%M%S")

# Load model
model = YOLO(config['model_path'], task='detect')

# Inference
results = model(
    source,
    conf=0.25,                  # confidence threshold
    iou=0.7,                    # NMS IoU threshold
    imgsz=config['img_size'],   # inference size (pixels)
    half=False,                 # use FP16 half-precision inference
    device=config['device'],    # device to use for inference
    save=True,                  # Images
    save_txt=True,              # Text files
    save_conf=True,             # Save confidences
    # save results to project/name relative to script directory or absolute path
    project=root,
    name=experiment_name,
)

# Save inference time to csv
inference_time_csv_writer(results, root, model.predictor.save_dir.name)
