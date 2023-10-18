from inference_utils import inference_time_csv_writer
from ultralytics import YOLO
# For more info visit: https://docs.ultralytics.com/modes/predict/

# Source
source = './data/test'

# Output directory
root = './outputs'
experiment_name = 'experiment_'

# Load model
model = YOLO('./models/custom_best.onnx', task='detect')

# Inference
results = model(
    source,
    conf=0.25,          # confidence threshold
    iou=0.7,            # NMS IoU threshold
    imgsz=640,
    half=False,         # use FP16 half-precision inference
    device='cpu',
    save=True,          # Images
    save_txt=True,      # Text files
    save_conf=True,     # Save confidences
    # save results to project/name relative to script directory or absolute path
    project=root,
    name=experiment_name,
)

# Save inference time to csv
inference_time_csv_writer(results, root, model.predictor.save_dir.name)
