import time
import pandas as pd

from ultralytics import YOLO
from pycocotools.coco import COCO
from cocoeval import COCOeval

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

# Load ground truth
print("🔌...LOADING GROUND TRUTH...")
cocoGt = COCO(dataset_root+'annotations/instances_val2017.json')

print("🔌...LOADING PREDICTIONS IN RUNS...")
cocoDt = cocoGt.loadRes(outputs_root + '/' + experiment_name + '/predictions.json')

print("🔌...LOADING EVALUATOR...")
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

print("🔌...RESHAPING EVALUATOR FOR CLIENT...")
cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2],
                           [0 ** 2, 16 ** 2],
                           [16 ** 2, 32 ** 2],
                           [32 ** 2, 96 ** 2],
                           [96 ** 2, 1e5 ** 2]]
cocoEval.params.areaRngLbl = ['all', 'tiny', 'small', 'medium', 'large']

print("✅...RESULTS...")

# Run evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# Function to append metrics
def append_metrics(metrics, metric_type, iou, area, max_dets, value):
    metrics.append({
        'Metric Type': metric_type,
        'IoU': iou,
        'Area': area,
        'Max Detections': max_dets,
        'Value': value
    })

# Initialize a list to store the metrics
metrics_ = []

# Extract metrics for bbox/segm evaluation
iou_types = ['0.50:0.95', '0.50', '0.75']
areas = ['all', 'tiny', 'small', 'medium', 'large']
max_dets = [1, 10, 100]

# Extract AP metrics (indices 0-14: 3 IoUs * 5 areas)
for i, iou in enumerate(iou_types):
    for j, area in enumerate(areas):
        idx = i * len(areas) + j
        append_metrics(metrics_, 'AP', iou, area, 100, cocoEval.stats[idx])

# Extract AR metrics (indices 15-17: 3 maxDets for 'all' area)
num_ap_metrics = len(iou_types) * len(areas)  # Total number of AP metrics

# Iterate over max_dets to append AR metrics
for i, md in enumerate(max_dets):
    for j, area in enumerate(areas):
        idx = num_ap_metrics + j + i * len(areas)  # Adjust index calculation for AR
        append_metrics(metrics_, 'AR', '0.50:0.95', area, md, cocoEval.stats[idx])

# Convert to DataFrame
df_metrics = pd.DataFrame(metrics_)

# Save to file (e.g., CSV)
df_metrics.to_csv(outputs_root + "/" + experiment_name + "/eval_metrics.csv", index=False)

print("✅...RESULTS SAVED...")
