import json
import os
import glob

import numpy as np
import pandas as pd

from ultralytics import YOLO
from pycocotools.coco import COCO
from cocoeval import COCOeval


print("🚀...WELCOME TO EVALUATION DETECTOR MODEL...")

print("🚀...Initializing model...")
model = YOLO('../inference_tools/Evaluation/models/detector_best.pt', task='detect')

print("🚀...INFERENCE MODE...🚀")
print("📦...GETTING PREDICTIONS...📦")
metrics = model.val(data='../inference_tools/Evaluation/datasets/Client_Validation_Set/data.yaml', save_json=True, plots=True)

metrics.box.maps

# Load ground truth
print("🔌...LOADING GROUND TRUTH...")
cocoGt = COCO('../inference_tools/Evaluation/datasets/Client_Validation_Set/annotations/instances_val2017.json')

print("🔌...LOADING PREDICTIONS IN RUNS...")
list_of_dirs = glob.glob('../runs/detect/val') + glob.glob('../runs/detect/val[0-9]*')
latest_dir = max(list_of_dirs, key=os.path.getmtime)

json_files = glob.glob(os.path.join(latest_dir, '*.json'))
if json_files:
    detection_results_file = max(json_files, key=os.path.getmtime)  # Get the latest json file
else:
    raise FileNotFoundError("❌...No JSON detection results file found in the latest 'val' directory...❌")


print("🔌...LOADING EVALUATOR...")
cocoDt = cocoGt.loadRes(detection_results_file)

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

print("🔌...RESHAPING EVALUATOR FOR CLIENT...")
cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 16 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
cocoEval.params.areaRngLbl = ['all', 'tiny', 'small', 'medium', 'large']

print("✅...RESULTS...")

# Run evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

results_file = "../inference_tools/Evaluation/results/"
if not os.path.exists(results_file):
    os.makedirs(results_file)

results_file = os.path.join(results_file, "results.json")

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)


with open(results_file, "w") as file:
    json.dump(cocoEval.stats, file, default=convert_numpy)

print("✅...RESULTS SAVED...")

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
for i, md in enumerate(max_dets):
    idx = 15 + i  # Starting index for AR metrics is 15
    append_metrics(metrics_, 'AR', '0.50:0.95', 'all', md, cocoEval.stats[idx])


# Convert to DataFrame
df_metrics = pd.DataFrame(metrics_)

# Save to file (e.g., CSV)
df_metrics.to_csv("../inference_tools/Evaluation/results/eval_metrics.csv", index=False)