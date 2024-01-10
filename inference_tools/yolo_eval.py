from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

model = YOLO('/Users/johnny/Projects//models/detector_best.pt', task='detect')
metrics = model.val(data='/Users/johnny/Projects/datasets/Client_Validation_Set/data.yaml', save_json=True,
                    plots=True, device='cpu')

metrics.box.maps

"""# Load ground truth
cocoGt = COCO('/Users/johnny/Projects/datasets/Client_Validation_Set/annotations/instances_val2017.json')

# Load detection results
cocoDt = cocoGt.loadRes('/Users/johnny/Projects/small-fast-detector/runs/detect/val18/predictions.json')

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

# Redefine the area ranges
cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 16 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
cocoEval.params.areaRngLbl = ['all', 'tiny', 'small', 'medium', 'large']

# Run evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()"""