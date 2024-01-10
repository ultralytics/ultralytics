from ultralytics import YOLO

model = YOLO('/Users/johnny/Projects//models/detector_best.pt')
metrics = model.val(data='/Users/johnny/Projects/datasets/Client_Validation_Set/data.yaml', save_json=True,
                    plots=True, device='cpu')

metrics.box.maps