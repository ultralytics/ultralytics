from ultralytics import YOLO

model = YOLO('/home/johnny/Projects/models/detector_best.pt')
metrics = model.val(data='/home/johnny/Projects/datasets/Client_Validation_Set/data.yaml')

metrics.box.maps