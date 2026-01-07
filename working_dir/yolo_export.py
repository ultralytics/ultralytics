from ultralytics.utils.benchmarks import ProfileModels
from ultralytics import RTDETR

# ProfileModels(paths=["yolo11s.pt"]).run()
model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr_dabpos.yaml")
model.save("yolo11-rtdetr_dabpos.pt")
ProfileModels(paths=["yolo11-rtdetr_dabpos.pt"]).run()
