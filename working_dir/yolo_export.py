from ultralytics.utils.benchmarks import ProfileModels
from ultralytics import RTDETR

# ProfileModels(paths=["yolo11s.pt"]).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr_p2_l3.yaml")
# model.save("yolo11-rtdetr_p2_l3.pt")
# ProfileModels(paths=["yolo11-rtdetr_p2_l3.pt"]).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr.yaml")
# model.save("yolo11-rtdetr.pt")
# ProfileModels(paths=["yolo11-rtdetr.pt"]).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr_p2_l3_efms.yaml")
# model.save("yolo11-rtdetr_p2_l3_efms.pt")
# ProfileModels(paths=["yolo11-rtdetr_p2_l3_efms.pt"]).run()


model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr_p2_l3_efms.yaml")
model.save("yolo11-rtdetr_p2_l3_efms.pt")
ProfileModels(paths=["yolo11-rtdetr_p2_l3_efms.pt"], imgsz=480).run()
