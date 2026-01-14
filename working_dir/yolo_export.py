from ultralytics.utils.benchmarks import ProfileModels
from ultralytics import RTDETR

# ProfileModels(paths=["yolo11n.pt"]).run()
# ProfileModels(paths=["yolo11s.pt"]).run()
# ProfileModels(paths=["yolo11l.pt"]).run()
# ProfileModels(paths=["yolo11x.pt"]).run()
# ProfileModels(paths=["yolo11l-rtdetr_p4_l6_efms_800_pretrained.pt"], imgsz=800).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr.yaml")
# model.save("yolo11-rtdetr.pt")
# ProfileModels(paths=["yolo11-rtdetr.pt"]).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11l-rtdetr.yaml")
# model.save("yolo11l-rtdetr.pt")
# ProfileModels(paths=["yolo11l-rtdetr.pt"]).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr_loss.yaml")
# model.save("yolo11-rtdetr_cuda.pt")
# ProfileModels(paths=["yolo11-rtdetr_cuda.pt"]).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr_p4_l6_efms.yaml")
# model.save("yolo11-rtdetr_p4_l6_efms.pt")
# ProfileModels(paths=["yolo11-rtdetr_p4_l6_efms.pt"]).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr_p2_l3_efms.yaml")
# model.save("yolo11-rtdetr_p2_l3_efms.pt")
# ProfileModels(paths=["yolo11-rtdetr_p2_l3_efms.pt"]).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr.yaml")
# model.save("yolo11-rtdetr.pt")
# ProfileModels(paths=["yolo11-rtdetr.pt"]).run()

model = RTDETR("ultralytics/cfg/models/11/yolo11n-rtdetr_p4_l3.yaml")
model.save("yolo11n-rtdetr_p4_l3_800.pt")
ProfileModels(paths=["yolo11n-rtdetr_p4_l3_800.pt"], imgsz=800).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr_p2_l3_efms.yaml")
# model.save("yolo11-rtdetr_p2_l3_efms_480.pt")
# ProfileModels(paths=["yolo11-rtdetr_p2_l3_efms_480.pt"], imgsz=480).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11-rtdetr_p2_l3_efms.yaml")
# model.save("yolo11-rtdetr_p2_l3_efms_800.pt")
# ProfileModels(paths=["yolo11-rtdetr_p2_l3_efms_800.pt"], imgsz=800).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11l-rtdetr_p2_l3_efms.yaml")
# model.save("yolo11l-rtdetr_p2_l3_efms_800.pt")
# ProfileModels(paths=["yolo11l-rtdetr_p2_l3_efms_800.pt"], imgsz=800).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11n-rtdetr_p4_l3_efms.yaml")
# model.save("yolo11n-rtdetr_p4_l3_efms_800.pt")
# ProfileModels(paths=["yolo11n-rtdetr_p4_l3_efms_800.pt"], imgsz=800).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11l-rtdetr_p4_l3_efms.yaml")
# model.save("yolo11l-rtdetr_p4_l3_efms_800.pt")
# ProfileModels(paths=["yolo11l-rtdetr_p4_l3_efms_800.pt"], imgsz=800).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11l-rtdetr_p4_l6_efms.yaml")
# model.save("yolo11l-rtdetr_p4_l6_efms_800.pt")
# ProfileModels(paths=["yolo11l-rtdetr_p4_l6_efms_800.pt"], imgsz=800).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11l-rtdetr_p4_l6_efms_cuda.yaml")
# model.save("yolo11l-rtdetr_p4_l6_efms_800_cuda.pt")
# ProfileModels(paths=["yolo11l-rtdetr_p4_l6_efms_800_cuda.pt"], imgsz=800).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11m-rtdetr_p4_l3_efms.yaml")
# model.save("yolo11m-rtdetr_p4_l3_efms_800.pt")
# ProfileModels(paths=["yolo11m-rtdetr_p4_l3_efms_800.pt"], imgsz=800).run()

# model = RTDETR("ultralytics/cfg/models/11/yolo11m-rtdetr_p4_l6_efms.yaml")
# model.save("yolo11m-rtdetr_p4_l6_efms_800.pt")
# ProfileModels(paths=["yolo11m-rtdetr_p4_l6_efms_800.pt"], imgsz=800).run()