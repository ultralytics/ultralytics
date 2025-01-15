# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- Coding: UTF-8 -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# ===============================================================================
# @File   : ultralytics//test_yolo.py
# @IDE    : PyCharm
# @Author : Sun Hao
# @Email  : 2865467769@qq.com
# @Date   : 2025/1/10 17:29
# @Desc   :
# ===============================================================================
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    results = model.train(data="coco8.yaml", epochs=50, imgsz=640, workers=0)

    # model = YOLO("yolo11n-seg.pt")
    # results = model.train(data="coco8-seg.yaml", epochs=50, imgsz=640, workers=0, seed=2025)

    # model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
    # results = model.train(data="coco8-pose.yaml", epochs=50, imgsz=640, workers=0)

    # model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)
    # results = model.train(data="dota8.yaml", epochs=50, imgsz=640, workers=0)

    # model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
    # results = model.train(data="mnist160", epochs=50, imgsz=64, workers=0)
