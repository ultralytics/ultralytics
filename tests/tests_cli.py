import os


def test_checks():
    os.system('yolo mode=checks')


def test_train_detect():
    os.system('yolo mode=train task=detect model=yolov8n.yaml data=coco128.yaml imgsz=32 epochs=1')


def test_train_segment():
    os.system('yolo mode=train task=segment model=yolov8n-seg.yaml data=coco128-seg.yaml imgsz=32 epochs=1')


def test_train_classify():
    pass
