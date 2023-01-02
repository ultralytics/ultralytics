import os

from ultralytics.yolo.utils import ROOT


def test_checks():
    os.system('yolo mode=checks')


# Train checks ---------------------------------------------------------------------------------------------------------
def test_train_detect():
    os.system('yolo mode=train task=detect model=yolov8n.yaml data=coco128.yaml imgsz=32 epochs=1')


def test_train_segment():
    os.system('yolo mode=train task=segment model=yolov8n-seg.yaml data=coco128-seg.yaml imgsz=32 epochs=1')


def test_train_classify():
    pass


# Val checks -----------------------------------------------------------------------------------------------------------
def test_val_detect():
    os.system('yolo mode=val task=detect model=yolov8n.pt data=coco128.yaml imgsz=32 epochs=1')


def test_val_segment():
    pass


def test_val_classify():
    pass


# Predict checks -------------------------------------------------------------------------------------------------------
def test_predict_detect():
    os.system(f"yolo mode=predict model=yolov8n.pt source={ROOT / 'assets'}")


def test_predict_segment():
    pass


def test_predict_classify():
    pass


# Export checks --------------------------------------------------------------------------------------------------------
def test_export_detect_torchscript():
    os.system('yolo mode=export model=yolov8n.pt format=torchscript')


def test_export_segment_torchscript():
    pass


def test_export_classify_torchscript():
    pass
