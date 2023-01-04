import os
from pathlib import Path

from ultralytics.yolo.utils import ROOT, SETTINGS

MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n.pt'
CFG = 'yolov8n.yaml'


def test_checks():
    os.system('yolo mode=checks')


# Train checks ---------------------------------------------------------------------------------------------------------
def test_train_detect():
    os.system(f'yolo mode=train task=detect model={MODEL} data=coco128.yaml imgsz=32 epochs=1')


def test_train_segment():
    os.system('yolo mode=train task=segment model=yolov8n-seg.yaml data=coco128-seg.yaml imgsz=32 epochs=1')


def test_train_classify():
    pass


# Val checks -----------------------------------------------------------------------------------------------------------
def test_val_detect():
    os.system(f'yolo mode=val task=detect model={MODEL} data=coco128.yaml imgsz=32 epochs=1')


def test_val_segment():
    pass


def test_val_classify():
    pass


# Predict checks -------------------------------------------------------------------------------------------------------
def test_predict_detect():
    os.system(f"yolo mode=predict model={MODEL} source={ROOT / 'assets'}")


def test_predict_segment():
    pass


def test_predict_classify():
    pass


# Export checks --------------------------------------------------------------------------------------------------------
def test_export_detect_torchscript():
    os.system(f'yolo mode=export model={MODEL} format=torchscript')


def test_export_segment_torchscript():
    pass


def test_export_classify_torchscript():
    pass
