# Ultralytics YOLO 🚀, GPL-3.0 license

import os
from pathlib import Path

from ultralytics.yolo.utils import ROOT, SETTINGS

MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n'
CFG = 'yolov8n'


def test_checks():
    os.system('yolo mode=checks')


# Train checks ---------------------------------------------------------------------------------------------------------
def test_train_det():
    os.system(f'yolo mode=train task=detect model={CFG}.yaml data=coco8.yaml imgsz=32 epochs=1')


def test_train_seg():
    os.system(f'yolo mode=train task=segment model={CFG}-seg.yaml data=coco8-seg.yaml imgsz=32 epochs=1')


def test_train_cls():
    os.system(f'yolo mode=train task=classify model={CFG}-cls.yaml data=mnist160 imgsz=32 epochs=1')


# Val checks -----------------------------------------------------------------------------------------------------------
def test_val_detect():
    os.system(f'yolo mode=val task=detect model={MODEL}.pt data=coco8.yaml imgsz=32 epochs=1')


def test_val_segment():
    os.system(f'yolo mode=val task=segment model={MODEL}-seg.pt data=coco8-seg.yaml imgsz=32 epochs=1')


def test_val_classify():
    pass


# Predict checks -------------------------------------------------------------------------------------------------------
def test_predict_detect():
    os.system(f"yolo mode=predict model={MODEL}.pt source={ROOT / 'assets'}")


def test_predict_segment():
    os.system(f"yolo mode=predict model={MODEL}-seg.pt source={ROOT / 'assets'}")


def test_predict_classify():
    pass


# Export checks --------------------------------------------------------------------------------------------------------
def test_export_detect_torchscript():
    os.system(f'yolo mode=export model={MODEL}.pt format=torchscript')


def test_export_segment_torchscript():
    os.system(f'yolo mode=export model={MODEL}-seg.pt format=torchscript')


def test_export_classify_torchscript():
    pass
