# Ultralytics YOLO 🚀, GPL-3.0 license

import subprocess
from pathlib import Path

from ultralytics.yolo.utils import LINUX, ONLINE, ROOT, SETTINGS

MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n'
CFG = 'yolov8n'


def run(cmd):
    # Run a subprocess command with check=True
    subprocess.run(cmd.split(), check=True)


def test_special_modes():
    run('yolo checks')
    run('yolo settings')
    run('yolo help')


# Train checks ---------------------------------------------------------------------------------------------------------
def test_train_det():
    run(f'yolo train detect model={CFG}.yaml data=coco8.yaml imgsz=32 epochs=1 v5loader')


def test_train_seg():
    run(f'yolo train segment model={CFG}-seg.yaml data=coco8-seg.yaml imgsz=32 epochs=1')


def test_train_cls():
    run(f'yolo train classify model={CFG}-cls.yaml data=imagenet10 imgsz=32 epochs=1')


def test_train_pose():
    run(f'yolo train pose model={CFG}-pose.yaml data=coco8-pose.yaml imgsz=32 epochs=1')


# Val checks -----------------------------------------------------------------------------------------------------------
def test_val_detect():
    run(f'yolo val detect model={MODEL}.pt data=coco8.yaml imgsz=32')


def test_val_segment():
    run(f'yolo val segment model={MODEL}-seg.pt data=coco8-seg.yaml imgsz=32')


def test_val_classify():
    run(f'yolo val classify model={MODEL}-cls.pt data=imagenet10 imgsz=32')


def test_val_pose():
    run(f'yolo val pose model={MODEL}-pose.pt data=coco8-pose.yaml imgsz=32')


# Predict checks -------------------------------------------------------------------------------------------------------
def test_predict_detect():
    run(f"yolo predict model={MODEL}.pt source={ROOT / 'assets'} imgsz=32 save save_crop save_txt")
    if ONLINE:
        run(f'yolo predict model={MODEL}.pt source=https://ultralytics.com/images/bus.jpg imgsz=32')
        run(f'yolo predict model={MODEL}.pt source=https://ultralytics.com/assets/decelera_landscape_min.mov imgsz=32')
        run(f'yolo predict model={MODEL}.pt source=https://ultralytics.com/assets/decelera_portrait_min.mov imgsz=32')


def test_predict_segment():
    run(f"yolo predict model={MODEL}-seg.pt source={ROOT / 'assets'} imgsz=32 save save_txt")


def test_predict_classify():
    run(f"yolo predict model={MODEL}-cls.pt source={ROOT / 'assets'} imgsz=32 save save_txt")


def test_predict_pose():
    run(f"yolo predict model={MODEL}-pose.pt source={ROOT / 'assets'} imgsz=32 save save_txt")


# Export checks --------------------------------------------------------------------------------------------------------
def test_export_detect_torchscript():
    run(f'yolo export model={MODEL}.pt format=torchscript')


def test_export_segment_torchscript():
    run(f'yolo export model={MODEL}-seg.pt format=torchscript')


def test_export_classify_torchscript():
    run(f'yolo export model={MODEL}-cls.pt format=torchscript')


def test_export_classify_pose():
    run(f'yolo export model={MODEL}-pose.pt format=torchscript')


def test_export_detect_edgetpu(enabled=False):
    if enabled and LINUX:
        run(f'yolo export model={MODEL}.pt format=edgetpu')
