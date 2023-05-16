# Ultralytics YOLO 🚀, AGPL-3.0 license

import subprocess
from pathlib import Path

import pytest

from ultralytics.yolo.utils import ONLINE, ROOT, SETTINGS

WEIGHT_DIR = Path(SETTINGS['weights_dir'])
TASK_ARGS = [  # (task, model, data)
    ('detect', 'yolov8n', 'coco8.yaml'), ('segment', 'yolov8n-seg', 'coco8-seg.yaml'),
    ('classify', 'yolov8n-cls', 'imagenet10'), ('pose', 'yolov8n-pose', 'coco8-pose.yaml')]
EXPORT_ARGS = [  # (model, format)
    ('yolov8n', 'torchscript'), ('yolov8n-seg', 'torchscript'), ('yolov8n-cls', 'torchscript'),
    ('yolov8n-pose', 'torchscript')]


def run(cmd):
    # Run a subprocess command with check=True
    subprocess.run(cmd.split(), check=True)


def test_special_modes():
    run('yolo checks')
    run('yolo settings')
    run('yolo help')


@pytest.mark.parametrize('task,model,data', TASK_ARGS)
def test_train(task, model, data):
    run(f'yolo train {task} model={model}.yaml data={data} imgsz=32 epochs=1')


@pytest.mark.parametrize('task,model,data', TASK_ARGS)
def test_val(task, model, data):
    run(f'yolo val {task} model={model}.pt data={data} imgsz=32')


@pytest.mark.parametrize('task,model,data', TASK_ARGS)
def test_predict(task, model, data):
    run(f"yolo predict model={model}.pt source={ROOT / 'assets'} imgsz=32 save save_crop save_txt")
    if ONLINE:
        run(f'yolo predict model={model}.pt source=https://ultralytics.com/images/bus.jpg imgsz=32')
        run(f'yolo predict model={model}.pt source=https://ultralytics.com/assets/decelera_landscape_min.mov imgsz=32')
        run(f'yolo predict model={model}.pt source=https://ultralytics.com/assets/decelera_portrait_min.mov imgsz=32')


@pytest.mark.parametrize('model,format', EXPORT_ARGS)
def test_export(model, format):
    run(f'yolo export model={model}.pt format={format}')


# Slow Tests
@pytest.mark.slow
@pytest.mark.parametrize('task,model,data', TASK_ARGS)
def test_train_gpu(task, model, data):
    run(f'yolo train {task} model={model}.yaml data={data} imgsz=32 epochs=1 device="0"')  # single GPU
    run(f'yolo train {task} model={model}.pt data={data} imgsz=32 epochs=1 device="0,1"')  # Multi GPU
