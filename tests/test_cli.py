# Ultralytics YOLO 🚀, AGPL-3.0 license

import subprocess

import pytest

from ultralytics.utils import ASSETS, WEIGHTS_DIR
from ultralytics.utils.checks import cuda_device_count, cuda_is_available

CUDA_IS_AVAILABLE = cuda_is_available()
CUDA_DEVICE_COUNT = cuda_device_count()
TASK_ARGS = [
    ('detect', 'yolov8n', 'coco8.yaml'),
    ('segment', 'yolov8n-seg', 'coco8-seg.yaml'),
    ('classify', 'yolov8n-cls', 'imagenet10'),
    ('pose', 'yolov8n-pose', 'coco8-pose.yaml'), ]  # (task, model, data)
EXPORT_ARGS = [
    ('yolov8n', 'torchscript'),
    ('yolov8n-seg', 'torchscript'),
    ('yolov8n-cls', 'torchscript'),
    ('yolov8n-pose', 'torchscript'), ]  # (model, format)


def run(cmd):
    """Execute a shell command using subprocess."""
    subprocess.run(cmd.split(), check=True)


def test_special_modes():
    """Test various special command modes of YOLO."""
    run('yolo help')
    run('yolo checks')
    run('yolo version')
    run('yolo settings reset')
    run('yolo cfg')


@pytest.mark.parametrize('task,model,data', TASK_ARGS)
def test_train(task, model, data):
    """Test YOLO training for a given task, model, and data."""
    run(f'yolo train {task} model={model}.yaml data={data} imgsz=32 epochs=1 cache=disk')


@pytest.mark.parametrize('task,model,data', TASK_ARGS)
def test_val(task, model, data):
    """Test YOLO validation for a given task, model, and data."""
    run(f'yolo val {task} model={WEIGHTS_DIR / model}.pt data={data} imgsz=32 save_txt save_json')


@pytest.mark.parametrize('task,model,data', TASK_ARGS)
def test_predict(task, model, data):
    """Test YOLO prediction on sample assets for a given task and model."""
    run(f'yolo predict model={WEIGHTS_DIR / model}.pt source={ASSETS} imgsz=32 save save_crop save_txt')


@pytest.mark.parametrize('model,format', EXPORT_ARGS)
def test_export(model, format):
    """Test exporting a YOLO model to different formats."""
    run(f'yolo export model={WEIGHTS_DIR / model}.pt format={format} imgsz=32')


def test_rtdetr(task='detect', model='yolov8n-rtdetr.yaml', data='coco8.yaml'):
    """Test the RTDETR functionality with the Ultralytics framework."""
    # Warning: MUST use imgsz=640
    run(f'yolo train {task} model={model} data={data} --imgsz= 640 epochs =1, cache = disk')  # add coma, spaces to args
    run(f"yolo predict {task} model={model} source={ASSETS / 'bus.jpg'} imgsz=640 save save_crop save_txt")


def test_fastsam(task='segment', model=WEIGHTS_DIR / 'FastSAM-s.pt', data='coco8-seg.yaml'):
    """Test FastSAM segmentation functionality within Ultralytics."""
    source = ASSETS / 'bus.jpg'

    run(f'yolo segment val {task} model={model} data={data} imgsz=32')
    run(f'yolo segment predict model={model} source={source} imgsz=32 save save_crop save_txt')

    from ultralytics import FastSAM
    from ultralytics.models.fastsam import FastSAMPrompt
    from ultralytics.models.sam import Predictor

    # Create a FastSAM model
    sam_model = FastSAM(model)  # or FastSAM-x.pt

    # Run inference on an image
    everything_results = sam_model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

    # Remove small regions
    new_masks, _ = Predictor.remove_small_regions(everything_results[0].masks.data, min_area=20)

    # Everything prompt
    prompt_process = FastSAMPrompt(source, everything_results, device='cpu')
    ann = prompt_process.everything_prompt()

    # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

    # Text prompt
    ann = prompt_process.text_prompt(text='a photo of a dog')

    # Point prompt
    # points default [[0,0]] [[x1,y1],[x2,y2]]
    # point_label default [0] [1,0] 0:background, 1:foreground
    ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
    prompt_process.plot(annotations=ann, output='./')


def test_mobilesam():
    """Test MobileSAM segmentation functionality using Ultralytics."""
    from ultralytics import SAM

    # Load the model
    model = SAM(WEIGHTS_DIR / 'mobile_sam.pt')

    # Source
    source = ASSETS / 'zidane.jpg'

    # Predict a segment based on a point prompt
    model.predict(source, points=[900, 370], labels=[1])

    # Predict a segment based on a box prompt
    model.predict(source, bboxes=[439, 437, 524, 709])

    # Predict all
    # model(source)


# Slow Tests -----------------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize('task,model,data', TASK_ARGS)
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason='DDP is not available')
def test_train_gpu(task, model, data):
    """Test YOLO training on GPU(s) for various tasks and models."""
    run(f'yolo train {task} model={model}.yaml data={data} imgsz=32 epochs=1 device=0')  # single GPU
    run(f'yolo train {task} model={model}.pt data={data} imgsz=32 epochs=1 device=0,1')  # multi GPU
