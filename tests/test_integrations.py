# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from pathlib import Path

import pytest

from ultralytics import YOLO, download
from ultralytics.utils import ASSETS, DATASETS_DIR, ROOT, SETTINGS, WEIGHTS_DIR
from ultralytics.utils.checks import check_requirements

MODEL = WEIGHTS_DIR / 'path with spaces' / 'yolov8n.pt'  # test spaces in path
CFG = 'yolov8n.yaml'
SOURCE = ASSETS / 'bus.jpg'
TMP = (ROOT / '../tests/tmp').resolve()  # temp directory for test files


@pytest.mark.skipif(not check_requirements('ray', install=False), reason='ray[tune] not installed')
def test_model_ray_tune():
    """Tune YOLO model with Ray optimization library."""
    YOLO('yolov8n-cls.yaml').tune(use_ray=True,
                                  data='imagenet10',
                                  grace_period=1,
                                  iterations=1,
                                  imgsz=32,
                                  epochs=1,
                                  plots=False,
                                  device='cpu')


@pytest.mark.skipif(not check_requirements('mlflow', install=False), reason='mlflow not installed')
def test_mlflow():
    """Test training with MLflow tracking enabled."""
    SETTINGS['mlflow'] = True
    YOLO('yolov8n-cls.yaml').train(data='imagenet10', imgsz=32, epochs=3, plots=False, device='cpu')


@pytest.mark.skipif(not check_requirements('tritonclient', install=False), reason='tritonclient[all] not installed')
def test_triton():
    """Test NVIDIA Triton Server functionalities."""
    check_requirements('tritonclient[all]')
    import subprocess
    import time

    from tritonclient.http import InferenceServerClient  # noqa

    # Create variables
    model_name = 'yolo'
    triton_repo_path = TMP / 'triton_repo'
    triton_model_path = triton_repo_path / model_name

    # Export model to ONNX
    f = YOLO(MODEL).export(format='onnx', dynamic=True)

    # Prepare Triton repo
    (triton_model_path / '1').mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / '1' / 'model.onnx')
    (triton_model_path / 'config.pbtxt').touch()

    # Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    tag = 'nvcr.io/nvidia/tritonserver:23.09-py3'  # 6.4 GB

    # Pull the image
    subprocess.call(f'docker pull {tag}', shell=True)

    # Run the Triton server and capture the container ID
    container_id = subprocess.check_output(
        f'docker run -d --rm -v {triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models',
        shell=True).decode('utf-8').strip()

    # Wait for the Triton server to start
    triton_client = InferenceServerClient(url='localhost:8000', verbose=False, ssl=False)

    # Wait until model is ready
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready(model_name)
            break
        time.sleep(1)

    # Check Triton inference
    YOLO(f'http://localhost:8000/{model_name}', 'detect')(SOURCE)  # exported model inference

    # Kill and remove the container at the end of the test
    subprocess.call(f'docker kill {container_id}', shell=True)


@pytest.mark.skipif(not check_requirements('pycocotools', install=False), reason='pycocotools not installed')
def test_pycocotools():
    """Validate model predictions using pycocotools."""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator

    # Download annotations after each dataset downloads first
    url = 'https://github.com/ultralytics/assets/releases/download/v8.1.0/'

    args = {'model': 'yolov8n.pt', 'data': 'coco8.yaml', 'save_json': True, 'imgsz': 64}
    validator = DetectionValidator(args=args)
    validator()
    validator.is_coco = True
    download(f'{url}instances_val2017.json', dir=DATASETS_DIR / 'coco8/annotations')
    _ = validator.eval_json(validator.stats)

    args = {'model': 'yolov8n-seg.pt', 'data': 'coco8-seg.yaml', 'save_json': True, 'imgsz': 64}
    validator = SegmentationValidator(args=args)
    validator()
    validator.is_coco = True
    download(f'{url}instances_val2017.json', dir=DATASETS_DIR / 'coco8-seg/annotations')
    _ = validator.eval_json(validator.stats)

    args = {'model': 'yolov8n-pose.pt', 'data': 'coco8-pose.yaml', 'save_json': True, 'imgsz': 64}
    validator = PoseValidator(args=args)
    validator()
    validator.is_coco = True
    download(f'{url}person_keypoints_val2017.json', dir=DATASETS_DIR / 'coco8-pose/annotations')
    _ = validator.eval_json(validator.stats)
