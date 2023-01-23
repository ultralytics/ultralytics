# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path

import cv2
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.yolo.utils import ROOT, SETTINGS

MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n.pt'
CFG = 'yolov8n.yaml'
SOURCE = ROOT / 'assets/bus.jpg'


def test_model_forward():
    model = YOLO(CFG)
    model.predict(SOURCE)
    model(SOURCE)


def test_model_info():
    model = YOLO(CFG)
    model.info()
    model = YOLO(MODEL)
    model.info(verbose=True)


def test_model_fuse():
    model = YOLO(CFG)
    model.fuse()
    model = YOLO(MODEL)
    model.fuse()


def test_predict_dir():
    model = YOLO(MODEL)
    model.predict(source=ROOT / "assets")


def test_predict_img():
    model = YOLO(MODEL)
    img = Image.open(str(SOURCE))
    output = model(source=img, save=True, verbose=True)  # PIL
    assert len(output) == 1, "predict test failed"
    img = cv2.imread(str(SOURCE))
    output = model(source=img, save=True, save_txt=True)  # ndarray
    assert len(output) == 1, "predict test failed"
    output = model(source=[img, img], save=True, save_txt=True)  # batch
    assert len(output) == 2, "predict test failed"
    output = model(source=[img, img], save=True, stream=True)  # stream
    assert len(list(output)) == 2, "predict test failed"
    tens = torch.zeros(320, 640, 3)
    output = model(tens.numpy())
    assert len(output) == 1, "predict test failed"


def test_val():
    model = YOLO(MODEL)
    model.val(data="coco8.yaml", imgsz=32)


def test_train_scratch():
    model = YOLO(CFG)
    model.train(data="coco8.yaml", epochs=1, imgsz=32)
    model(SOURCE)


def test_train_pretrained():
    model = YOLO(MODEL)
    model.train(data="coco8.yaml", epochs=1, imgsz=32)
    model(SOURCE)


def test_export_torchscript():
    """
                       Format     Argument           Suffix    CPU    GPU
    0                 PyTorch            -              .pt   True   True
    1             TorchScript  torchscript     .torchscript   True   True
    2                    ONNX         onnx            .onnx   True   True
    3                OpenVINO     openvino  _openvino_model   True  False
    4                TensorRT       engine          .engine  False   True
    5                  CoreML       coreml         .mlmodel   True  False
    6   TensorFlow SavedModel  saved_model     _saved_model   True   True
    7     TensorFlow GraphDef           pb              .pb   True   True
    8         TensorFlow Lite       tflite          .tflite   True  False
    9     TensorFlow Edge TPU      edgetpu  _edgetpu.tflite  False  False
    10          TensorFlow.js         tfjs       _web_model  False  False
    11           PaddlePaddle       paddle    _paddle_model   True   True
    """
    from ultralytics.yolo.engine.exporter import export_formats
    print(export_formats())

    model = YOLO(MODEL)
    model.export(format='torchscript')


def test_export_onnx():
    model = YOLO(MODEL)
    model.export(format='onnx')


def test_export_openvino():
    model = YOLO(MODEL)
    model.export(format='openvino')


def test_export_coreml():
    model = YOLO(MODEL)
    model.export(format='coreml')


def test_export_paddle():
    model = YOLO(MODEL)
    model.export(format='paddle')


def test_all_model_yamls():
    for m in list((ROOT / 'models').rglob('*.yaml')):
        YOLO(m.name)


def test_workflow():
    model = YOLO(MODEL)
    model.train(data="coco8.yaml", epochs=1, imgsz=32)
    model.val()
    model.predict(SOURCE)
    model.export(format="onnx", opset=12)  # export a model to ONNX format
