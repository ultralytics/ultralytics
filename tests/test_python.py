# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.yolo.data.build import load_inference_source
from ultralytics.yolo.utils import LINUX, ROOT, SETTINGS

MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n.pt'
CFG = 'yolov8n.yaml'
SOURCE = ROOT / 'assets/bus.jpg'


def test_model_forward():
    model = YOLO(CFG)
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
    model(source=ROOT / 'assets')


def test_predict_img():
    model = YOLO(MODEL)
    img = Image.open(str(SOURCE))
    output = model(source=img, save=True, verbose=True)  # PIL
    assert len(output) == 1, 'predict test failed'
    img = cv2.imread(str(SOURCE))
    output = model(source=img, save=True, save_txt=True)  # ndarray
    assert len(output) == 1, 'predict test failed'
    output = model(source=[img, img], save=True, save_txt=True)  # batch
    assert len(output) == 2, 'predict test failed'
    output = model(source=[img, img], save=True, stream=True)  # stream
    assert len(list(output)) == 2, 'predict test failed'
    tens = torch.zeros(320, 640, 3)
    output = model(tens.numpy())
    assert len(output) == 1, 'predict test failed'
    # test multiple source
    imgs = [
        SOURCE,  # filename
        Path(SOURCE),  # Path
        'https://ultralytics.com/images/zidane.jpg',  # URI
        cv2.imread(str(SOURCE)),  # OpenCV
        Image.open(SOURCE),  # PIL
        np.zeros((320, 640, 3))]  # numpy
    output = model(imgs)
    assert len(output) == 6, 'predict test failed!'


def test_val():
    model = YOLO(MODEL)
    model.val(data='coco8.yaml', imgsz=32)


def test_train_scratch():
    model = YOLO(CFG)
    model.train(data='coco8.yaml', epochs=1, imgsz=32)
    model(SOURCE)


def test_train_pretrained():
    model = YOLO(MODEL)
    model.train(data='coco8.yaml', epochs=1, imgsz=32)
    model(SOURCE)


def test_export_torchscript():
    model = YOLO(MODEL)
    f = model.export(format='torchscript')
    YOLO(f)(SOURCE)  # exported model inference


def test_export_onnx():
    model = YOLO(MODEL)
    f = model.export(format='onnx')
    YOLO(f)(SOURCE)  # exported model inference


def test_export_openvino():
    model = YOLO(MODEL)
    f = model.export(format='openvino')
    YOLO(f)(SOURCE)  # exported model inference


def test_export_coreml():  # sourcery skip: move-assign
    model = YOLO(MODEL)
    model.export(format='coreml')
    # if MACOS:
    #    YOLO(f)(SOURCE)  # model prediction only supported on macOS


def test_export_tflite(enabled=False):
    # TF suffers from install conflicts on Windows and macOS
    if enabled and LINUX:
        model = YOLO(MODEL)
        f = model.export(format='tflite')
        YOLO(f)(SOURCE)


def test_export_pb(enabled=False):
    # TF suffers from install conflicts on Windows and macOS
    if enabled and LINUX:
        model = YOLO(MODEL)
        f = model.export(format='pb')
        YOLO(f)(SOURCE)


def test_export_paddle(enabled=False):
    # Paddle protobuf requirements conflicting with onnx protobuf requirements
    if enabled:
        model = YOLO(MODEL)
        model.export(format='paddle')


def test_all_model_yamls():
    for m in list((ROOT / 'models').rglob('*.yaml')):
        YOLO(m.name)


def test_workflow():
    model = YOLO(MODEL)
    model.train(data='coco8.yaml', epochs=1, imgsz=32)
    model.val()
    model.predict(SOURCE)
    model.export(format='onnx')  # export a model to ONNX format


def test_predict_callback_and_setup():

    def on_predict_batch_end(predictor):
        # results -> List[batch_size]
        path, _, im0s, _, _ = predictor.batch
        # print('on_predict_batch_end', im0s[0].shape)
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)

    model = YOLO('yolov8n.pt')
    model.add_callback('on_predict_batch_end', on_predict_batch_end)

    dataset = load_inference_source(source=SOURCE, transforms=model.transforms)
    bs = dataset.bs  # noqa access predictor properties
    results = model.predict(dataset, stream=True)  # source already setup
    for _, (result, im0, bs) in enumerate(results):
        print('test_callback', im0.shape)
        print('test_callback', bs)
        boxes = result.boxes  # Boxes object for bbox outputs
        print(boxes)


def test_result():
    model = YOLO('yolov8n-seg.pt')
    img = str(ROOT / 'assets/bus.jpg')
    res = model([img, img])
    res[0].numpy()
    res[0].cpu().numpy()
    resimg = res[0].visualize(show_conf=False)
    print(resimg)
