# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.yolo.data.build import load_inference_source
from ultralytics.yolo.utils import LINUX, ONLINE, ROOT, SETTINGS, metrics

MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n.pt'
CFG = 'yolov8n.yaml'
SOURCE = ROOT / 'assets/bus.jpg'
SOURCE_GREYSCALE = Path(f'{SOURCE.parent / SOURCE.stem}_greyscale.jpg')
SOURCE_RGBA = Path(f'{SOURCE.parent / SOURCE.stem}_4ch.png')

# Convert SOURCE to greyscale and 4-ch
im = Image.open(SOURCE)
im.convert('L').save(SOURCE_GREYSCALE)  # greyscale
im.convert('RGBA').save(SOURCE_RGBA)  # 4-ch PNG with alpha


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
    seg_model = YOLO('yolov8n-seg.pt')
    cls_model = YOLO('yolov8n-cls.pt')
    im = cv2.imread(str(SOURCE))
    assert len(model(source=Image.open(SOURCE), save=True, verbose=True)) == 1  # PIL
    assert len(model(source=im, save=True, save_txt=True)) == 1  # ndarray
    assert len(model(source=[im, im], save=True, save_txt=True)) == 2  # batch
    assert len(list(model(source=[im, im], save=True, stream=True))) == 2  # stream
    assert len(model(torch.zeros(320, 640, 3).numpy())) == 1  # tensor to numpy
    batch = [
        str(SOURCE),  # filename
        Path(SOURCE),  # Path
        'https://ultralytics.com/images/zidane.jpg' if ONLINE else SOURCE,  # URI
        cv2.imread(str(SOURCE)),  # OpenCV
        Image.open(SOURCE),  # PIL
        np.zeros((320, 640, 3))]  # numpy
    assert len(model(batch)) == len(batch)  # multiple sources in a batch

    # Test tensor inference
    im = cv2.imread(str(SOURCE))  # OpenCV
    t = cv2.resize(im, (32, 32))
    t = torch.from_numpy(t.transpose((2, 0, 1)))
    t = torch.stack([t, t, t, t])
    results = model(t)
    assert len(results) == t.shape[0]
    results = seg_model(t)
    assert len(results) == t.shape[0]
    results = cls_model(t)
    assert len(results) == t.shape[0]


def test_predict_grey_and_4ch():
    model = YOLO(MODEL)
    for f in SOURCE_RGBA, SOURCE_GREYSCALE:
        for source in Image.open(f), cv2.imread(str(f)), f:
            model(source, save=True, verbose=True)


def test_val():
    model = YOLO(MODEL)
    model.val(data='coco8.yaml', imgsz=32)


def test_val_scratch():
    model = YOLO(CFG)
    model.val(data='coco8.yaml', imgsz=32)


def test_amp():
    if torch.cuda.is_available():
        from ultralytics.yolo.engine.trainer import check_amp
        model = YOLO(MODEL).model.cuda()
        assert check_amp(model)


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


def test_export_torchscript_scratch():
    model = YOLO(CFG)
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
    # test callback addition for prediction
    def on_predict_batch_end(predictor):  # results -> List[batch_size]
        path, _, im0s, _, _ = predictor.batch
        # print('on_predict_batch_end', im0s[0].shape)
        im0s = im0s if isinstance(im0s, list) else [im0s]
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)

    model = YOLO(MODEL)
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
    model = YOLO('yolov8n-pose.pt')
    res = model([SOURCE, SOURCE])
    res[0].plot(conf=True, boxes=False)
    res[0].plot(pil=True)
    res[0] = res[0].cpu().numpy()
    print(res[0].path, res[0].keypoints)

    model = YOLO('yolov8n-seg.pt')
    res = model([SOURCE, SOURCE])
    res[0].plot(conf=True, boxes=False, masks=True)
    res[0].plot(pil=True)
    res[0] = res[0].cpu().numpy()
    print(res[0].path, res[0].masks.data)

    model = YOLO('yolov8n.pt')
    res = model(SOURCE)
    res[0].plot(pil=True)
    res[0].plot()
    res[0] = res[0].cpu().numpy()
    print(res[0].path)

    model = YOLO('yolov8n-cls.pt')
    res = model(SOURCE)
    res[0].plot(probs=False)
    res[0].plot(pil=True)
    res[0].plot()
    res[0] = res[0].cpu().numpy()
    print(res[0].path)


def test_metrics():
    """
    Test the compute_metrics method in metrics. The expected values are from the return value of
    precision_recall_fscore_support in sklearn with same y_true and y_pred
    """

    # Binary classification
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 1, 0]
    precision, recall, f1_score, _, _, _ = metrics.compute_metrics(y_true, y_pred)
    assert precision == 0.6666666666666666
    assert recall == 0.6666666666666666
    assert f1_score == 0.6666666666666666

    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 0, 0]
    precision, recall, f1_score, _, _, _ = metrics.compute_metrics(y_true, y_pred)
    assert precision == 0.3
    assert recall == 0.375
    assert f1_score == 0.33333333333333326

    # Multi-class classification
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 1, 0, 1, 1, 2, 1, 2, 0]
    precision, recall, f1_score, _, _, _ = metrics.compute_metrics(y_true, y_pred)
    assert precision == 0.5555555555555555
    assert recall == 0.5555555555555555
    assert f1_score == 0.546031746031746

    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 3]
    y_pred = [0, 1, 0, 1, 1, 2, 1, 3, 0]
    precision, recall, f1_score, _, _, _ = metrics.compute_metrics(y_true, y_pred)
    assert precision == 0.29166666666666663
    assert recall == 0.3333333333333333
    assert f1_score == 0.30952380952380953

    # y_pred contains unknown class
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 3]
    y_pred = [0, 1, 0, 1, 1, 2, 1, 2, 0]
    precision, recall, f1_score, _, _, _ = metrics.compute_metrics(y_true, y_pred)
    assert precision == 0.41666666666666663
    assert recall == 0.4583333333333333
    assert f1_score == 0.43452380952380953

    # y_true contains unknown class
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 1, 0, 1, 1, 2, 1, 2, 3]
    precision, recall, f1_score, _, _, _ = metrics.compute_metrics(y_true, y_pred)
    assert precision == 0.5
    assert recall == 0.41666666666666663
    assert f1_score == 0.44285714285714284

    # y_true class label int numbers are not continuous.
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 4, 7]
    y_pred = [0, 1, 0, 1, 1, 2, 1, 2, 3, 4]
    precision, recall, f1_score, _, _, _ = metrics.compute_metrics(y_true, y_pred)
    assert precision == 0.3333333333333333
    assert recall == 0.3055555555555555
    assert f1_score == 0.3119047619047619

    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 4, 10]
    y_pred = [0, 1, 0, 1, 1, 2, 1, 2, 3, 0]
    precision, recall, f1_score, _, _, _ = metrics.compute_metrics(y_true, y_pred)
    assert precision == 0.27777777777777773
    assert recall == 0.3055555555555555
    assert f1_score == 0.2896825396825397

    # division by zero
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 4, 4]
    y_pred = [0, 1, 0, 1, 1, 2, 1, 2, 3, 1]
    precision, recall, f1_score, _, _, _ = metrics.compute_metrics(y_true, y_pred)
    assert precision == 0.38
    assert recall == 0.36666666666666664
    assert f1_score == 0.36
