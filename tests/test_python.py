# Ultralytics YOLO 🚀, AGPL-3.0 license

import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
from ultralytics.data.build import load_inference_source
from ultralytics.utils import LINUX, MACOS, ONLINE, ROOT, SETTINGS
from ultralytics.utils.torch_utils import TORCH_1_9

WEIGHTS_DIR = Path(SETTINGS['weights_dir'])
MODEL = WEIGHTS_DIR / 'path with spaces' / 'yolov8n.pt'  # test spaces in path
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
    model(SOURCE, imgsz=32)


def test_model_info():
    model = YOLO(MODEL)
    model.info(verbose=True)


def test_model_fuse():
    model = YOLO(MODEL)
    model.fuse()


def test_predict_dir():
    model = YOLO(MODEL)
    model(source=ROOT / 'assets', imgsz=32)


def test_predict_img():
    model = YOLO(MODEL)
    seg_model = YOLO(WEIGHTS_DIR / 'yolov8n-seg.pt')
    cls_model = YOLO(WEIGHTS_DIR / 'yolov8n-cls.pt')
    pose_model = YOLO(WEIGHTS_DIR / 'yolov8n-pose.pt')
    im = cv2.imread(str(SOURCE))
    assert len(model(source=Image.open(SOURCE), save=True, verbose=True, imgsz=32)) == 1  # PIL
    assert len(model(source=im, save=True, save_txt=True, imgsz=32)) == 1  # ndarray
    assert len(model(source=[im, im], save=True, save_txt=True, imgsz=32)) == 2  # batch
    assert len(list(model(source=[im, im], save=True, stream=True, imgsz=32))) == 2  # stream
    assert len(model(torch.zeros(320, 640, 3).numpy(), imgsz=32)) == 1  # tensor to numpy
    batch = [
        str(SOURCE),  # filename
        Path(SOURCE),  # Path
        'https://ultralytics.com/images/zidane.jpg' if ONLINE else SOURCE,  # URI
        cv2.imread(str(SOURCE)),  # OpenCV
        Image.open(SOURCE),  # PIL
        np.zeros((320, 640, 3))]  # numpy
    assert len(model(batch, imgsz=32)) == len(batch)  # multiple sources in a batch

    # Test tensor inference
    im = cv2.imread(str(SOURCE))  # OpenCV
    t = cv2.resize(im, (32, 32))
    t = ToTensor()(t)
    t = torch.stack([t, t, t, t])
    results = model(t, imgsz=32)
    assert len(results) == t.shape[0]
    results = seg_model(t, imgsz=32)
    assert len(results) == t.shape[0]
    results = cls_model(t, imgsz=32)
    assert len(results) == t.shape[0]
    results = pose_model(t, imgsz=32)
    assert len(results) == t.shape[0]


def test_predict_grey_and_4ch():
    model = YOLO(MODEL)
    for f in SOURCE_RGBA, SOURCE_GREYSCALE:
        for source in Image.open(f), cv2.imread(str(f)), f:
            model(source, save=True, verbose=True, imgsz=32)


def test_track_stream():
    # Test YouTube streaming inference (short 10 frame video) with non-default ByteTrack tracker
    model = YOLO(MODEL)
    model.track('https://youtu.be/G17sBkb38XQ', imgsz=32, tracker='bytetrack.yaml')


def test_val():
    model = YOLO(MODEL)
    model.val(data='coco8.yaml', imgsz=32)


def test_amp():
    if torch.cuda.is_available():
        from ultralytics.utils.checks import check_amp
        model = YOLO(MODEL).model.cuda()
        assert check_amp(model)


def test_train_scratch():
    model = YOLO(CFG)
    model.train(data='coco8.yaml', epochs=1, imgsz=32, cache='disk', batch=-1)  # test disk caching with AutoBatch
    model(SOURCE)


def test_train_pretrained():
    model = YOLO(MODEL)
    model.train(data='coco8.yaml', epochs=1, imgsz=32, cache='ram')  # test RAM caching
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
    if not MACOS:
        model = YOLO(MODEL)
        f = model.export(format='openvino')
        YOLO(f)(SOURCE)  # exported model inference


def test_export_coreml():  # sourcery skip: move-assign
    model = YOLO(MODEL)
    model.export(format='coreml', nms=True)
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
    for m in (ROOT / 'cfg' / 'models').rglob('*.yaml'):
        if 'rtdetr' in m.name:
            if TORCH_1_9:  # torch<=1.8 issue - TypeError: __init__() got an unexpected keyword argument 'batch_first'
                RTDETR(m.name)
        else:
            YOLO(m.name)


def test_workflow():
    model = YOLO(MODEL)
    model.train(data='coco8.yaml', epochs=1, imgsz=32)
    model.val()
    model.predict(SOURCE)
    model.export(format='onnx')  # export a model to ONNX format


def test_predict_callback_and_setup():
    # Test callback addition for prediction
    def on_predict_batch_end(predictor):  # results -> List[batch_size]
        path, im0s, _, _ = predictor.batch
        im0s = im0s if isinstance(im0s, list) else [im0s]
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)

    model = YOLO(MODEL)
    model.add_callback('on_predict_batch_end', on_predict_batch_end)

    dataset = load_inference_source(source=SOURCE)
    bs = dataset.bs  # noqa access predictor properties
    results = model.predict(dataset, stream=True)  # source already setup
    for r, im0, bs in results:
        print('test_callback', im0.shape)
        print('test_callback', bs)
        boxes = r.boxes  # Boxes object for bbox outputs
        print(boxes)


def test_results():
    for m in 'yolov8n-pose.pt', 'yolov8n-seg.pt', 'yolov8n.pt', 'yolov8n-cls.pt':
        model = YOLO(m)
        results = model([SOURCE, SOURCE])
        for r in results:
            r = r.cpu().numpy()
            r = r.to(device='cpu', dtype=torch.float32)
            r.save_txt(txt_file='runs/tests/label.txt', save_conf=True)
            r.save_crop(save_dir='runs/tests/crops/')
            r.tojson(normalize=True)
            r.plot(pil=True)
            r.plot(conf=True, boxes=True)
            print(r)
            print(r.path)
            for k in r.keys:
                print(getattr(r, k))


def test_data_utils():
    # Test functions in ultralytics/data/utils.py
    from ultralytics.data.utils import HUBDatasetStats, autosplit, zip_directory
    from ultralytics.utils.downloads import download

    # from ultralytics.utils.files import WorkingDirectory
    # with WorkingDirectory(ROOT.parent / 'tests'):

    Path('tests/coco8.zip').unlink(missing_ok=True)
    Path('coco8.zip').unlink(missing_ok=True)
    download('https://github.com/ultralytics/hub/raw/master/example_datasets/coco8.zip', unzip=False)
    shutil.move('coco8.zip', 'tests')
    shutil.rmtree('tests/coco8', ignore_errors=True)
    stats = HUBDatasetStats('tests/coco8.zip', task='detect')
    stats.get_json(save=False)
    stats.process_images()

    autosplit('tests/coco8')
    zip_directory('tests/coco8/images/val')  # zip
    shutil.rmtree('tests/coco8', ignore_errors=True)
    shutil.rmtree('tests/coco8-hub', ignore_errors=True)
