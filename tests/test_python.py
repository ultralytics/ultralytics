# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA
from ultralytics.data.build import load_inference_source
from ultralytics.utils import ASSETS, DEFAULT_CFG, LINUX, MACOS, ONLINE, ROOT, SETTINGS, WINDOWS
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_9

WEIGHTS_DIR = Path(SETTINGS['weights_dir'])
MODEL = WEIGHTS_DIR / 'path with spaces' / 'yolov8n.pt'  # test spaces in path
CFG = 'yolov8n.yaml'
SOURCE = ASSETS / 'bus.jpg'
TMP = (ROOT / '../tests/tmp').resolve()  # temp directory for test files


def test_model_forward():
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment


def test_model_methods():
    model = YOLO(MODEL)

    # Model methods
    model.info(verbose=True, detailed=True)
    model = model.reset_weights()
    model = model.load(MODEL)
    model.to('cpu')
    model.fuse()
    model.clear_callback('on_train_start')
    model._reset_callbacks()

    # Model properties
    _ = model.names
    _ = model.device
    _ = model.transforms
    _ = model.task_map


def test_model_profile():
    # Test profile=True model argument
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel()  # build model
    im = torch.randn(1, 3, 64, 64)  # requires min imgsz=64
    _ = model.predict(im, profile=True)


def test_predict_txt():
    # Write a list of sources (file, dir, glob, recursive glob) to a txt file
    txt_file = TMP / 'sources.txt'
    with open(txt_file, 'w') as f:
        for x in [ASSETS / 'bus.jpg', ASSETS, ASSETS / '*', ASSETS / '**/*.jpg']:
            f.write(f'{x}\n')
    _ = YOLO(MODEL)(source=txt_file, imgsz=32)


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
    # Convert SOURCE to greyscale and 4-ch
    im = Image.open(SOURCE)
    directory = TMP / 'im4'
    directory.mkdir(parents=True, exist_ok=True)

    source_greyscale = directory / 'greyscale.jpg'
    source_rgba = directory / '4ch.png'
    source_non_utf = directory / 'non_UTF_测试文件_tést_image.jpg'
    source_spaces = directory / 'image with spaces.jpg'

    im.convert('L').save(source_greyscale)  # greyscale
    im.convert('RGBA').save(source_rgba)  # 4-ch PNG with alpha
    im.save(source_non_utf)  # non-UTF characters in filename
    im.save(source_spaces)  # spaces in filename

    # Inference
    model = YOLO(MODEL)
    for f in source_rgba, source_greyscale, source_non_utf, source_spaces:
        for source in Image.open(f), cv2.imread(str(f)), f:
            results = model(source, save=True, verbose=True, imgsz=32)
            assert len(results) == 1  # verify that an image was run
        f.unlink()  # cleanup


@pytest.mark.skipif(not ONLINE, reason='environment is offline')
def test_track_stream():
    # Test YouTube streaming inference (short 10 frame video) with non-default ByteTrack tracker
    # imgsz=160 required for tracking for higher confidence and better matches
    import yaml

    model = YOLO(MODEL)
    model.predict('https://youtu.be/G17sBkb38XQ', imgsz=96, save=True)
    model.track('https://ultralytics.com/assets/decelera_portrait_min.mov', imgsz=160, tracker='bytetrack.yaml')
    model.track('https://ultralytics.com/assets/decelera_portrait_min.mov', imgsz=160, tracker='botsort.yaml')

    # Test Global Motion Compensation (GMC) methods
    for gmc in 'orb', 'sift', 'ecc':
        with open(ROOT / 'cfg/trackers/botsort.yaml', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        tracker = TMP / f'botsort-{gmc}.yaml'
        data['gmc_method'] = gmc
        with open(tracker, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f)
        model.track('https://ultralytics.com/assets/decelera_portrait_min.mov', imgsz=160, tracker=tracker)


def test_val():
    YOLO(MODEL).val(data='coco8.yaml', imgsz=32, save_hybrid=True)


def test_train_scratch():
    model = YOLO(CFG)
    model.train(data='coco8.yaml', epochs=2, imgsz=32, cache='disk', batch=-1, close_mosaic=1, name='model')
    model(SOURCE)


def test_train_pretrained():
    model = YOLO(WEIGHTS_DIR / 'yolov8n-seg.pt')
    model.train(data='coco8-seg.yaml', epochs=1, imgsz=32, cache='ram', copy_paste=0.5, mixup=0.5, name=0)
    model(SOURCE)


def test_export_torchscript():
    f = YOLO(MODEL).export(format='torchscript', optimize=False)
    YOLO(f)(SOURCE)  # exported model inference


def test_export_onnx():
    f = YOLO(MODEL).export(format='onnx', dynamic=True)
    YOLO(f)(SOURCE)  # exported model inference


def test_export_openvino():
    f = YOLO(MODEL).export(format='openvino')
    YOLO(f)(SOURCE)  # exported model inference


def test_export_coreml():
    if not WINDOWS:  # RuntimeError: BlobWriter not loaded with coremltools 7.0 on windows
        if MACOS:
            f = YOLO(MODEL).export(format='coreml')
            YOLO(f)(SOURCE)  # model prediction only supported on macOS for nms=False models
        else:
            YOLO(MODEL).export(format='coreml', nms=True)


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
        YOLO(MODEL).export(format='paddle')


def test_export_ncnn():
    f = YOLO(MODEL).export(format='ncnn')
    YOLO(f)(SOURCE)  # exported model inference


def test_all_model_yamls():
    for m in (ROOT / 'cfg' / 'models').rglob('*.yaml'):
        if 'rtdetr' in m.name:
            if TORCH_1_9:  # torch<=1.8 issue - TypeError: __init__() got an unexpected keyword argument 'batch_first'
                _ = RTDETR(m.name)(SOURCE, imgsz=640)  # must be 640
        else:
            YOLO(m.name)


def test_workflow():
    model = YOLO(MODEL)
    model.train(data='coco8.yaml', epochs=1, imgsz=32, optimizer='SGD')
    model.val(imgsz=32)
    model.predict(SOURCE, imgsz=32)
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
    results = model.predict(dataset, stream=True, imgsz=160)  # source already setup
    for r, im0, bs in results:
        print('test_callback', im0.shape)
        print('test_callback', bs)
        boxes = r.boxes  # Boxes object for bbox outputs
        print(boxes)


def test_results():
    for m in 'yolov8n-pose.pt', 'yolov8n-seg.pt', 'yolov8n.pt', 'yolov8n-cls.pt':
        results = YOLO(m)([SOURCE, SOURCE], imgsz=160)
        for r in results:
            r = r.cpu().numpy()
            r = r.to(device='cpu', dtype=torch.float32)
            r.save_txt(txt_file='runs/tests/label.txt', save_conf=True)
            r.save_crop(save_dir='runs/tests/crops/')
            r.tojson(normalize=True)
            r.plot(pil=True)
            r.plot(conf=True, boxes=True)
            print(r, len(r), r.path)


@pytest.mark.skipif(not ONLINE, reason='environment is offline')
def test_data_utils():
    # Test functions in ultralytics/data/utils.py
    from ultralytics.data.utils import HUBDatasetStats, autosplit
    from ultralytics.utils.downloads import zip_directory

    # from ultralytics.utils.files import WorkingDirectory
    # with WorkingDirectory(ROOT.parent / 'tests'):

    for task in 'detect', 'segment', 'pose', 'classify':
        file = Path(TASK2DATA[task]).with_suffix('.zip')  # i.e. coco8.zip
        download(f'https://github.com/ultralytics/hub/raw/main/example_datasets/{file}', unzip=False, dir=TMP)
        stats = HUBDatasetStats(TMP / file, task=task)
        stats.get_json(save=True)
        stats.process_images()

    autosplit(TMP / 'coco8')
    zip_directory(TMP / 'coco8/images/val')  # zip


@pytest.mark.skipif(not ONLINE, reason='environment is offline')
def test_data_converter():
    # Test dataset converters
    from ultralytics.data.converter import coco80_to_coco91_class, convert_coco

    file = 'instances_val2017.json'
    download(f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{file}', dir=TMP)
    convert_coco(labels_dir=TMP, use_segments=True, use_keypoints=False, cls91to80=True)
    coco80_to_coco91_class()


def test_data_annotator():
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(ASSETS, det_model='yolov8n.pt', sam_model='mobile_sam.pt', output_dir=TMP / 'auto_annotate_labels')


def test_events():
    # Test event sending
    from ultralytics.hub.utils import Events

    events = Events()
    events.enabled = True
    cfg = copy(DEFAULT_CFG)  # does not require deepcopy
    cfg.mode = 'test'
    events(cfg)


def test_cfg_init():
    from ultralytics.cfg import check_dict_alignment, copy_default_cfg, smart_value

    with contextlib.suppress(SyntaxError):
        check_dict_alignment({'a': 1}, {'b': 2})
    copy_default_cfg()
    [smart_value(x) for x in ['none', 'true', 'false']]


def test_utils_init():
    from ultralytics.utils import get_git_branch, get_git_origin_url, get_ubuntu_version, is_github_actions_ci

    get_ubuntu_version()
    is_github_actions_ci()
    get_git_origin_url()
    get_git_branch()


def test_utils_checks():
    from ultralytics.utils.checks import (check_imgsz, check_imshow, check_requirements, check_yolov5u_filename,
                                          git_describe, print_args)

    check_yolov5u_filename('yolov5n.pt')
    # check_imshow(warn=True)
    git_describe(ROOT)
    check_requirements()  # check requirements.txt
    check_imgsz([600, 600], max_dim=1)
    check_imshow()
    print_args()


def test_utils_benchmarks():
    from ultralytics.utils.benchmarks import ProfileModels

    ProfileModels(['yolov8n.yaml'], imgsz=32, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


def test_utils_torchutils():
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.utils.torch_utils import get_flops_with_torch_profiler, profile, time_sync

    x = torch.randn(1, 64, 20, 20)
    m = Conv(64, 64, k=1, s=2)

    profile(x, [m], n=3)
    get_flops_with_torch_profiler(m)
    time_sync()


@pytest.mark.skipif(not ONLINE, reason='environment is offline')
def test_utils_downloads():
    from ultralytics.utils.downloads import get_google_drive_file_info

    get_google_drive_file_info('https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link')


def test_utils_ops():
    from ultralytics.utils.ops import (ltwh2xywh, ltwh2xyxy, make_divisible, xywh2ltwh, xywh2xyxy, xywhn2xyxy,
                                       xywhr2xyxyxyxy, xyxy2ltwh, xyxy2xywh, xyxy2xywhn, xyxyxyxy2xywhr)

    make_divisible(17, torch.tensor([8]))

    boxes = torch.rand(10, 4)  # xywh
    torch.allclose(boxes, xyxy2xywh(xywh2xyxy(boxes)))
    torch.allclose(boxes, xyxy2xywhn(xywhn2xyxy(boxes)))
    torch.allclose(boxes, ltwh2xywh(xywh2ltwh(boxes)))
    torch.allclose(boxes, xyxy2ltwh(ltwh2xyxy(boxes)))

    boxes = torch.rand(10, 5)  # xywhr for OBB
    boxes[:, 4] = torch.randn(10) * 30
    torch.allclose(boxes, xyxyxyxy2xywhr(xywhr2xyxyxyxy(boxes)), rtol=1e-3)


def test_utils_files():
    from ultralytics.utils.files import file_age, file_date, get_latest_run, spaces_in_path

    file_age(SOURCE)
    file_date(SOURCE)
    get_latest_run(ROOT / 'runs')

    path = TMP / 'path/with spaces'
    path.mkdir(parents=True, exist_ok=True)
    with spaces_in_path(path) as new_path:
        print(new_path)


def test_nn_modules_conv():
    from ultralytics.nn.modules.conv import CBAM, Conv2, ConvTranspose, DWConvTranspose2d, Focus

    c1, c2 = 8, 16  # input and output channels
    x = torch.zeros(4, c1, 10, 10)  # BCHW

    # Run all modules not otherwise covered in tests
    DWConvTranspose2d(c1, c2)(x)
    ConvTranspose(c1, c2)(x)
    Focus(c1, c2)(x)
    CBAM(c1)(x)

    # Fuse ops
    m = Conv2(c1, c2)
    m.fuse_convs()
    m(x)


def test_nn_modules_block():
    from ultralytics.nn.modules.block import C1, C3TR, BottleneckCSP, C3Ghost, C3x

    c1, c2 = 8, 16  # input and output channels
    x = torch.zeros(4, c1, 10, 10)  # BCHW

    # Run all modules not otherwise covered in tests
    C1(c1, c2)(x)
    C3x(c1, c2)(x)
    C3TR(c1, c2)(x)
    C3Ghost(c1, c2)(x)
    BottleneckCSP(c1, c2)(x)


@pytest.mark.skipif(not ONLINE, reason='environment is offline')
def test_hub():
    from ultralytics.hub import export_fmts_hub, logout
    from ultralytics.hub.utils import smart_request

    export_fmts_hub()
    logout()
    smart_request('GET', 'http://github.com', progress=True)
