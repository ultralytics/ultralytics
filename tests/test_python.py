# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import urllib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from tests import CFG, IS_TMP_WRITEABLE, MODEL, SOURCE, TMP
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import MODELS, TASK2DATA, TASKS
from ultralytics.data.build import load_inference_source
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    LOGGER,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    checks,
)
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_9


def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment


def test_model_methods():
    """Test various methods and properties of the YOLO model."""
    model = YOLO(MODEL)

    # Model methods
    model.info(verbose=True, detailed=True)
    model = model.reset_weights()
    model = model.load(MODEL)
    model.to("cpu")
    model.fuse()
    model.clear_callback("on_train_start")
    model.reset_callbacks()

    # Model properties
    _ = model.names
    _ = model.device
    _ = model.transforms
    _ = model.task_map


def test_model_profile():
    """Test profiling of the YOLO model with 'profile=True' argument."""
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel()  # build model
    im = torch.randn(1, 3, 64, 64)  # requires min imgsz=64
    _ = model.predict(im, profile=True)


@pytest.mark.skipif(not IS_TMP_WRITEABLE, reason="directory is not writeable")
def test_predict_txt():
    """Test YOLO predictions with sources (file, dir, glob, recursive glob) specified in a text file."""
    txt_file = TMP / "sources.txt"
    with open(txt_file, "w") as f:
        for x in [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]:
            f.write(f"{x}\n")
    _ = YOLO(MODEL)(source=txt_file, imgsz=32)


@pytest.mark.parametrize("model_name", MODELS)
def test_predict_img(model_name):
    """Test YOLO prediction on various types of image sources."""
    model = YOLO(WEIGHTS_DIR / model_name)
    im = cv2.imread(str(SOURCE))  # uint8 numpy array
    assert len(model(source=Image.open(SOURCE), save=True, verbose=True, imgsz=32)) == 1  # PIL
    assert len(model(source=im, save=True, save_txt=True, imgsz=32)) == 1  # ndarray
    assert len(model(torch.rand((2, 3, 32, 32)), imgsz=32)) == 2  # batch-size 2 Tensor, FP32 0.0-1.0 RGB order
    assert len(model(source=[im, im], save=True, save_txt=True, imgsz=32)) == 2  # batch
    assert len(list(model(source=[im, im], save=True, stream=True, imgsz=32))) == 2  # stream
    assert len(model(torch.zeros(320, 640, 3).numpy().astype(np.uint8), imgsz=32)) == 1  # tensor to numpy
    batch = [
        str(SOURCE),  # filename
        Path(SOURCE),  # Path
        "https://ultralytics.com/images/zidane.jpg" if ONLINE else SOURCE,  # URI
        cv2.imread(str(SOURCE)),  # OpenCV
        Image.open(SOURCE),  # PIL
        np.zeros((320, 640, 3), dtype=np.uint8),  # numpy
    ]
    assert len(model(batch, imgsz=32)) == len(batch)  # multiple sources in a batch


@pytest.mark.parametrize("model", MODELS)
def test_predict_visualize(model):
    """Test model predict methods with 'visualize=True' arguments."""
    YOLO(WEIGHTS_DIR / model)(SOURCE, imgsz=32, visualize=True)


def test_predict_grey_and_4ch():
    """Test YOLO prediction on SOURCE converted to greyscale and 4-channel images."""
    im = Image.open(SOURCE)
    directory = TMP / "im4"
    directory.mkdir(parents=True, exist_ok=True)

    source_greyscale = directory / "greyscale.jpg"
    source_rgba = directory / "4ch.png"
    source_non_utf = directory / "non_UTF_测试文件_tést_image.jpg"
    source_spaces = directory / "image with spaces.jpg"

    im.convert("L").save(source_greyscale)  # greyscale
    im.convert("RGBA").save(source_rgba)  # 4-ch PNG with alpha
    im.save(source_non_utf)  # non-UTF characters in filename
    im.save(source_spaces)  # spaces in filename

    # Inference
    model = YOLO(MODEL)
    for f in source_rgba, source_greyscale, source_non_utf, source_spaces:
        for source in Image.open(f), cv2.imread(str(f)), f:
            results = model(source, save=True, verbose=True, imgsz=32)
            assert len(results) == 1  # verify that an image was run
        f.unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_youtube():
    """
    Test YouTube inference.

    Note: ConnectionError may occur during this test due to network instability or YouTube server availability.
    """
    model = YOLO(MODEL)
    try:
        model.predict("https://youtu.be/G17sBkb38XQ", imgsz=96, save=True)
    # Handle internet connection errors and 'urllib.error.HTTPError: HTTP Error 429: Too Many Requests'
    except (urllib.error.HTTPError, ConnectionError) as e:
        LOGGER.warning(f"WARNING: YouTube Test Error: {e}")


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(not IS_TMP_WRITEABLE, reason="directory is not writeable")
def test_track_stream():
    """
    Test streaming tracking (short 10 frame video) with non-default ByteTrack tracker.

    Note imgsz=160 required for tracking for higher confidence and better matches
    """
    video_url = "https://ultralytics.com/assets/decelera_portrait_min.mov"
    model = YOLO(MODEL)
    model.track(video_url, imgsz=160, tracker="bytetrack.yaml")
    model.track(video_url, imgsz=160, tracker="botsort.yaml", save_frames=True)  # test frame saving also

    # Test Global Motion Compensation (GMC) methods
    for gmc in "orb", "sift", "ecc":
        with open(ROOT / "cfg/trackers/botsort.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        tracker = TMP / f"botsort-{gmc}.yaml"
        data["gmc_method"] = gmc
        with open(tracker, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        model.track(video_url, imgsz=160, tracker=tracker)


def test_val():
    """Test the validation mode of the YOLO model."""
    YOLO(MODEL).val(data="coco8.yaml", imgsz=32, save_hybrid=True)


def test_train_scratch():
    """Test training the YOLO model from scratch."""
    model = YOLO(CFG)
    model.train(data="coco8.yaml", epochs=2, imgsz=32, cache="disk", batch=-1, close_mosaic=1, name="model")
    model(SOURCE)


def test_train_pretrained():
    """Test training the YOLO model from a pre-trained state."""
    model = YOLO(WEIGHTS_DIR / "yolov8n-seg.pt")
    model.train(data="coco8-seg.yaml", epochs=1, imgsz=32, cache="ram", copy_paste=0.5, mixup=0.5, name=0)
    model(SOURCE)


def test_all_model_yamls():
    """Test YOLO model creation for all available YAML configurations."""
    for m in (ROOT / "cfg" / "models").rglob("*.yaml"):
        if "rtdetr" in m.name:
            if TORCH_1_9:  # torch<=1.8 issue - TypeError: __init__() got an unexpected keyword argument 'batch_first'
                _ = RTDETR(m.name)(SOURCE, imgsz=640)  # must be 640
        else:
            YOLO(m.name)


def test_workflow():
    """Test the complete workflow including training, validation, prediction, and exporting."""
    model = YOLO(MODEL)
    model.train(data="coco8.yaml", epochs=1, imgsz=32, optimizer="SGD")
    model.val(imgsz=32)
    model.predict(SOURCE, imgsz=32)
    model.export(format="torchscript")


def test_predict_callback_and_setup():
    """Test callback functionality during YOLO prediction."""

    def on_predict_batch_end(predictor):
        """Callback function that handles operations at the end of a prediction batch."""
        path, im0s, _ = predictor.batch
        im0s = im0s if isinstance(im0s, list) else [im0s]
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)  # results is List[batch_size]

    model = YOLO(MODEL)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

    dataset = load_inference_source(source=SOURCE)
    bs = dataset.bs  # noqa access predictor properties
    results = model.predict(dataset, stream=True, imgsz=160)  # source already setup
    for r, im0, bs in results:
        print("test_callback", im0.shape)
        print("test_callback", bs)
        boxes = r.boxes  # Boxes object for bbox outputs
        print(boxes)


@pytest.mark.parametrize("model", MODELS)
def test_results(model):
    """Test various result formats for the YOLO model."""
    results = YOLO(WEIGHTS_DIR / model)([SOURCE, SOURCE], imgsz=160)
    for r in results:
        r = r.cpu().numpy()
        r = r.to(device="cpu", dtype=torch.float32)
        r.save_txt(txt_file=TMP / "runs/tests/label.txt", save_conf=True)
        r.save_crop(save_dir=TMP / "runs/tests/crops/")
        r.tojson(normalize=True)
        r.plot(pil=True)
        r.plot(conf=True, boxes=True)
        print(r, len(r), r.path)


def test_labels_and_crops():
    """Test output from prediction args for saving detection labels and crops."""
    imgs = [SOURCE, ASSETS / "zidane.jpg"]
    results = YOLO(WEIGHTS_DIR / "yolov8n.pt")(imgs, imgsz=160, save_txt=True, save_crop=True)
    save_path = Path(results[0].save_dir)
    for r in results:
        im_name = Path(r.path).stem
        cls_idxs = r.boxes.cls.int().tolist()
        # Check label path
        labels = save_path / f"labels/{im_name}.txt"
        assert labels.exists()
        # Check detections match label count
        assert len(r.boxes.data) == len([line for line in labels.read_text().splitlines() if line])
        # Check crops path and files
        crop_dirs = list((save_path / "crops").iterdir())
        crop_files = [f for p in crop_dirs for f in p.glob("*")]
        # Crop directories match detections
        assert all(r.names.get(c) in {d.name for d in crop_dirs} for c in cls_idxs)
        # Same number of crops as detections
        assert len([f for f in crop_files if im_name in f.name]) == len(r.boxes.data)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_data_utils():
    """Test utility functions in ultralytics/data/utils.py."""
    from ultralytics.data.utils import HUBDatasetStats, autosplit
    from ultralytics.utils.downloads import zip_directory

    # from ultralytics.utils.files import WorkingDirectory
    # with WorkingDirectory(ROOT.parent / 'tests'):

    for task in TASKS:
        file = Path(TASK2DATA[task]).with_suffix(".zip")  # i.e. coco8.zip
        download(f"https://github.com/ultralytics/hub/raw/main/example_datasets/{file}", unzip=False, dir=TMP)
        stats = HUBDatasetStats(TMP / file, task=task)
        stats.get_json(save=True)
        stats.process_images()

    autosplit(TMP / "coco8")
    zip_directory(TMP / "coco8/images/val")  # zip


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_data_converter():
    """Test dataset converters."""
    from ultralytics.data.converter import coco80_to_coco91_class, convert_coco

    file = "instances_val2017.json"
    download(f"https://github.com/ultralytics/yolov5/releases/download/v1.0/{file}", dir=TMP)
    convert_coco(labels_dir=TMP, save_dir=TMP / "yolo_labels", use_segments=True, use_keypoints=False, cls91to80=True)
    coco80_to_coco91_class()


def test_data_annotator():
    """Test automatic data annotation."""
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(
        ASSETS,
        det_model=WEIGHTS_DIR / "yolov8n.pt",
        sam_model=WEIGHTS_DIR / "mobile_sam.pt",
        output_dir=TMP / "auto_annotate_labels",
    )


def test_events():
    """Test event sending functionality."""
    from ultralytics.hub.utils import Events

    events = Events()
    events.enabled = True
    cfg = copy(DEFAULT_CFG)  # does not require deepcopy
    cfg.mode = "test"
    events(cfg)


def test_cfg_init():
    """Test configuration initialization utilities."""
    from ultralytics.cfg import check_dict_alignment, copy_default_cfg, smart_value

    with contextlib.suppress(SyntaxError):
        check_dict_alignment({"a": 1}, {"b": 2})
    copy_default_cfg()
    (Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")).unlink(missing_ok=False)
    [smart_value(x) for x in ["none", "true", "false"]]


def test_utils_init():
    """Test initialization utilities."""
    from ultralytics.utils import get_git_branch, get_git_origin_url, get_ubuntu_version, is_github_action_running

    get_ubuntu_version()
    is_github_action_running()
    get_git_origin_url()
    get_git_branch()


def test_utils_checks():
    """Test various utility checks."""
    checks.check_yolov5u_filename("yolov5n.pt")
    checks.git_describe(ROOT)
    checks.check_requirements()  # check requirements.txt
    checks.check_imgsz([600, 600], max_dim=1)
    checks.check_imshow(warn=True)
    checks.check_version("ultralytics", "8.0.0")
    checks.print_args()


@pytest.mark.skipif(WINDOWS, reason="Windows profiling is extremely slow (cause unknown)")
def test_utils_benchmarks():
    """Test model benchmarking."""
    from ultralytics.utils.benchmarks import ProfileModels

    ProfileModels(["yolov8n.yaml"], imgsz=32, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


def test_utils_torchutils():
    """Test Torch utility functions."""
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.utils.torch_utils import get_flops_with_torch_profiler, profile, time_sync

    x = torch.randn(1, 64, 20, 20)
    m = Conv(64, 64, k=1, s=2)

    profile(x, [m], n=3)
    get_flops_with_torch_profiler(m)
    time_sync()


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_utils_downloads():
    """Test file download utilities."""
    from ultralytics.utils.downloads import get_google_drive_file_info

    get_google_drive_file_info("https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link")


def test_utils_ops():
    """Test various operations utilities."""
    from ultralytics.utils.ops import (
        ltwh2xywh,
        ltwh2xyxy,
        make_divisible,
        xywh2ltwh,
        xywh2xyxy,
        xywhn2xyxy,
        xywhr2xyxyxyxy,
        xyxy2ltwh,
        xyxy2xywh,
        xyxy2xywhn,
        xyxyxyxy2xywhr,
    )

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
    """Test file handling utilities."""
    from ultralytics.utils.files import file_age, file_date, get_latest_run, spaces_in_path

    file_age(SOURCE)
    file_date(SOURCE)
    get_latest_run(ROOT / "runs")

    path = TMP / "path/with spaces"
    path.mkdir(parents=True, exist_ok=True)
    with spaces_in_path(path) as new_path:
        print(new_path)


@pytest.mark.slow
def test_utils_patches_torch_save():
    """Test torch_save backoff when _torch_save throws RuntimeError."""
    from unittest.mock import MagicMock, patch

    from ultralytics.utils.patches import torch_save

    mock = MagicMock(side_effect=RuntimeError)

    with patch("ultralytics.utils.patches._torch_save", new=mock):
        with pytest.raises(RuntimeError):
            torch_save(torch.zeros(1), TMP / "test.pt")

    assert mock.call_count == 4, "torch_save was not attempted the expected number of times"


def test_nn_modules_conv():
    """Test Convolutional Neural Network modules."""
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
    """Test Neural Network block modules."""
    from ultralytics.nn.modules.block import C1, C3TR, BottleneckCSP, C3Ghost, C3x

    c1, c2 = 8, 16  # input and output channels
    x = torch.zeros(4, c1, 10, 10)  # BCHW

    # Run all modules not otherwise covered in tests
    C1(c1, c2)(x)
    C3x(c1, c2)(x)
    C3TR(c1, c2)(x)
    C3Ghost(c1, c2)(x)
    BottleneckCSP(c1, c2)(x)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_hub():
    """Test Ultralytics HUB functionalities."""
    from ultralytics.hub import export_fmts_hub, logout
    from ultralytics.hub.utils import smart_request

    export_fmts_hub()
    logout()
    smart_request("GET", "https://github.com", progress=True)


@pytest.fixture
def image():
    """Loads an image from a predefined source using OpenCV."""
    return cv2.imread(str(SOURCE))


@pytest.mark.parametrize(
    "auto_augment, erasing, force_color_jitter",
    [
        (None, 0.0, False),
        ("randaugment", 0.5, True),
        ("augmix", 0.2, False),
        ("autoaugment", 0.0, True),
    ],
)
def test_classify_transforms_train(image, auto_augment, erasing, force_color_jitter):
    """Tests classification transforms during training with various augmentation settings."""
    from ultralytics.data.augment import classify_augmentations

    transform = classify_augmentations(
        size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip=0.5,
        vflip=0.5,
        auto_augment=auto_augment,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        force_color_jitter=force_color_jitter,
        erasing=erasing,
    )

    transformed_image = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    assert transformed_image.shape == (3, 224, 224)
    assert torch.is_tensor(transformed_image)
    assert transformed_image.dtype == torch.float32


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_model_tune():
    """Tune YOLO model for performance."""
    YOLO("yolov8n-pose.pt").tune(data="coco8-pose.yaml", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
    YOLO("yolov8n-cls.pt").tune(data="imagenet10", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")


def test_model_embeddings():
    """Test YOLO model embeddings."""
    model_detect = YOLO(MODEL)
    model_segment = YOLO(WEIGHTS_DIR / "yolov8n-seg.pt")

    for batch in [SOURCE], [SOURCE, SOURCE]:  # test batch size 1 and 2
        assert len(model_detect.embed(source=batch, imgsz=32)) == len(batch)
        assert len(model_segment.embed(source=batch, imgsz=32)) == len(batch)


@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="YOLOWorld with CLIP is not supported in Python 3.12")
def test_yolo_world():
    """Tests YOLO world models with different configurations, including classes, detection, and training scenarios."""
    model = YOLO("yolov8s-world.pt")  # no YOLOv8n-world model yet
    model.set_classes(["tree", "window"])
    model(SOURCE, conf=0.01)

    model = YOLO("yolov8s-worldv2.pt")  # no YOLOv8n-world model yet
    # Training from a pretrained model. Eval is included at the final stage of training.
    # Use dota8.yaml which has fewer categories to reduce the inference time of CLIP model
    model.train(
        data="dota8.yaml",
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
    )

    # test WorWorldTrainerFromScratch
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

    model = YOLO("yolov8s-worldv2.yaml")  # no YOLOv8n-world model yet
    model.train(
        data={"train": {"yolo_data": ["dota8.yaml"]}, "val": {"yolo_data": ["dota8.yaml"]}},
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
        trainer=WorldTrainerFromScratch,
    )
