import sys

import psutil
import pytest


@pytest.fixture
def measure_memory_malloc(capsys):
    import tracemalloc

    tracemalloc.start()
    yield
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")
    total_memory = sum(stat.size for stat in stats)

    with capsys.disabled():
        sys.stdout.write("\n{}".format(f"Total allocated memory: {total_memory / (1024 * 1024):.2f} MB"))


@pytest.fixture(autouse=True)
def measure_memory_psutil(capsys):
    yield
    total_memory = psutil.Process().memory_info().rss / 1024**2
    with capsys.disabled():
        sys.stdout.write("\n{}".format(f"Total allocated memory after function: {total_memory} MB"))


def test_pytorch():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    for i in range(100):
        model.predict()


def test_openvino():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.export(format="openvino")
    model = YOLO("yolov8n_openvino_model")
    for i in range(100):
        model.predict()


def test_openvino_int8():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.export(format="openvino", int8=True)
    model = YOLO("yolov8n_openvino_model")
    for i in range(100):
        model.predict()


def test_engine():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.export(format="engine")
    model = YOLO("yolov8n.engine")
    for i in range(100):
        model.predict()


def test_engine_int8():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.export(format="engine", int8=True)
    model = YOLO("yolov8n.engine")
    for i in range(100):
        model.predict()
