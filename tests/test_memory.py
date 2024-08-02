import os
import sys

import psutil
import pytest

from ultralytics import YOLO


@pytest.fixture(autouse=True)
def measure_memory_psutil(capsys):
    base_memory = psutil.Process().memory_info().rss / 1024**2
    yield
    total_memory = psutil.Process().memory_info().rss / 1024**2
    with capsys.disabled():
        sys.stdout.write("\n{}".format(f"Total allocated memory after function: {total_memory - base_memory} MB"))


# Exports
def test_export_openvino():
    if os.path.exists("yolov8n_openvino_model"):
        return
    model = YOLO("yolov8n.pt")
    model.export(format="openvino")


def test_export_openvino_int8():
    if os.path.exists("yolov8n_int8_openvino_model"):
        return
    model = YOLO("yolov8n.pt")
    model.export(format="openvino", int8=True)


def test_export_engine_int8():
    if os.path.exists("yolov8n_int8.engine"):
        return
    model = YOLO("yolov8n.pt")
    model.export(format="engine", int8=True)
    os.rename("yolov8n.engine", "yolov8n_int8.engine")


def test_export_engine():
    if os.path.exists("yolov8n.engine"):
        return
    model = YOLO("yolov8n.pt")
    model.export(format="engine")


# Load and inference


def test_pytorch():
    model = YOLO("yolov8n.pt")
    for i in range(10):
        model.predict()
    return model


def test_openvino():
    model = YOLO("yolov8n_openvino_model")
    for i in range(10):
        model.predict()
    return model


def test_openvino_int8():
    model = YOLO("yolov8n_int8_openvino_model")
    for i in range(10):
        model.predict()
    return model


def test_engine():
    model = YOLO("yolov8n.engine")
    for i in range(10):
        model.predict()
    return model


def test_engine_int8():
    model = YOLO("yolov8n_int8.engine")
    for i in range(10):
        model.predict()
    return model
