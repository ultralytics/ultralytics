# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import json
import random
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from ultralytics.data.converter import convert_ndjson_to_yolo
from ultralytics.utils import YAML


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve converter fixtures without writing requests to the test log."""

    def log_message(self, _format, *args):
        pass


def _write_manifest(path, base_url, *, missing_depth=False):
    records = [
        {"type": "dataset", "task": "depth"},
        {
            "type": "image",
            "file": "camera/train.jpg",
            "url": f"{base_url}/train.jpg?signature=image",
            "split": "train",
            "depth": {
                "url": f"{base_url}/train.npy?signature=depth",
                "hash": "depth-train",
                "shape": [3, 4],
                "encoding": "npy-f32",
                "unit": "m",
            },
        },
        {
            "type": "image",
            "file": "val.jpg",
            "url": f"{base_url}/test.jpg?signature=image",
            "split": "val",
            "depth": {
                "url": f"{base_url}/missing.npy" if missing_depth else f"{base_url}/test.npy?signature=depth",
                "hash": "depth-test",
                "shape": [3, 4],
                "encoding": "npy-f32",
                "unit": "m",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))


@pytest.fixture
def depth_server(tmp_path):
    """Serve paired image and depth fixtures over HTTP."""
    source = tmp_path / "source"
    source.mkdir()
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    for split, value in (("train", 0), ("test", 255)):
        cv2.imwrite(str(source / f"{split}.jpg"), np.full((3, 4, 3), value, dtype=np.uint8))
        np.save(source / f"{split}.npy", depth)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=source))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", depth
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_convert_depth_ndjson_downloads_image_target_pairs(tmp_path, depth_server):
    """Download depth targets beside images using matching indexed stems."""
    base_url, depth = depth_server
    manifest = tmp_path / "depth.ndjson"
    _write_manifest(manifest, base_url)

    yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))

    data = YAML.load(yaml_path)
    assert data["task"] == "depth"
    assert data["nc"] == 1
    assert data["names"] == {0: "depth"}
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert not (yaml_path.parent / "labels").exists()
    for index, split in enumerate(("train", "val"), 1):
        assert (yaml_path.parent / "images" / split / f"{index}.jpg").is_file()
        np.testing.assert_array_equal(np.load(yaml_path.parent / "depth" / split / f"{index}.npy"), depth)


def test_convert_depth_ndjson_reuses_existing_conversion(tmp_path, depth_server, monkeypatch):
    """Reuse complete depth conversions and reconvert when the completion marker is invalidated."""
    base_url, _ = depth_server
    manifest = tmp_path / "depth.ndjson"
    _write_manifest(manifest, base_url)
    yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))

    monkeypatch.setattr(YAML, "save", lambda *_args, **_kwargs: pytest.fail("cache missed"))
    assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == yaml_path

    monkeypatch.undo()
    depth_path = yaml_path.parent / "depth" / "val" / "2.npy"
    depth_path.unlink()
    data = YAML.load(yaml_path)
    data.pop("complete")
    YAML.save(yaml_path, data)
    assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == yaml_path
    assert depth_path.is_file()


def test_convert_depth_ndjson_removes_incomplete_pair(tmp_path, depth_server):
    """Fail depth conversion without leaving an image whose paired target failed to download."""
    base_url, _ = depth_server
    manifest = tmp_path / "incomplete.ndjson"
    _write_manifest(manifest, base_url, missing_depth=True)

    with pytest.raises(RuntimeError, match=r"Downloaded 1/2 images"):
        asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))

    dataset_dir = next(p for p in (tmp_path / "datasets").iterdir() if p.is_dir())
    assert not (dataset_dir / "images" / "val" / "2.jpg").exists()
    assert not (dataset_dir / "depth" / "val" / "2.npy").exists()
    assert not (dataset_dir / "data.yaml").exists()


def test_convert_depth_ndjson_rejects_incomplete_descriptor(tmp_path):
    """Reject incomplete depth descriptors before issuing downloads."""
    manifest = tmp_path / "invalid.ndjson"
    records = [
        {"type": "dataset", "task": "depth"},
        {
            "type": "image",
            "file": "train.jpg",
            "url": "http://127.0.0.1:1/train.jpg",
            "split": "train",
            "depth": {"url": "http://127.0.0.1:1/train.npy", "shape": [3, 4], "encoding": "npy-f32"},
        },
    ]
    manifest.write_text("\n".join(json.dumps(record) for record in records))

    with pytest.raises(ValueError, match="encoding='npy-f32' and unit='m'"):
        asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))


def test_convert_ndjson_preserves_non_depth_auto_split(tmp_path, depth_server):
    """Keep the existing deterministic automatic split behavior for non-depth tasks."""
    base_url, _ = depth_server
    records = [
        {"type": "dataset", "task": "detect", "class_names": {"0": "object"}},
        *[
            {
                "type": "image",
                "file": f"{index}.jpg",
                "url": f"{base_url}/train.jpg?signature={index}",
                "split": "train",
                "annotations": {"boxes": [[0, 0.5, 0.5, 1, 1]]},
            }
            for index in range(10)
        ],
    ]
    manifest = tmp_path / "detect.ndjson"
    manifest.write_text("\n".join(json.dumps(record) for record in records))

    yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))

    order = list(range(1, len(records)))
    random.Random(0).shuffle(order)
    assert (yaml_path.parent / "images" / "val" / f"{order[0]}.jpg").is_file()
