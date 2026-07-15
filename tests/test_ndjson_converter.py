import asyncio
import hashlib
import json
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


def _write_depth_ndjson(path, base_url, depth_hash, splits=("train", "test")):
    records = [
        {"type": "dataset", "task": "depth"},
        {
            "type": "image",
            "file": "train.jpg",
            "url": f"{base_url}/train.jpg?signature=image",
            "split": splits[0],
            "depth": {
                "url": f"{base_url}/train.npy?signature=depth",
                "hash": depth_hash,
                "shape": [3, 4],
                "encoding": "npy-f32",
                "unit": "m",
            },
        },
        {
            "type": "image",
            "file": "test.jpg",
            "url": f"{base_url}/test.jpg?signature=image",
            "split": splits[1],
            "depth": {
                "url": f"{base_url}/test.npy?signature=depth",
                "hash": depth_hash,
                "shape": [3, 4],
                "encoding": "npy-f32",
                "unit": "m",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))


def test_convert_depth_ndjson_downloads_pairs_and_reuses_cache(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    for split in ("train", "test"):
        cv2.imwrite(str(source / f"{split}.jpg"), np.zeros((3, 4, 3), dtype=np.uint8))
        np.save(source / f"{split}.npy", depth)
    depth_hash = hashlib.sha256((source / "train.npy").read_bytes()).hexdigest()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=source))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = tmp_path / "depth.ndjson"
    _write_depth_ndjson(manifest, f"http://127.0.0.1:{server.server_port}", depth_hash)

    try:
        yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    data = YAML.load(yaml_path)
    assert data["task"] == "depth"
    assert data["nc"] == 1
    assert data["names"] == {0: "depth"}
    assert data["val"] == "images/test"
    assert not (yaml_path.parent / "labels").exists()
    for split in ("train", "test"):
        assert (yaml_path.parent / "images" / split / f"{split}.jpg").is_file()
        np.testing.assert_array_equal(np.load(yaml_path.parent / "depth" / split / f"{split}.npy"), depth)

    _write_depth_ndjson(
        manifest,
        f"http://127.0.0.1:{server.server_port}",
        depth_hash,
        splits=("test", "train"),
    )
    resplit_yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
    assert resplit_yaml_path != yaml_path
    assert (resplit_yaml_path.parent / "images" / "test" / "train.jpg").is_file()
    assert (resplit_yaml_path.parent / "depth" / "test" / "train.npy").is_file()
    assert (resplit_yaml_path.parent / "images" / "train" / "test.jpg").is_file()
    assert (resplit_yaml_path.parent / "depth" / "train" / "test.npy").is_file()
    assert (yaml_path.parent / "images" / "train" / "train.jpg").is_file()
    assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == resplit_yaml_path


def test_convert_depth_ndjson_rejects_incomplete_descriptor_before_download(tmp_path):
    manifest = tmp_path / "invalid.ndjson"
    records = [
        {"type": "dataset", "task": "depth"},
        {
            "type": "image",
            "file": "train.jpg",
            "url": "http://127.0.0.1:1/train.jpg",
            "split": "train",
            "depth": {
                "url": "http://127.0.0.1:1/train.npy",
                "shape": [3, 4],
                "encoding": "npy-f32",
                "unit": "m",
            },
        },
    ]
    manifest.write_text("\n".join(json.dumps(record) for record in records))

    with pytest.raises(ValueError, match="content hash"):
        asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
