# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from ultralytics.data.converter import convert_ndjson_to_yolo
from ultralytics.data.utils import load_depth, save_depth_png
from ultralytics.utils import YAML


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve converter fixtures without writing requests to the test log."""

    def log_message(self, _format, *args):
        pass


def _write_manifest(path, base_url, *, missing_depth=False, depth_scale=None):
    records = [
        {"type": "dataset", "task": "depth", **({"depth_scale": depth_scale} if depth_scale else {})},
        {
            "type": "image",
            "file": "camera/train.jpg",
            "url": f"{base_url}/train.jpg?signature=image",
            "split": "train",
            "depth": {
                "url": f"{base_url}/train.png?signature=depth",
                "hash": "depth-train",
                "shape": [3, 4],
            },
        },
        {
            "type": "image",
            "file": "val.jpg",
            "url": f"{base_url}/test.jpg?signature=image",
            "split": "val",
            "depth": {
                "url": f"{base_url}/missing.png" if missing_depth else f"{base_url}/test.png?signature=depth",
                "hash": "depth-test",
                "shape": [3, 4],
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
        save_depth_png(source / f"{split}.png", depth)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=source))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", depth
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_convert_depth_ndjson_downloads_image_target_pairs(tmp_path, depth_server):
    """Download depth targets beside images using matching indexed stems and the default scale."""
    base_url, depth = depth_server
    manifest = tmp_path / "depth.ndjson"
    _write_manifest(manifest, base_url)

    yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))

    data = YAML.load(yaml_path)
    assert data["task"] == "depth"
    assert data["nc"] == 1
    assert data["names"] == {0: "depth"}
    assert data["depth_scale"] == 1000
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert not (yaml_path.parent / "labels").exists()
    for index, split in enumerate(("train", "val"), 1):
        assert (yaml_path.parent / "images" / split / f"{index}.jpg").is_file()
        np.testing.assert_allclose(load_depth(yaml_path.parent / "depth" / split / f"{index}.png"), depth, atol=1e-3)


def test_convert_depth_ndjson_preserves_scale(tmp_path, depth_server):
    """Copy a dataset-level PNG scale into the generated training YAML."""
    base_url, _ = depth_server
    manifest = tmp_path / "depth.ndjson"
    _write_manifest(manifest, base_url, depth_scale=256)

    data = YAML.load(asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")))

    assert data["depth_scale"] == 256


def test_convert_depth_ndjson_reuses_existing_conversion(tmp_path, depth_server, monkeypatch):
    """Reuse a complete subset and reconvert when its completion marker is invalidated."""
    base_url, _ = depth_server
    manifest = tmp_path / "depth.ndjson"
    _write_manifest(manifest, base_url)
    yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets", fraction=[1, 1]))

    monkeypatch.setattr(YAML, "save", lambda *_args, **_kwargs: pytest.fail("cache missed"))
    assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets", fraction=[1.0, 1.0])) == yaml_path

    monkeypatch.undo()
    depth_path = yaml_path.parent / "depth" / "val" / "2.png"
    depth_path.unlink()
    data = YAML.load(yaml_path)
    data.pop("complete")
    YAML.save(yaml_path, data)
    assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets", fraction=[1, 1])) == yaml_path
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
    assert not (dataset_dir / "depth" / "val" / "2.png").exists()
    assert not (dataset_dir / "data.yaml").exists()


def test_convert_depth_ndjson_rejects_missing_url(tmp_path):
    """Reject a missing depth URL before issuing downloads."""
    manifest = tmp_path / "invalid.ndjson"
    records = [
        {"type": "dataset", "task": "depth"},
        {
            "type": "image",
            "file": "train.jpg",
            "url": "http://127.0.0.1:1/train.jpg",
            "split": "train",
            "depth": {},
        },
    ]
    manifest.write_text("\n".join(json.dumps(record) for record in records))

    with pytest.raises(ValueError, match=r"missing depth\.url"):
        asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))


def test_convert_ndjson_selects_split_fractions(tmp_path, depth_server):
    """Select requested splits while preserving full metadata and two-item list behavior."""
    base_url, _ = depth_server
    records = [
        {"type": "dataset", "task": "detect"},
        *[
            {
                "type": "image",
                "file": f"{index}.jpg",
                "url": f"{base_url}/train.jpg?signature={index}",
                "split": "test" if index == 9 else "train",
                "annotations": {"boxes": [[2 if index == 9 else 0, 0.5, 0.5, 1, 1]]},
            }
            for index in range(10)
        ],
    ]
    manifest = tmp_path / "detect.ndjson"
    manifest.write_text("\n".join(json.dumps(record) for record in records))

    for fraction, expected_test in (([0.25, 1], {"10.jpg"}), ([0.25, 1, 0], set())):
        yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets", fraction=fraction))
        files = [
            {p.name for p in (yaml_path.parent / "images" / split).glob("*")} for split in ("train", "val", "test")
        ]
        assert files == [{"1.jpg", "9.jpg"}, {"8.jpg"}, expected_test]
        assert YAML.load(yaml_path)["nc"] == 3
