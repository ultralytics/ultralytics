# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import hashlib
import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest
import xxhash

from ultralytics.data import converter
from ultralytics.data.converter import convert_ndjson_to_yolo
from ultralytics.utils import YAML, clean_url


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve converter fixtures without writing requests to the test log."""

    def log_message(self, _format, *args):
        pass


class _NoLengthHandler(BaseHTTPRequestHandler):
    """Serve fixture bytes without Content-Length to exercise streaming limits."""

    image = b""
    depth = b""

    def do_GET(self):
        if self.path.split("?", 1)[0].endswith("test.jpg"):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(self.depth if self.path.split("?", 1)[0].endswith(".npy") else self.image)

    def log_message(self, _format, *args):
        pass


def _write_depth_ndjson(
    path,
    base_url,
    depth_hash,
    image_hashes,
    splits=("train", "test"),
    files=("camera/train.jpg", "test.jpg"),
):
    records = [
        {"type": "dataset", "task": "depth", "path": "../../escape", "download": "malicious.py"},
        {
            "type": "image",
            "file": files[0],
            "url": f"{base_url}/train.jpg?signature=image",
            "hash": image_hashes[0],
            "width": 4,
            "height": 3,
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
            "file": files[1],
            "url": f"{base_url}/test.jpg?signature=image",
            "hash": image_hashes[1],
            "width": 4,
            "height": 3,
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


def test_convert_ndjson_preserves_non_depth_cache_identity(tmp_path):
    """Keep the established ordered, eight-character cache identity for non-Depth tasks."""
    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "val"):
        cv2.imwrite(str(source / f"{split}.jpg"), np.zeros((3, 4, 3), dtype=np.uint8))
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=source))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records = [
        {"type": "dataset", "task": "detect", "class_names": {"0": "object"}},
        {
            "type": "image",
            "file": "train.jpg",
            "url": f"http://127.0.0.1:{server.server_port}/train.jpg?signature=train",
            "split": "train",
            "annotations": {"boxes": [[0, 0.5, 0.5, 1, 1]]},
        },
        {
            "type": "image",
            "file": "val.jpg",
            "url": f"http://127.0.0.1:{server.server_port}/val.jpg?signature=val",
            "split": "val",
            "annotations": {"boxes": [[0, 0.5, 0.5, 1, 1]]},
        },
    ]
    manifest = tmp_path / "detect.ndjson"
    manifest.write_text("\n".join(json.dumps(record) for record in records))
    hasher = hashlib.sha256()
    for record in records:
        stable = {key: value for key, value in record.items() if key != "url"}
        if record.get("file"):
            stable["_source"] = clean_url(record["url"])
        hasher.update(json.dumps(stable, sort_keys=True).encode())

    try:
        yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    expected = hasher.hexdigest()[:8]
    assert yaml_path.parent.name == f"detect-{expected}"
    assert YAML.load(yaml_path)["hash"] == expected


def test_convert_depth_ndjson_bounds_chunked_images_and_redacts_urls(tmp_path, monkeypatch):
    """Enforce streamed RGB limits without exposing signed URL query strings in failures."""
    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "image.jpg"
    depth_path = source / "depth.npy"
    cv2.imwrite(str(image_path), np.zeros((3, 4, 3), dtype=np.uint8))
    np.save(depth_path, np.arange(12, dtype=np.float32).reshape(3, 4))
    _NoLengthHandler.image = image_path.read_bytes()
    _NoLengthHandler.depth = depth_path.read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NoLengthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = tmp_path / "bounded.ndjson"
    image_hash = xxhash.xxh3_128_hexdigest(_NoLengthHandler.image)
    depth_hash = xxhash.xxh3_128_hexdigest(_NoLengthHandler.depth)
    _write_depth_ndjson(manifest, f"http://127.0.0.1:{server.server_port}", depth_hash, (image_hash, image_hash))
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for record in records[1:]:
        record["bytes"] = 1
        record["url"] = f"{record['url']}&token=super-secret"
    manifest.write_text("\n".join(json.dumps(record) for record in records))
    warnings = []
    monkeypatch.setattr(converter.LOGGER, "warning", warnings.append)

    try:
        with pytest.raises(RuntimeError, match="Failed to download any images"):
            asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert warnings
    assert all("super-secret" not in warning for warning in warnings)


def test_convert_depth_ndjson_downloads_pairs_and_reuses_cache(tmp_path, monkeypatch):
    """Preserve nested image-depth pairs and reuse validated cached targets."""
    source = tmp_path / "source"
    source.mkdir()
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    for split, value in (("train", 0), ("test", 255)):
        cv2.imwrite(str(source / f"{split}.jpg"), np.full((3, 4, 3), value, dtype=np.uint8))
        np.save(source / f"{split}.npy", depth)
    depth_hash = xxhash.xxh3_128_hexdigest((source / "train.npy").read_bytes())
    image_hashes = tuple(
        xxhash.xxh3_128_hexdigest((source / f"{split}.jpg").read_bytes()) for split in ("train", "test")
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=source))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = tmp_path / "depth.ndjson"
    _write_depth_ndjson(manifest, f"http://127.0.0.1:{server.server_port}", depth_hash, image_hashes)

    try:
        yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
        cached_depth = yaml_path.parent / "depth" / "train" / "camera" / "train.npy"
        for invalid_cache in (b"corrupt cache", b""):
            cached_depth.write_bytes(invalid_cache)
            assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == yaml_path
            np.testing.assert_array_equal(np.load(cached_depth), depth)
        cached_image = yaml_path.parent / "images" / "train" / "camera" / "train.jpg"
        for invalid_cache in (b"truncated image", b""):
            cached_image.write_bytes(invalid_cache)
            assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == yaml_path
            assert cv2.imread(str(cached_image)) is not None
        cached_image.write_bytes((source / "test.jpg").read_bytes())
        assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == yaml_path
        assert cv2.imread(str(cached_image)).mean() < 1
        (yaml_path.parent / "images" / "train" / "orphan.jpg").write_bytes((source / "train.jpg").read_bytes())
        (yaml_path.parent / "depth" / "train" / "orphan.npy").write_bytes((source / "train.npy").read_bytes())
        assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == yaml_path
        assert not (yaml_path.parent / "images" / "train" / "orphan.jpg").exists()
        assert not (yaml_path.parent / "depth" / "train" / "orphan.npy").exists()

        cached_yaml = YAML.load(yaml_path)
        cached_yaml["train"] = "images/test"
        YAML.save(yaml_path, cached_yaml)
        assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == yaml_path
        assert YAML.load(yaml_path)["train"] == "images/train"

        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        for record in records[1:]:
            record["hash"] = record["hash"].upper()
            record["depth"]["hash"] = record["depth"]["hash"].upper()
        manifest.write_text("\n".join(json.dumps(record) for record in records))
        assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == yaml_path

        train_only_manifest = tmp_path / "train-only.ndjson"
        _write_depth_ndjson(
            train_only_manifest,
            f"http://127.0.0.1:{server.server_port}",
            depth_hash,
            image_hashes,
            splits=("train", "train"),
        )
        train_only_yaml = asyncio.run(convert_ndjson_to_yolo(train_only_manifest, tmp_path / "datasets"))
        records = train_only_manifest.read_text().splitlines()
        train_only_manifest.write_text("\n".join([records[0], *reversed(records[1:])]))
        with monkeypatch.context() as cache_guard:
            cache_guard.setattr(YAML, "save", lambda *_args, **_kwargs: pytest.fail("train-only cache missed"))
            assert asyncio.run(convert_ndjson_to_yolo(train_only_manifest, tmp_path / "datasets")) == train_only_yaml

        _write_depth_ndjson(
            manifest,
            f"http://127.0.0.1:{server.server_port}",
            depth_hash,
            image_hashes,
            splits=("test", "train"),
        )
        resplit_yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
        assert resplit_yaml_path != yaml_path
        assert (resplit_yaml_path.parent / "images" / "test" / "camera" / "train.jpg").is_file()
        assert (resplit_yaml_path.parent / "depth" / "test" / "camera" / "train.npy").is_file()
        assert (resplit_yaml_path.parent / "images" / "train" / "test.jpg").is_file()
        assert (resplit_yaml_path.parent / "depth" / "train" / "test.npy").is_file()
        assert (yaml_path.parent / "images" / "train" / "camera" / "train.jpg").is_file()
        assert asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets")) == resplit_yaml_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    data = YAML.load(yaml_path)
    assert data["task"] == "depth"
    assert data["nc"] == 1
    assert data["names"] == {0: "depth"}
    assert data["val"] == "images/test"
    assert "path" not in data
    assert "download" not in data
    assert not (yaml_path.parent / "labels").exists()
    for split in ("train", "test"):
        relative_path = Path("camera/train.jpg") if split == "train" else Path("test.jpg")
        assert (yaml_path.parent / "images" / split / relative_path).is_file()
        np.testing.assert_array_equal(
            np.load(yaml_path.parent / "depth" / split / relative_path.with_suffix(".npy")), depth
        )


def test_convert_depth_ndjson_does_not_reuse_images_by_filename(tmp_path):
    """Keep the record URL authoritative when different assets share an output filename."""
    source = tmp_path / "source"
    source.mkdir()
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    cv2.imwrite(str(source / "train.jpg"), np.zeros((3, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(source / "test.jpg"), np.full((3, 4, 3), 255, dtype=np.uint8))
    for split in ("train", "test"):
        np.save(source / f"{split}.npy", depth)
    depth_hash = xxhash.xxh3_128_hexdigest((source / "train.npy").read_bytes())
    image_hashes = tuple(
        xxhash.xxh3_128_hexdigest((source / f"{split}.jpg").read_bytes()) for split in ("train", "test")
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=source))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = tmp_path / "same-name.ndjson"
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        _write_depth_ndjson(manifest, base_url, depth_hash, image_hashes, files=("frame.jpg", "frame.jpg"))
        asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))

        _write_depth_ndjson(
            manifest,
            base_url,
            depth_hash,
            image_hashes,
            splits=("test", "train"),
            files=("frame.jpg", "frame.jpg"),
        )
        yaml_path = asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert cv2.imread(str(yaml_path.parent / "images" / "test" / "frame.jpg")).mean() < 1
    assert cv2.imread(str(yaml_path.parent / "images" / "train" / "frame.jpg")).mean() > 254


def test_convert_depth_ndjson_commits_concurrent_conversions_atomically(tmp_path):
    """Allow concurrent first conversions to share one complete immutable manifest directory."""
    source = tmp_path / "source"
    source.mkdir()
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    for split in ("train", "test"):
        cv2.imwrite(str(source / f"{split}.jpg"), np.zeros((3, 4, 3), dtype=np.uint8))
        np.save(source / f"{split}.npy", depth)
    depth_hash = xxhash.xxh3_128_hexdigest((source / "train.npy").read_bytes())
    image_hashes = tuple(
        xxhash.xxh3_128_hexdigest((source / f"{split}.jpg").read_bytes()) for split in ("train", "test")
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=source))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = tmp_path / "concurrent.ndjson"
    _write_depth_ndjson(manifest, f"http://127.0.0.1:{server.server_port}", depth_hash, image_hashes)

    async def convert_twice():
        return await asyncio.gather(
            convert_ndjson_to_yolo(manifest, tmp_path / "datasets"),
            convert_ndjson_to_yolo(manifest, tmp_path / "datasets"),
        )

    try:
        yaml_paths = asyncio.run(convert_twice())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert yaml_paths[0] == yaml_paths[1]
    assert YAML.load(yaml_paths[0])["hash"]
    assert len(list((tmp_path / "datasets").glob("concurrent-*/data.yaml"))) == 1


def test_convert_depth_ndjson_rejects_incomplete_descriptor_before_download(tmp_path):
    """Reject incomplete depth descriptors before issuing downloads."""
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


@pytest.mark.parametrize(
    ("file", "split"),
    [
        ("../../victim.jpg", "train"),
        ("/tmp/victim.jpg", "train"),
        ("C:/victim.jpg", "train"),
        ("camera\\victim.jpg", "train"),
        ("CON.jpg", "train"),
        ("aux/frame.jpg", "train"),
        ("frame. ", "train"),
        ("frame?.jpg", "train"),
        ("frame\0.jpg", "train"),
        ("camera/\x1f.jpg", "train"),
        ("camera/\x7f.jpg", "train"),
        ("train.jpg", "../train"),
    ],
)
def test_convert_depth_ndjson_rejects_unsafe_output_paths(tmp_path, file, split):
    """Reject manifest paths that can escape or vary across host filesystems before downloading."""
    manifest = tmp_path / "unsafe.ndjson"
    _write_depth_ndjson(manifest, "http://127.0.0.1:1", "0" * 32, ("0" * 32, "1" * 32))
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[1]["file"] = file
    records[1]["split"] = split
    manifest.write_text("\n".join(json.dumps(record) for record in records))

    with pytest.raises(ValueError, match=r"relative path|split must"):
        asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))


def test_convert_depth_ndjson_rejects_oversized_declared_images(tmp_path):
    """Reject excessive decoded dimensions before issuing downloads."""
    manifest = tmp_path / "oversized.ndjson"
    _write_depth_ndjson(manifest, "http://127.0.0.1:1", "0" * 32, ("0" * 32, "1" * 32))
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[1]["width"] = 8_001
    records[1]["height"] = 8_000
    records[1]["depth"]["shape"] = [8_000, 8_001]
    manifest.write_text("\n".join(json.dumps(record) for record in records))

    with pytest.raises(ValueError, match="may not exceed"):
        asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))


def test_convert_depth_ndjson_rejects_casefolded_output_collisions(tmp_path):
    """Reject output names that alias on case-insensitive filesystems."""
    manifest = tmp_path / "collision.ndjson"
    _write_depth_ndjson(
        manifest,
        "http://127.0.0.1:1",
        "0" * 32,
        ("0" * 32, "1" * 32),
        splits=("train", "train"),
        files=("Frame.jpg", "frame.jpg"),
    )

    with pytest.raises(ValueError, match="duplicate output path"):
        asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))


@pytest.mark.parametrize("invalid_field", ["hash", "shape"])
def test_convert_depth_ndjson_rejects_invalid_npy_payload(tmp_path, invalid_field):
    """Reject depth targets whose payload does not match declared integrity metadata."""
    source = tmp_path / "source"
    source.mkdir()
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    for split in ("train", "test"):
        cv2.imwrite(str(source / f"{split}.jpg"), np.zeros((3, 4, 3), dtype=np.uint8))
        np.save(source / f"{split}.npy", depth)
    depth_hash = xxhash.xxh3_128_hexdigest((source / "train.npy").read_bytes())
    image_hashes = tuple(
        xxhash.xxh3_128_hexdigest((source / f"{split}.jpg").read_bytes()) for split in ("train", "test")
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=source))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = tmp_path / "invalid-payload.ndjson"
    _write_depth_ndjson(manifest, f"http://127.0.0.1:{server.server_port}", depth_hash, image_hashes)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[2]["depth"][invalid_field] = "0" * 32 if invalid_field == "hash" else [2, 6]
    manifest.write_text("\n".join(json.dumps(record) for record in records))

    try:
        error = ValueError if invalid_field == "shape" else RuntimeError
        message = "declared image dimensions" if invalid_field == "shape" else "validate all depth pairs"
        with pytest.raises(error, match=message):
            asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
