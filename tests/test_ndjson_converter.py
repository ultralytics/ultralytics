# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest
import xxhash

from ultralytics.data.converter import convert_ndjson_to_yolo
from ultralytics.utils import YAML


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve converter fixtures without writing requests to the test log."""

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
        {"type": "dataset", "task": "depth"},
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
        with pytest.raises(RuntimeError, match="validate all depth pairs"):
            asyncio.run(convert_ndjson_to_yolo(manifest, tmp_path / "datasets"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
