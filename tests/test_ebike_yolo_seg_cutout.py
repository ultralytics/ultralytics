# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import base64
import importlib.util
import io
import json
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

APP_PATH = Path(__file__).parents[1] / "validate/ebike_yolo_seg_cutout/app.py"
SPEC = importlib.util.spec_from_file_location("ebike_yolo_seg_cutout_app", APP_PATH)
APP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)


def _result(classes, confidences, boxes, masks=None):
    return SimpleNamespace(
        names={0: "person", 2: "car", 3: "motorcycle"},
        boxes=SimpleNamespace(
            cls=np.asarray(classes, dtype=np.float32),
            conf=np.asarray(confidences, dtype=np.float32),
            xyxy=np.asarray(boxes, dtype=np.float32),
        ),
        masks=None if masks is None else SimpleNamespace(data=np.asarray(masks)),
    )


def _image_payload(name="bike.jpg"):
    image = Image.new("RGB", (12, 10), (220, 225, 230))
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return {"name": name, "data": f"data:image/jpeg;base64,{encoded}"}


def _fake_cutout_result(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cutout_path = output_dir / "vehicle_cutout.png"
    Image.new("RGBA", (4, 4), (10, 20, 30, 255)).save(cutout_path)
    return APP.BatchCutoutResult(
        selection=APP.SegmentationSelection(0, 3, "motorcycle", (0, 0, 4, 4), 0.9),
        cutout_path=cutout_path,
    )


def test_select_target_instance_uses_highest_confidence_mask_index():
    result = _result(
        classes=[3, 2, 3],
        confidences=[0.62, 0.99, 0.91],
        boxes=[[1.2, 2.1, 8.8, 9.2], [0, 0, 20, 20], [2.2, 3.1, 15.7, 11.4]],
    )

    selection = APP.select_target_instance(result, (12, 16), target_class_id=3)

    assert selection.index == 2
    assert selection.class_id == 3
    assert selection.class_name == "motorcycle"
    assert selection.box == (2, 3, 16, 12)
    assert selection.confidence == pytest.approx(0.91)


def test_select_target_instance_does_not_replace_class_3_with_other_class():
    result = _result(classes=[2], confidences=[0.99], boxes=[[0, 0, 15, 11]])

    with pytest.raises(APP.TargetNotDetectedError, match="3 号类别 motorcycle"):
        APP.select_target_instance(result, (12, 16), target_class_id=3)


def test_pipeline_binds_selected_box_and_mask_by_same_index(tmp_path):
    image_path = tmp_path / "bike.jpg"
    Image.new("RGB", (16, 12), (180, 190, 200)).save(image_path)
    masks = np.zeros((2, 12, 16), dtype=np.float32)
    masks[1, 4:10, 3:14] = 1
    result = _result(
        classes=[3, 3],
        confidences=[0.42, 0.93],
        boxes=[[1, 1, 8, 8], [2, 3, 15, 11]],
        masks=masks,
    )
    calls = {}

    class FakeModel:
        def predict(self, **kwargs):
            calls.update(kwargs)
            return [result]

    pipeline = APP.YoloSegCutoutPipeline(
        APP.CutoutConfig(model_path=tmp_path / "ignored.pt", target_class_id=3, device="cpu"),
        model=FakeModel(),
    )

    output = pipeline.cutout(image_path, tmp_path / "output")
    batch_output = pipeline.cutout_only(image_path, tmp_path / "batch-output")

    assert output.selection.index == 1
    assert output.selection.confidence == pytest.approx(0.93)
    assert calls["classes"] == [3]
    assert calls["retina_masks"] is True
    assert calls["device"] == "cpu"
    with Image.open(output.cutout_path) as cutout:
        alpha = np.asarray(cutout.getchannel("A"))
        assert cutout.mode == "RGBA"
        assert set(np.unique(alpha)) == {0, 255}
        assert np.count_nonzero(alpha == 255) == 6 * 11
    assert batch_output.selection.index == 1
    assert [path.name for path in (tmp_path / "batch-output").iterdir()] == ["vehicle_cutout.png"]


def test_normalize_mask_restores_original_size_and_rejects_empty_mask():
    small_mask = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    mask = APP._normalize_mask(small_mask, (8, 10))

    assert mask.shape == (8, 10)
    assert mask.dtype == bool
    assert np.any(mask)
    with pytest.raises(APP.SegmentationError, match="轮廓为空"):
        APP._normalize_mask(np.zeros((2, 2), dtype=np.float32), (8, 10))


def test_load_image_rejects_pixel_limit_before_exif_copy(tmp_path, monkeypatch):
    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (10, 10), "white").save(image_path)
    monkeypatch.setattr(APP, "MAX_IMAGE_PIXELS", 50)

    def fail_if_called(_):
        raise AssertionError("超像素图片不应进入 EXIF 转置")

    monkeypatch.setattr(APP.ImageOps, "exif_transpose", fail_if_called)

    with pytest.raises(ValueError, match="图片像素过大"):
        APP._load_rgb_image(image_path)


def test_batch_continues_after_undetected_image_and_packages_relative_paths(tmp_path, monkeypatch):
    model_path = tmp_path / "yolo11l-seg.pt"
    model_path.write_bytes(b"test")

    class FakePipeline:
        def __init__(self):
            self.config = APP.CutoutConfig(model_path=model_path)
            self.calls = 0

        @staticmethod
        def load_model():
            return object()

        def cutout_only(self, image_path, output_dir):
            self.calls += 1
            if self.calls == 2:
                raise APP.TargetNotDetectedError("未检测到 motorcycle")
            return _fake_cutout_result(output_dir)

    monkeypatch.setattr(APP, "RUNS_DIR", tmp_path / "runs")
    service = APP.CutoutService(FakePipeline())
    started = service.start_batch({"relative_paths": ["样本目录/子目录/a.jpg", "样本目录/b.jpg"]})
    batch_id = started["batch_id"]

    first = service.process_batch_item(
        {"batch_id": batch_id, "relative_path": "样本目录/子目录/a.jpg", "image": _image_payload("a.jpg")}
    )
    second = service.process_batch_item(
        {"batch_id": batch_id, "relative_path": "样本目录/b.jpg", "image": _image_payload("b.jpg")}
    )
    finished = service.finish_batch({"batch_id": batch_id})

    assert first["success"] is True
    assert second == {"relative_path": "样本目录/b.jpg", "success": False, "error": "未检测到 motorcycle"}
    assert finished["success_count"] == 1
    assert finished["failed_count"] == 1
    archive_path = APP.RUNS_DIR / batch_id / f"{batch_id}_cutouts.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "整车抠图/样本目录/子目录/a_jpg_cutout.png",
        }
        assert archive.getinfo("整车抠图/样本目录/子目录/a_jpg_cutout.png").compress_type == zipfile.ZIP_STORED
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["processed_count"] == 2
    assert manifest["success_count"] == 1
    assert manifest["failed_count"] == 1
    assert [item["relative_path"] for item in manifest["records"]] == ["样本目录/子目录/a.jpg", "样本目录/b.jpg"]
    assert batch_id not in service.batch_jobs
    assert not (APP.RUNS_DIR / batch_id / "cutouts").exists()
    assert service.finish_batch({"batch_id": batch_id}) == finished


def test_batch_rejects_parent_directory_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(APP, "RUNS_DIR", tmp_path / "runs")
    pipeline = SimpleNamespace(config=APP.CutoutConfig(model_path=tmp_path / "model.pt"))
    service = APP.CutoutService(pipeline)

    with pytest.raises(ValueError, match="相对路径无效"):
        service.start_batch({"relative_paths": ["../outside.jpg"]})


def test_batch_records_runtime_and_browser_failures(tmp_path, monkeypatch):
    class FailingPipeline:
        config = APP.CutoutConfig(model_path=tmp_path / "model.pt")

        @staticmethod
        def load_model():
            return object()

        @staticmethod
        def cutout_only(image_path, output_dir):
            raise RuntimeError("推理后端异常")

    monkeypatch.setattr(APP, "RUNS_DIR", tmp_path / "runs")
    service = APP.CutoutService(FailingPipeline())
    paths = ["目录/runtime.jpg", "目录/read.jpg"]
    batch_id = service.start_batch({"relative_paths": paths})["batch_id"]

    runtime_record = service.process_batch_item(
        {"batch_id": batch_id, "relative_path": paths[0], "image": _image_payload("runtime.jpg")}
    )
    browser_record = service.process_batch_item(
        {"batch_id": batch_id, "relative_path": paths[1], "error": "无法读取图片：read.jpg"}
    )
    finished = service.finish_batch({"batch_id": batch_id})

    assert runtime_record["success"] is False
    assert "RuntimeError: 推理后端异常" in runtime_record["error"]
    assert browser_record == {
        "relative_path": paths[1],
        "success": False,
        "error": "浏览器未上传图片：无法读取图片：read.jpg",
    }
    assert finished["processed_count"] == 2
    assert finished["failed_count"] == 2
    manifest = json.loads((APP.RUNS_DIR / batch_id / "manifest.json").read_text(encoding="utf-8"))
    assert [record["relative_path"] for record in manifest["records"]] == paths


def test_batch_does_not_downgrade_global_model_failure_to_single_item(tmp_path, monkeypatch):
    class BrokenModelPipeline:
        config = APP.CutoutConfig(model_path=tmp_path / "model.pt")

        @staticmethod
        def load_model():
            return object()

        @staticmethod
        def cutout_only(image_path, output_dir):
            raise APP.ModelNotReadyError("模型全局不可用")

    monkeypatch.setattr(APP, "RUNS_DIR", tmp_path / "runs")
    service = APP.CutoutService(BrokenModelPipeline())
    batch_id = service.start_batch({"relative_paths": ["目录/a.jpg"]})["batch_id"]

    with pytest.raises(APP.ModelNotReadyError, match="全局不可用"):
        service.process_batch_item(
            {"batch_id": batch_id, "relative_path": "目录/a.jpg", "image": _image_payload("a.jpg")}
        )

    assert service.batch_jobs[batch_id].records == {}


def test_batch_finish_waits_for_item_fills_missing_and_is_idempotent(tmp_path, monkeypatch):
    item_started = threading.Event()
    release_item = threading.Event()

    class BlockingPipeline:
        config = APP.CutoutConfig(model_path=tmp_path / "model.pt")

        @staticmethod
        def load_model():
            return object()

        @staticmethod
        def cutout_only(image_path, output_dir):
            item_started.set()
            assert release_item.wait(5)
            return _fake_cutout_result(output_dir)

    monkeypatch.setattr(APP, "RUNS_DIR", tmp_path / "runs")
    service = APP.CutoutService(BlockingPipeline())
    paths = ["目录/a.jpg", "目录/b.jpg"]
    batch_id = service.start_batch({"relative_paths": paths})["batch_id"]
    item_result = {}
    finish_results = []
    errors = []
    finish_barrier = threading.Barrier(3)

    def process_item():
        try:
            item_result.update(
                service.process_batch_item(
                    {"batch_id": batch_id, "relative_path": paths[0], "image": _image_payload("a.jpg")}
                )
            )
        except Exception as error:  # noqa: BLE001  # pragma: no cover - thread failures reported below
            errors.append(error)

    def finish_batch():
        try:
            finish_barrier.wait()
            finish_results.append(service.finish_batch({"batch_id": batch_id}))
        except Exception as error:  # noqa: BLE001  # pragma: no cover - thread failures reported below
            errors.append(error)

    item_thread = threading.Thread(target=process_item)
    finish_threads = [threading.Thread(target=finish_batch) for _ in range(2)]
    item_thread.start()
    assert item_started.wait(2)
    for thread in finish_threads:
        thread.start()
    finish_barrier.wait()
    assert not finish_results

    release_item.set()
    item_thread.join(5)
    for thread in finish_threads:
        thread.join(5)

    assert not errors
    assert item_result["success"] is True
    assert len(finish_results) == 2
    assert finish_results[0] == finish_results[1]
    assert finish_results[0]["processed_count"] == 2
    assert finish_results[0]["success_count"] == 1
    assert finish_results[0]["failed_count"] == 1
    manifest = json.loads((APP.RUNS_DIR / batch_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["records"][1] == {
        "relative_path": paths[1],
        "success": False,
        "error": "未处理：任务在图片上传前停止或请求中断",
    }
    assert batch_id not in service.batch_jobs


def test_health_uses_real_model_load_and_caches_global_failure(tmp_path, monkeypatch):
    model_path = tmp_path / "yolo11l-seg.pt"
    model_path.write_bytes(b"invalid checkpoint")
    load_calls = []

    class UnloadableYOLO:
        def __init__(self, path):
            load_calls.append(path)
            raise RuntimeError("checkpoint 无法反序列化")

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", UnloadableYOLO)
    pipeline = APP.YoloSegCutoutPipeline(APP.CutoutConfig(model_path=model_path))
    service = APP.CutoutService(pipeline)

    first = service.health()
    second = service.health()

    assert first["status"] == "waiting_model"
    assert first["model_ready"] is False
    assert first["model_error"] == "YOLO 分割权重无法加载：yolo11l-seg.pt"
    assert second == first
    assert load_calls == [model_path]
    with pytest.raises(APP.ModelNotReadyError, match="权重无法加载"):
        service.start_batch({"relative_paths": ["目录/a.jpg"]})
    assert not service.batch_jobs


def test_health_only_reports_ready_after_segment_task_and_class_validation(tmp_path, monkeypatch):
    model_path = tmp_path / "yolo11l-seg.pt"
    model_path.write_bytes(b"stub")

    class SegmentModel:
        def __init__(self):
            self.task = "segment"
            self.names = {3: "motorcycle"}

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda path: SegmentModel())
    pipeline = APP.YoloSegCutoutPipeline(APP.CutoutConfig(model_path=model_path))

    health = APP.CutoutService(pipeline).health()

    assert health["status"] == "model_ready"
    assert health["model_ready"] is True
    assert health["model_error"] is None


def test_server_start_and_periodic_action_clean_expired_batch_artifacts(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    model_path = tmp_path / "model.pt"
    pipeline = SimpleNamespace(
        config=APP.CutoutConfig(model_path=model_path), model=object(), load_model=lambda: object()
    )
    monkeypatch.setattr(APP, "RUNS_DIR", runs_dir)
    service = APP.CutoutService(pipeline)
    active_id = service.start_batch({"relative_paths": ["目录/a.jpg"]})["batch_id"]
    active_dir = runs_dir / active_id
    (active_dir / "cutouts").mkdir()
    (active_dir / "cutouts/00001.png").write_bytes(b"temporary")
    orphan_dir = runs_dir / "batch_20260101_000000_deadbeef"
    orphan_dir.mkdir()
    (orphan_dir / "manifest.json").write_text("{}", encoding="utf-8")
    completed_dir = runs_dir / "batch_20260101_000001_deadbeef"
    completed_dir.mkdir()
    (completed_dir / f"{completed_dir.name}_cutouts.zip").write_bytes(b"completed")
    (completed_dir / "cutouts").mkdir()
    (completed_dir / "cutouts/00001.png").write_bytes(b"duplicate")
    monkeypatch.setattr(APP, "BATCH_IDLE_TIMEOUT_SECONDS", 0)

    server = APP.CutoutWebServer(("127.0.0.1", 0), service)
    try:
        periodic_orphan = runs_dir / "batch_20260101_000002_deadbeef"
        periodic_orphan.mkdir()
        server.next_cleanup_at = 0
        server.service_actions()
    finally:
        server.server_close()

    assert active_id not in service.batch_jobs
    assert not active_dir.exists()
    assert not orphan_dir.exists()
    assert completed_dir.exists()
    assert not (completed_dir / "cutouts").exists()
    assert not periodic_orphan.exists()


def test_single_failure_removes_uploaded_source_and_case_directory(tmp_path, monkeypatch):
    class UndetectedPipeline:
        config = APP.CutoutConfig(model_path=tmp_path / "model.pt")

        @staticmethod
        def cutout(image_path, output_dir):
            raise APP.TargetNotDetectedError("未检测到 motorcycle")

    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(APP, "RUNS_DIR", runs_dir)
    service = APP.CutoutService(UndetectedPipeline())

    with pytest.raises(APP.TargetNotDetectedError, match="未检测到"):
        service.process({"image": _image_payload()})

    assert not list(runs_dir.iterdir())


def test_send_file_streams_large_archive_in_chunks(tmp_path):
    archive_path = tmp_path / "large.zip"
    content = b"x" * (2 * 1024 * 1024 + 17)
    archive_path.write_bytes(content)
    writes = []

    class RecordingWriter(io.BytesIO):
        def write(self, value):
            writes.append(len(value))
            return super().write(value)

    handler = APP.CutoutWebHandler.__new__(APP.CutoutWebHandler)
    handler.wfile = RecordingWriter()
    handler.send_response = lambda status: None
    handler.send_header = lambda key, value: None
    handler.end_headers = lambda: None

    handler._send_file(200, archive_path, "application/zip")

    assert handler.wfile.getvalue() == content
    assert writes == [1024 * 1024, 1024 * 1024, 17]


def test_web_page_exposes_single_and_directory_workflows():
    html = (APP.STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'id="singleInput"' in html
    assert 'id="batchInput"' in html
    assert 'id="batchStop"' in html
    assert 'id="singleStop"' not in html
    assert "webkitdirectory" in html
    assert 'download="电动自行车_透明抠图.png"' in html
    assert "/api/batch/start" in html
    assert "/api/batch/item" in html
    assert "/api/batch/finish" in html
    assert "relative_paths: relativePaths" in html
    assert "AbortController" in html
    assert "归档响应中断，正在重新获取归档结果" in html
    assert '"protocol"' in html
    assert "if (controller.signal.aborted) throw error" in html
    assert '["network", "timeout", "protocol"]' in html
    assert "file.size > maxImageBytes" in html
    assert "[400, 503].includes(error.status)" in html
    assert "min-width: 320px" not in html
