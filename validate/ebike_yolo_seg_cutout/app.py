# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO11l-seg 电动自行车整车轮廓提取网页。."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import mimetypes
import re
import shutil
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
STATIC_DIR = SCRIPT_DIR / "static"
RUNS_DIR = SCRIPT_DIR / "runs"
MODELS_DIR = SCRIPT_DIR / "models"

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
ARTIFACT_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".zip"})
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_REQUEST_BYTES = math.ceil(MAX_IMAGE_BYTES * 4 / 3) + 1024 * 1024
MAX_BATCH_FILES = 2000
BATCH_IDLE_TIMEOUT_SECONDS = 60 * 60
BATCH_CLEANUP_INTERVAL_SECONDS = 60
BATCH_ID_PATTERN = re.compile(r"^batch_\d{8}_\d{6}_[0-9a-f]{8}$")

LOGGER = logging.getLogger("ebike_yolo_seg_cutout")


class ModelNotReadyError(RuntimeError):
    """表示 YOLO 分割权重缺失或类别配置不符合当前方案。."""


class TargetNotDetectedError(RuntimeError):
    """表示 YOLO 未检出指定的整车类别。."""


class SegmentationError(RuntimeError):
    """表示所选整车检测没有可用的实例掩码。."""


@dataclass(frozen=True)
class CutoutConfig:
    """保存 YOLO 整车分割的模型、类别和推理配置。."""

    model_path: Path
    target_class_id: int = 3
    confidence: float = 0.25
    imgsz: int = 640
    device: str | int = "cpu"


@dataclass(frozen=True)
class SegmentationSelection:
    """保存最高置信度整车实例及其在 Results 中的索引。."""

    index: int
    class_id: int
    class_name: str
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class CutoutResult:
    """保存单次整车轮廓提取的产物和可回溯指标。."""

    selection: SegmentationSelection
    mask_fill_ratio: float
    cutout_size: tuple[int, int]
    detection_preview_path: Path
    mask_preview_path: Path
    cutout_path: Path


@dataclass(frozen=True)
class BatchCutoutResult:
    """保存批量模式所需的最小结果，避免生成不会交付的预览图。."""

    selection: SegmentationSelection
    cutout_path: Path


@dataclass
class BatchJob:
    """保存一个目录批次的固定输入清单、处理记录和独立互斥锁。."""

    batch_id: str
    batch_dir: Path
    relative_paths: tuple[str, ...]
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: Any = field(default_factory=threading.Lock)
    updated_at: float = field(default_factory=time.monotonic)


def _to_numpy(value: Any) -> np.ndarray:
    """将 PyTorch 张量或数组统一转换为 NumPy 数组。."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _to_float(value: Any) -> float:
    """将模型标量转换为 Python ``float``。."""
    return float(value.item()) if hasattr(value, "item") else float(value)


def _load_rgb_image(path: Path) -> np.ndarray:
    """解码图片并应用 EXIF 方向，避免手机照片检测结果与像素错位。."""
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片格式：{path.suffix or '无扩展名'}")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"单张图片不能超过 {MAX_IMAGE_BYTES // 1024 // 1024} MB")
    try:
        with Image.open(path) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"图片像素过大，不能超过 {MAX_IMAGE_PIXELS:,} 像素")
            source = ImageOps.exif_transpose(source)
            return np.asarray(source.convert("RGB")).copy()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise ValueError(f"图片无法解码：{path.name}") from error


def _save_upload(value: object, target_dir: Path) -> Path:
    """解码浏览器 Data URL，并使用受控文件名保存本次输入。."""
    if not isinstance(value, dict):
        raise TypeError("上传图片数据格式错误")
    name, data_url = value.get("name"), value.get("data")
    if not isinstance(name, str) or not isinstance(data_url, str):
        raise TypeError("上传图片缺少文件名或内容")
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片格式：{suffix or '无扩展名'}")
    marker = ";base64,"
    if not data_url.startswith("data:image/") or marker not in data_url:
        raise ValueError("上传图片编码格式错误")
    try:
        content = base64.b64decode(data_url.split(marker, 1)[1], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("上传图片 Base64 解码失败") from error
    if not content:
        raise ValueError("上传图片内容为空")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"单张图片不能超过 {MAX_IMAGE_BYTES // 1024 // 1024} MB")

    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"source{suffix}"
    path.write_bytes(content)
    return path


def select_target_instance(
    result: Any,
    image_shape: tuple[int, int],
    target_class_id: int,
) -> SegmentationSelection:
    """只选择目标类别中置信度最高的实例，保留索引以绑定同一张掩码。."""
    height, width = image_shape
    best: tuple[float, float, int, np.ndarray] | None = None
    for index, (class_id, confidence, box) in enumerate(zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)):
        class_id_value = int(_to_float(class_id))
        if class_id_value != target_class_id:
            continue
        confidence_value = _to_float(confidence)
        box_array = _to_numpy(box).astype(float).reshape(-1)
        if box_array.size != 4:
            raise RuntimeError(f"YOLO 整车框坐标数量异常：{box_array.size}")
        area = max(0.0, box_array[2] - box_array[0]) * max(0.0, box_array[3] - box_array[1])
        candidate = confidence_value, area, index, box_array
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        class_name = str(result.names.get(target_class_id, target_class_id))
        raise TargetNotDetectedError(
            f"未检测到 {target_class_id} 号类别 {class_name}，请更换角度完整、光线清晰的整车图片"
        )

    confidence, _, index, (x1, y1, x2, y2) = best
    clipped_box = (
        max(0, min(width, math.floor(x1))),
        max(0, min(height, math.floor(y1))),
        max(0, min(width, math.ceil(x2))),
        max(0, min(height, math.ceil(y2))),
    )
    if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
        raise RuntimeError(f"YOLO 返回的整车框无效：{clipped_box}")
    return SegmentationSelection(
        index=index,
        class_id=target_class_id,
        class_name=str(result.names.get(target_class_id, target_class_id)),
        box=clipped_box,
        confidence=confidence,
    )


def _normalize_mask(mask_value: Any, image_shape: tuple[int, int]) -> np.ndarray:
    """将 YOLO 实例掩码还原到原图尺寸并转换为布尔掩码。."""
    height, width = image_shape
    mask = _to_numpy(mask_value).squeeze()
    if mask.ndim != 2:
        raise SegmentationError(f"YOLO 掩码维度异常：{mask.shape}")
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    mask = mask > 0.5
    if not np.any(mask):
        raise SegmentationError("YOLO 返回的整车轮廓为空")
    return mask


def _build_cutout(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """用实例掩码生成透明背景 RGBA 图，并按轮廓范围紧凑裁剪。."""
    rows = np.flatnonzero(np.any(mask, axis=1))
    columns = np.flatnonzero(np.any(mask, axis=0))
    if not len(rows) or not len(columns):
        raise SegmentationError("YOLO 返回的整车轮廓为空")
    x1, x2 = int(columns.min()), int(columns.max()) + 1
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    padding = max(2, round(max(x2 - x1, y2 - y1) * 0.02))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(rgb.shape[1], x2 + padding), min(rgb.shape[0], y2 + padding)

    cropped_rgb = rgb[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    rgba = np.empty((*cropped_rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = cropped_rgb
    rgba[..., 3] = cropped_mask.astype(np.uint8) * 255
    return Image.fromarray(rgba)


def _build_mask_preview(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """分块混合轮廓颜色，避免大图布尔索引产生整幅浮点临时数组。."""
    overlay = rgb.copy()
    accent = (20, 157, 132)
    for row_start in range(0, overlay.shape[0], 256):
        row_end = min(row_start + 256, overlay.shape[0])
        selected = mask[row_start:row_end]
        if not np.any(selected):
            continue
        chunk = overlay[row_start:row_end]
        for channel, color in enumerate(accent):
            values = chunk[..., channel]
            blended = (values[selected].astype(np.uint16) * 48 + color * 52 + 50) // 100
            values[selected] = blended.astype(np.uint8)
    return Image.fromarray(overlay)


def _save_result_images(
    rgb: np.ndarray,
    mask: np.ndarray,
    selection: SegmentationSelection,
    output_dir: Path,
) -> CutoutResult:
    """保存整车框、轮廓叠加预览和最终透明 PNG。."""
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_preview_path = output_dir / "01_vehicle_detection.jpg"
    mask_preview_path = output_dir / "02_vehicle_mask.png"
    cutout_path = output_dir / "03_vehicle_cutout.png"

    from ultralytics.utils.plotting import Annotator

    line_width = max(3, min(rgb.shape[:2]) // 250)
    x1, y1, x2, y2 = selection.box
    accent = (20, 134, 111)
    label = f"{selection.class_id} {selection.class_name}  {selection.confidence:.3f}"
    annotator = Annotator(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), line_width=line_width)
    annotator.box_label(selection.box, label, color=accent[::-1])
    preview = cv2.cvtColor(annotator.result(), cv2.COLOR_BGR2RGB)
    Image.fromarray(preview).save(detection_preview_path, quality=92, optimize=True)
    del annotator, preview

    overlay = _build_mask_preview(rgb, mask)
    overlay.save(mask_preview_path, optimize=True)
    del overlay

    cutout = _build_cutout(rgb, mask)
    cutout.save(cutout_path, optimize=True)
    box_area = max(1, (x2 - x1) * (y2 - y1))
    return CutoutResult(
        selection=selection,
        mask_fill_ratio=float(np.count_nonzero(mask)) / box_area,
        cutout_size=cutout.size,
        detection_preview_path=detection_preview_path,
        mask_preview_path=mask_preview_path,
        cutout_path=cutout_path,
    )


class YoloSegCutoutPipeline:
    """复用单个 YOLO11l-seg 模型执行整车检测、实例分割和透明图生成。."""

    def __init__(self, config: CutoutConfig, model: Any = None) -> None:
        self.config = config
        self.model = model
        self.model_error: str | None = None

    def load_model(self) -> Any:
        """加载并校验实例分割模型；确定性失败在本进程内直接复用。."""
        if self.model is not None:
            return self.model
        if self.model_error is not None:
            raise ModelNotReadyError(self.model_error)
        if not self.config.model_path.is_file():
            raise ModelNotReadyError(f"YOLO 分割权重不存在：{self.config.model_path}")
        LOGGER.info("加载 YOLO 整车分割模型，权重=%s", self.config.model_path)
        try:
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from ultralytics import YOLO

            model = YOLO(self.config.model_path)
        except Exception as error:
            self.model_error = f"YOLO 分割权重无法加载：{self.config.model_path.name}"
            raise ModelNotReadyError(self.model_error) from error
        if getattr(model, "task", None) != "segment":
            self.model_error = f"当前权重不是分割模型：task={getattr(model, 'task', None)}"
            raise ModelNotReadyError(self.model_error)
        if self.config.target_class_id not in model.names:
            self.model_error = f"当前权重不包含 {self.config.target_class_id} 号类别"
            raise ModelNotReadyError(self.model_error)
        LOGGER.info(
            "整车类别已确认，class_id=%d，class_name=%s",
            self.config.target_class_id,
            model.names[self.config.target_class_id],
        )
        self.model = model
        return model

    def _segment(self, image_path: Path) -> tuple[np.ndarray, np.ndarray, SegmentationSelection]:
        """执行一次共享分割推理，检测框和掩码始终使用同一 Results 索引。."""
        rgb = _load_rgb_image(image_path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        model = self.load_model()
        predictions = model.predict(
            source=bgr,
            classes=[self.config.target_class_id],
            conf=self.config.confidence,
            imgsz=self.config.imgsz,
            retina_masks=True,
            device=self.config.device,
            verbose=False,
        )
        if len(predictions) != 1:
            raise RuntimeError(f"YOLO 推理结果数量异常：期望 1，实际 {len(predictions)}")
        result = predictions[0]
        selection = select_target_instance(result, rgb.shape[:2], self.config.target_class_id)
        if result.masks is None or selection.index >= len(result.masks.data):
            raise SegmentationError("所选整车检测没有对应的实例掩码")
        mask = _normalize_mask(result.masks.data[selection.index], rgb.shape[:2])
        LOGGER.info(
            "检出整车，类别=%d %s，置信度=%.4f，框=%s",
            *(
                selection.class_id,
                selection.class_name,
                selection.confidence,
                selection.box,
            ),
        )
        return rgb, mask, selection

    def cutout(self, image_path: Path, output_dir: Path) -> CutoutResult:
        """为单张模式保存检测框、掩码预览和透明抠图。."""
        rgb, mask, selection = self._segment(image_path)
        return _save_result_images(rgb, mask, selection, output_dir)

    def cutout_only(self, image_path: Path, output_dir: Path) -> BatchCutoutResult:
        """为批量模式只编码最终透明 PNG，避免生成后立即删除预览图。."""
        rgb, mask, selection = self._segment(image_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        cutout_path = output_dir / "vehicle_cutout.png"
        cutout = _build_cutout(rgb, mask)
        cutout.save(cutout_path, optimize=True)
        return BatchCutoutResult(selection=selection, cutout_path=cutout_path)


def _resolve_device() -> str | int:
    """服务器优先使用 CUDA/ROCm 兼容设备，不可用时回退 CPU。."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class CutoutService:
    """管理模型复用和单次任务产物，串行保护非线程安全的推理状态。."""

    def __init__(self, pipeline: YoloSegCutoutPipeline) -> None:
        self.pipeline = pipeline
        self.lock = threading.Lock()
        self.batch_lock = threading.Lock()
        self.batch_jobs: dict[str, BatchJob] = {}

    def _cleanup_expired_batch_jobs(self) -> None:
        """回收超过闲置期限且没有在途处理的批次内存与临时目录。."""
        now = time.monotonic()
        with self.batch_lock:
            candidates = [job for job in self.batch_jobs.values() if now - job.updated_at >= BATCH_IDLE_TIMEOUT_SECONDS]
        for job in candidates:
            if not job.lock.acquire(blocking=False):
                continue
            removed = False
            try:
                with self.batch_lock:
                    if (
                        self.batch_jobs.get(job.batch_id) is job
                        and time.monotonic() - job.updated_at >= BATCH_IDLE_TIMEOUT_SECONDS
                    ):
                        self.batch_jobs.pop(job.batch_id)
                        removed = True
            finally:
                job.lock.release()
            if removed:
                try:
                    shutil.rmtree(job.batch_dir)
                except FileNotFoundError:
                    pass
                except OSError:
                    LOGGER.warning("闲置批量任务目录清理失败，任务=%s", job.batch_id, exc_info=True)
                else:
                    LOGGER.info("回收闲置批量抠图任务，任务=%s", job.batch_id)

    def _cleanup_stale_batch_directories(self) -> None:
        """清理服务重启后遗留的过期批次，并回收已归档批次的松散临时文件。."""
        if not RUNS_DIR.is_dir():
            return
        with self.batch_lock:
            active_ids = set(self.batch_jobs)
        now = time.time()
        for batch_dir in RUNS_DIR.iterdir():
            if not batch_dir.is_dir() or not BATCH_ID_PATTERN.fullmatch(batch_dir.name) or batch_dir.name in active_ids:
                continue
            if (batch_dir / f"{batch_dir.name}_cutouts.zip").is_file():
                self._cleanup_completed_batch_temporary_files(batch_dir)
                continue
            try:
                expired = now - batch_dir.stat().st_mtime >= BATCH_IDLE_TIMEOUT_SECONDS
            except FileNotFoundError:
                continue
            if expired:
                try:
                    shutil.rmtree(batch_dir)
                except FileNotFoundError:
                    pass
                except OSError:
                    LOGGER.warning("遗留批量临时目录清理失败，任务=%s", batch_dir.name, exc_info=True)
                else:
                    LOGGER.info("清理服务重启遗留的批量临时目录，任务=%s", batch_dir.name)

    @staticmethod
    def _cleanup_completed_batch_temporary_files(batch_dir: Path) -> None:
        """归档完成后仅保留 ZIP 和清单；失败项由周期清理继续重试。."""
        temporary_paths = [batch_dir / "cutouts", *batch_dir.glob("item_*"), *batch_dir.glob("*.tmp")]
        for path in temporary_paths:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("批量临时产物清理失败，任务=%s，路径=%s", batch_dir.name, path.name, exc_info=True)

    def cleanup_expired_artifacts(self) -> None:
        """由服务启动和服务器定时任务统一触发批次回收。."""
        self._cleanup_expired_batch_jobs()
        self._cleanup_stale_batch_directories()

    def health(self) -> dict[str, object]:
        """以实际 YOLO 加载与任务/类别校验结果报告模型状态。."""
        config = self.pipeline.config
        try:
            with self.lock:
                self.pipeline.load_model()
        except ModelNotReadyError as error:
            model_error = str(error)
        else:
            model_error = None
        model_ready = self.pipeline.model is not None
        return {
            "status": "model_ready" if model_ready else "waiting_model",
            "model_ready": model_ready,
            "model_error": model_error,
            "model_name": config.model_path.name,
            "target_class_id": config.target_class_id,
            "confidence": config.confidence,
            "device": str(config.device),
        }

    def process(self, payload: dict[str, object]) -> dict[str, object]:
        """保存上传图片，并返回可预览、可下载的结果地址。."""
        case_id = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        case_dir = RUNS_DIR / case_id
        try:
            image_path = _save_upload(payload.get("image"), case_dir / "input")
            LOGGER.info("开始整车轮廓提取，任务=%s，图片=%s", case_id, image_path.name)
            with self.lock:
                result = self.pipeline.cutout(image_path, case_dir)
        except Exception:
            shutil.rmtree(case_dir, ignore_errors=True)
            raise
        try:
            shutil.rmtree(case_dir / "input")
        except OSError:
            LOGGER.warning("单张任务原图清理失败，任务=%s", case_id, exc_info=True)
        LOGGER.info(
            "完成整车轮廓提取，任务=%s，置信度=%.4f，轮廓占框比=%.4f",
            case_id,
            result.selection.confidence,
            result.mask_fill_ratio,
        )
        prefix = f"/runs/{case_id}"
        return {
            "case_id": case_id,
            "detection": {
                "class_id": result.selection.class_id,
                "class_name": result.selection.class_name,
                "confidence": result.selection.confidence,
                "box": list(result.selection.box),
            },
            "mask_fill_ratio": result.mask_fill_ratio,
            "cutout_size": list(result.cutout_size),
            "device": str(self.pipeline.config.device),
            "artifacts": {
                "detection": f"{prefix}/{result.detection_preview_path.name}",
                "mask": f"{prefix}/{result.mask_preview_path.name}",
                "cutout": f"{prefix}/{result.cutout_path.name}",
            },
        }

    def start_batch(self, payload: dict[str, object]) -> dict[str, object]:
        """登记目录内的完整相对路径清单，后续每张图片使用独立请求处理。."""
        raw_paths = payload.get("relative_paths")
        if not isinstance(raw_paths, list):
            raise TypeError("批量任务必须提供图片相对路径清单")
        if not raw_paths:
            raise ValueError("所选目录中没有可处理的图片")
        if len(raw_paths) > MAX_BATCH_FILES:
            raise ValueError(f"单次批量最多处理 {MAX_BATCH_FILES} 张图片")
        relative_paths = tuple(self._resolve_batch_relative_path(path).as_posix() for path in raw_paths)
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("批量图片相对路径不能重复")
        with self.lock:
            self.pipeline.load_model()

        batch_id = "batch_" + datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        batch_dir = RUNS_DIR / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        with self.batch_lock:
            self.batch_jobs[batch_id] = BatchJob(batch_id, batch_dir, relative_paths)
        LOGGER.info("创建批量抠图任务，任务=%s，图片数=%d", batch_id, len(relative_paths))
        return {"batch_id": batch_id, "file_count": len(relative_paths)}

    @staticmethod
    def _resolve_batch_relative_path(raw_path: object) -> Path:
        """校验浏览器传入的目录相对路径，禁止绝对路径和父目录跳转。."""
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("批量图片相对路径不能为空")
        relative = PurePosixPath(raw_path.strip().replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"批量图片相对路径无效：{raw_path}")
        if not relative.parts or relative.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的批量图片格式：{relative.suffix or '无扩展名'}")
        return Path(*relative.parts)

    @staticmethod
    def _validate_batch_id(batch_id: object) -> str:
        """校验批次 ID，确保只能定位服务生成的任务目录。."""
        if not isinstance(batch_id, str) or not BATCH_ID_PATTERN.fullmatch(batch_id):
            raise ValueError("批量任务 ID 无效")
        return batch_id

    def _get_batch_job(self, batch_id: object) -> BatchJob:
        """从短生命周期注册表读取批次，完成归档后不再保留内存状态。."""
        batch_id = self._validate_batch_id(batch_id)
        with self.batch_lock:
            job = self.batch_jobs.get(batch_id)
            if job is None:
                raise ValueError("批量任务不存在、已完成或服务已重启")
            return job

    def _write_batch_manifest(self, job: BatchJob) -> Path:
        """每处理一张图即原子更新清单，服务异常时仍可回溯已完成项。."""
        records = [
            {key: value for key, value in job.records[path].items() if not key.startswith("_")}
            for path in job.relative_paths
            if path in job.records
        ]
        success_count = sum(bool(item["success"]) for item in records)
        manifest = {
            "batch_id": job.batch_id,
            "expected_count": len(job.relative_paths),
            "processed_count": len(records),
            "success_count": success_count,
            "failed_count": len(records) - success_count,
            "records": records,
        }
        manifest_path = job.batch_dir / "manifest.json"
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(manifest_path)
        return manifest_path

    def process_batch_item(self, payload: dict[str, object]) -> dict[str, object]:
        """处理批量任务的一张图，任何单图失败都落入清单并允许批次继续。."""
        job = self._get_batch_job(payload.get("batch_id"))
        relative_path = self._resolve_batch_relative_path(payload.get("relative_path"))
        relative_key = relative_path.as_posix()
        with job.lock:
            with self.batch_lock:
                if self.batch_jobs.get(job.batch_id) is not job:
                    raise ValueError("批量任务已完成")
            job.updated_at = time.monotonic()
            if relative_key not in job.relative_paths:
                raise ValueError(f"批量图片未在任务开始时登记：{relative_key}")
            if relative_key in job.records:
                raise ValueError(f"批量图片重复提交：{relative_key}")
            item_number = job.relative_paths.index(relative_key) + 1
            client_error = payload.get("error")
            if client_error is not None:
                if not isinstance(client_error, str) or not client_error.strip():
                    raise TypeError("浏览器失败原因必须是非空文本")
                record = {
                    "relative_path": relative_key,
                    "success": False,
                    "error": f"浏览器未上传图片：{client_error.strip()[:500]}",
                }
            else:
                try:
                    with TemporaryDirectory(prefix=f"item_{item_number:05d}_", dir=job.batch_dir) as temporary_dir:
                        item_dir = Path(temporary_dir)
                        image_path = _save_upload(payload.get("image"), item_dir / "input")
                        with self.lock:
                            result = self.pipeline.cutout_only(image_path, item_dir)
                        cutout_dir = job.batch_dir / "cutouts"
                        cutout_dir.mkdir(parents=True, exist_ok=True)
                        cutout_path = cutout_dir / f"{item_number:05d}.png"
                        result.cutout_path.replace(cutout_path)
                    archive_name = (
                        Path("整车抠图")
                        / relative_path.parent
                        / f"{relative_path.stem}_{relative_path.suffix.lstrip('.').lower()}_cutout.png"
                    )
                    record = {
                        "relative_path": relative_key,
                        "success": True,
                        "confidence": result.selection.confidence,
                        "class_id": result.selection.class_id,
                        "class_name": result.selection.class_name,
                        "archive_path": archive_name.as_posix(),
                        "_cutout_path": str(cutout_path),
                    }
                except ModelNotReadyError:
                    raise
                except (
                    FileNotFoundError,
                    TypeError,
                    ValueError,
                    TargetNotDetectedError,
                    SegmentationError,
                ) as error:
                    record = {"relative_path": relative_key, "success": False, "error": str(error)}
                    LOGGER.info("批量图片未完成抠图，任务=%s，图片=%s，原因=%s", job.batch_id, relative_key, error)
                except Exception as error:
                    record = {
                        "relative_path": relative_key,
                        "success": False,
                        "error": f"处理异常：{type(error).__name__}: {error}",
                    }
                    LOGGER.exception("批量图片处理异常，任务=%s，图片=%s", job.batch_id, relative_key)

            job.records[relative_key] = record
            self._write_batch_manifest(job)
            job.updated_at = time.monotonic()
            return {key: value for key, value in record.items() if not key.startswith("_")}

    def _build_completed_batch_response(self, batch_id: str) -> dict[str, object]:
        """从磁盘返回已完成任务，支持收尾响应丢失后的幂等重试。."""
        batch_dir = RUNS_DIR / batch_id
        manifest_path = batch_dir / "manifest.json"
        archive_path = batch_dir / f"{batch_id}_cutouts.zip"
        if not manifest_path.is_file() or not archive_path.is_file():
            raise ValueError("批量任务不存在或服务已重启")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "batch_id": batch_id,
            "processed_count": manifest["processed_count"],
            "success_count": manifest["success_count"],
            "failed_count": manifest["failed_count"],
            "archive_url": f"/runs/{batch_id}/{archive_path.name}",
        }

    def finish_batch(self, payload: dict[str, object]) -> dict[str, object]:
        """补齐未处理项并原子打包，成功后立即释放批次内存状态。."""
        batch_id = self._validate_batch_id(payload.get("batch_id"))
        with self.batch_lock:
            job = self.batch_jobs.get(batch_id)
        if job is None:
            return self._build_completed_batch_response(batch_id)

        with job.lock:
            with self.batch_lock:
                if self.batch_jobs.get(batch_id) is not job:
                    return self._build_completed_batch_response(batch_id)
            job.updated_at = time.monotonic()
            for relative_path in job.relative_paths:
                if relative_path not in job.records:
                    job.records[relative_path] = {
                        "relative_path": relative_path,
                        "success": False,
                        "error": "未处理：任务在图片上传前停止或请求中断",
                    }
            manifest_path = self._write_batch_manifest(job)
            records = [job.records[path] for path in job.relative_paths]
            archive_path = job.batch_dir / f"{job.batch_id}_cutouts.zip"
            temporary_archive = archive_path.with_suffix(".zip.tmp")
            with zipfile.ZipFile(temporary_archive, "w") as archive:
                archive.write(manifest_path, "manifest.json", compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
                for record in records:
                    if record["success"]:
                        archive.write(record["_cutout_path"], record["archive_path"], compress_type=zipfile.ZIP_STORED)
            temporary_archive.replace(archive_path)
            self._cleanup_completed_batch_temporary_files(job.batch_dir)

            success_count = sum(bool(item["success"]) for item in records)
            with self.batch_lock:
                self.batch_jobs.pop(job.batch_id, None)
        LOGGER.info(
            "完成批量抠图，任务=%s，成功=%d，失败=%d，压缩包=%s",
            job.batch_id,
            success_count,
            len(records) - success_count,
            archive_path,
        )
        return {
            "batch_id": job.batch_id,
            "processed_count": len(records),
            "success_count": success_count,
            "failed_count": len(records) - success_count,
            "archive_url": f"/runs/{job.batch_id}/{archive_path.name}",
        }


class CutoutWebHandler(BaseHTTPRequestHandler):
    """提供单页工具、轮廓提取接口和本次任务产物。."""

    service: CutoutService
    server_version = "EbikeYoloSegCutout/1.0"

    def _send_bytes(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, content, "application/json; charset=utf-8")

    def _send_file(self, status: int, path: Path, content_type: str) -> None:
        """分块发送产物，避免批量 ZIP 按整包大小占用内存。."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                self.wfile.write(chunk)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(HTTPStatus.OK, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self._send_json(HTTPStatus.OK, self.service.health())
            return
        if parsed.path.startswith("/runs/"):
            self._serve_artifact(parsed.path)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "页面不存在"})

    def do_POST(self) -> None:
        request_path = urlparse(self.path).path
        if request_path not in {"/api/cutout", "/api/batch/start", "/api/batch/item", "/api/batch/finish"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
            return
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请求必须使用 application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Content-Length 格式错误"})
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "请求为空或上传图片过大"})
            return

        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise TypeError("请求内容必须是 JSON 对象")
            if request_path == "/api/cutout":
                response = self.service.process(payload)
            elif request_path == "/api/batch/start":
                response = self.service.start_batch(payload)
            elif request_path == "/api/batch/item":
                response = self.service.process_batch_item(payload)
            else:
                response = self.service.finish_batch(payload)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except ModelNotReadyError as error:
            LOGGER.error("模型未就绪：%s", error)
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
            return
        except (TargetNotDetectedError, SegmentationError) as error:
            LOGGER.info("轮廓提取未完成：%s", error)
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
            return
        except (RuntimeError, cv2.error) as error:
            LOGGER.exception("模型推理失败")
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": f"模型推理失败：{error}"})
            return
        except Exception as error:
            LOGGER.exception("抠图服务异常")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"服务异常：{type(error).__name__}: {error}"})
            return
        self._send_json(HTTPStatus.OK, response)

    def _serve_artifact(self, request_path: str) -> None:
        relative = Path(unquote(request_path[len("/runs/") :]))
        target = (RUNS_DIR / relative).resolve()
        try:
            target.relative_to(RUNS_DIR.resolve())
        except ValueError:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "禁止访问该文件"})
            return
        if not target.is_file() or target.suffix.lower() not in ARTIFACT_SUFFIXES:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "结果图片不存在"})
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send_file(HTTPStatus.OK, target, content_type)

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("HTTP 客户端=%s %s", self.address_string(), format % args)


class CutoutWebServer(ThreadingHTTPServer):
    """在 HTTP 服务生命周期内启动并定时执行批次回收。."""

    def __init__(self, server_address: tuple[str, int], service: CutoutService) -> None:
        self.cutout_service = service
        CutoutWebHandler.service = service
        self.cutout_service.cleanup_expired_artifacts()
        self.next_cleanup_at = time.monotonic() + BATCH_CLEANUP_INTERVAL_SECONDS
        super().__init__(server_address, CutoutWebHandler)

    def service_actions(self) -> None:
        """利用 ``serve_forever`` 的周期回调清理闲置任务，无需额外常驻线程。."""
        now = time.monotonic()
        if now < self.next_cleanup_at:
            return
        self.next_cleanup_at = now + BATCH_CLEANUP_INTERVAL_SECONDS
        try:
            self.cutout_service.cleanup_expired_artifacts()
        except Exception:
            LOGGER.exception("批量任务定时清理异常")


def build_service() -> CutoutService:
    """使用代码内固定配置构建服务，调整本方法即可切换权重、阈值和设备。."""
    config = CutoutConfig(
        model_path=MODELS_DIR / "yolo11l-seg.pt",
        target_class_id=3,
        confidence=0.25,
        imgsz=640,
        device=_resolve_device(),
    )
    return CutoutService(YoloSegCutoutPipeline(config))


def main() -> int:
    """使用代码内固定地址启动本地网页服务，不接收命令行参数。."""
    host, port = "127.0.0.1", 8768
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    service = build_service()
    server = CutoutWebServer((host, port), service)
    LOGGER.info("整车轮廓提取服务已启动，地址=http://%s:%d/", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("整车轮廓提取服务已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
