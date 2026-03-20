# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D-Robotics RDK export logic."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.utils import ARM64, LINUX, LOGGER, YAML, colorstr


def bpu_detect_forward(self, x):
    """Return raw detect head branch outputs for RDK export."""
    res = []
    cv3 = self.one2one_cv3 if hasattr(self, "one2one_cv3") else self.cv3
    cv2 = self.one2one_cv2 if hasattr(self, "one2one_cv2") else self.cv2
    for i in range(self.nl):
        res.append(cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        res.append(cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return res


def bpu_classify_forward(self, x):
    """Return classification logits through a Conv2d path for RDK export."""
    if x.ndim == 4:
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        return self.conv_linear(x).flatten(1)
    return self.linear(torch.flatten(self.pool(self.conv(x)), 1))


def apply_rdk_patches(model):
    """Apply export-time model patches required by the RDK toolchain."""
    from ultralytics.nn.modules import Classify, Detect, OBB, Pose, Segment, v10Detect

    if getattr(model, "task", None) != "detect":
        raise NotImplementedError("Initial RDK export support is currently limited to detection models.")

    for module in model.modules():
        if isinstance(module, Detect) and not isinstance(module, (Segment, Pose, OBB)):
            module.forward = bpu_detect_forward.__get__(module, Detect)
            LOGGER.info(f"{colorstr('RDK:')} patched {type(module).__name__} head for export.")
        elif isinstance(module, v10Detect):
            module.forward = bpu_detect_forward.__get__(module, v10Detect)
            LOGGER.info(f"{colorstr('RDK:')} patched {type(module).__name__} head for export.")
        elif isinstance(module, Classify):
            in_features = module.linear.in_features
            out_features = module.linear.out_features
            module.conv_linear = torch.nn.Conv2d(in_features, out_features, 1)
            module.conv_linear.weight.data = module.linear.weight.data.view(out_features, in_features, 1, 1)
            module.conv_linear.bias.data = module.linear.bias.data
            module.forward = bpu_classify_forward.__get__(module, Classify)


def _prepare_calibration_data(data, cal_data_dir: Path, imgsz: tuple[int, int]) -> None:
    """Generate calibration tensors from the training split of a detection dataset."""
    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(data)
    train_path = dataset.get("train", "")
    if not train_path:
        raise ValueError(f"No 'train' split found in {data}.")

    if isinstance(train_path, list):
        train_path = train_path[0]

    img_dir = Path(train_path)
    if img_dir.is_file() and img_dir.suffix == ".txt":
        img_paths = [Path(x.strip()) for x in img_dir.read_text().splitlines() if x.strip()]
    else:
        img_paths = list(img_dir.rglob("*.*"))

    img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not img_paths:
        raise ValueError(f"No images found for calibration in {train_path}.")

    cal_data_dir.mkdir(parents=True, exist_ok=True)
    width, height = imgsz[1], imgsz[0]
    sample_num = min(20, len(img_paths))
    for idx, img_path in enumerate(random.sample(img_paths, sample_num)):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)
        image.tofile(cal_data_dir / f"cal_{idx}.rgbchw")


def _check_rdk_export_requirements(prefix: str) -> None:
    """Validate the host environment required for RDK export."""
    if not LINUX or ARM64:
        raise RuntimeError(f"{prefix} export is only supported on x86_64 Linux hosts.")
    if shutil.which("hb_mapper") is None:
        raise FileNotFoundError(f"{prefix} required tool 'hb_mapper' was not found in PATH.")


def export_rdk(model, args, onnx_path: str | Path, metadata: dict, prefix: str = colorstr("RDK:")):
    """Export an Ultralytics detection model to an RDK deployment package."""
    _check_rdk_export_requirements(prefix)
    if not args.data:
        raise ValueError(f"{prefix} export requires a detection dataset via `data=...` for calibration.")

    imgsz = args.imgsz if isinstance(args.imgsz, (tuple, list)) else (args.imgsz, args.imgsz)
    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"{prefix} intermediate ONNX file not found at {onnx_path}.")

    save_dir = onnx_path.parent
    workspace = save_dir / ".rdk_export"
    cal_data_dir = workspace / "calibration_data"
    compiler_dir = workspace / "compiler_output"
    model_dir = save_dir / f"{onnx_path.stem}_rdk_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if workspace.exists():
        shutil.rmtree(workspace)
    compiler_dir.mkdir(parents=True, exist_ok=True)

    _prepare_calibration_data(args.data, cal_data_dir, tuple(imgsz))

    output_prefix = f"{onnx_path.stem}_bayese_{imgsz[1]}x{imgsz[0]}_nv12"
    yaml_path = workspace / "hb_mapper_config.yaml"
    yaml_path.write_text(
        f"""model_parameters:
  onnx_model: '{onnx_path}'
  march: "bayes-e"
  layer_out_dump: False
  working_dir: '{compiler_dir}'
  output_model_file_prefix: '{output_prefix}'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
calibration_parameters:
  cal_data_dir: '{cal_data_dir}'
  cal_data_type: 'float32'
  calibration_type: 'default'
compiler_parameters:
  jobs: 16
  compile_mode: 'latency'
  debug: true
  optimize_level: 'O3'
""",
        encoding="utf-8",
    )

    LOGGER.info(f"{prefix} compiling ONNX with hb_mapper...")
    subprocess.run(
        ["hb_mapper", "makertbin", "--config", str(yaml_path.name), "--model-type", "onnx"],
        check=True,
        cwd=workspace,
    )

    compiled_bin = next(compiler_dir.rglob("*.bin"), None)
    if compiled_bin is None:
        raise FileNotFoundError(f"{prefix} compilation completed but no .bin artifact was produced.")

    shutil.copy2(compiled_bin, model_dir / f"{onnx_path.stem}.bin")
    YAML.save(model_dir / "metadata.yaml", metadata)
    return model_dir
