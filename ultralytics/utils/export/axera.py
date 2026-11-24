# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_version

AXERA_CHIPS = frozenset({"AX650", "AX8860"})
AXERA_NPU_MODES = {"AX650": "NPU3", "AX8860": "NPU4"}  # multi-core default per chip


def _calibration_images(dataset: Any) -> list[Path]:
    """Return original image paths from a calibration dataloader's dataset."""
    source = getattr(dataset, "dataset", dataset)
    image_paths = getattr(source, "im_files", None)
    if image_paths is None and hasattr(source, "samples"):
        image_paths = [x[0] for x in source.samples]
    if not image_paths:
        raise ValueError("AXERA export requires a calibration dataset with image file paths.")
    return [Path(x) for x in image_paths]


def _stage_calibration(images: list[Path], calib_dir: Path, calib_zip: Path) -> int:
    """Copy calibration images under space-free names and zip them for Pulsar2 Image mode."""
    shutil.rmtree(calib_dir, ignore_errors=True)
    calib_dir.mkdir(parents=True)
    for i, image in enumerate(images):
        suffix = image.suffix.lower() or ".jpg"
        shutil.copy2(image, calib_dir / f"{i}{suffix}")

    calib_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(calib_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for image in sorted(calib_dir.iterdir()):
            if image.is_file():
                z.write(image, image.name)
    return len(images)


def _graph_output_names(onnx_file: Path) -> list[str]:
    """Return the ONNX graph output names, in graph order."""
    import onnx

    return [output.name for output in onnx.load(str(onnx_file), load_external_data=False).graph.output]


def _split_head(onnx_file: Path, output_dir: Path, head: str) -> Path | None:
    """Split the graph at the head convolutions so the decode runs in FP32 ``tail.onnx``.

    Fixed-point activations cannot hold both box pixels (~0-640) and class scores (~0-1) in one scale, so the quantized
    body stops at the per-level head convs and ``tail.onnx`` keeps the decode for the host. Applies to detect, segment,
    pose and OBB builds at INT8 (``8``) or INT8 weights (``"w8a16"``); extra graph outputs such as segment prototypes
    pass straight through the tail. Pure 16-bit float graphs keep the whole graph.
    """
    import onnx

    graph = onnx.load(str(onnx_file), load_external_data=False).graph
    ops = {"Reshape", "Concat"}  # head convs feed only these per-level ops
    ends = {x for n in graph.node if n.op_type in ops for x in n.input}
    ends -= {x for n in graph.node if n.op_type not in ops for x in n.input}
    cuts = [
        n.output[0]
        for n in graph.node
        if n.op_type == "Conv" and head in n.name and "/dfl/" not in n.name and n.output[0] in ends
    ]
    if not cuts:
        return None
    outputs = [x.name for x in graph.output]
    cuts += [x for x in outputs if x != "output0"]  # e.g. segment prototypes pass straight through the tail
    onnx.utils.extract_model(str(onnx_file), str(output_dir / "tail.onnx"), cuts, outputs)
    body = output_dir / "body.onnx"
    onnx.utils.extract_model(str(onnx_file), str(body), [x.name for x in graph.input], cuts)
    return body


def _build_config(
    onnx_file: Path,
    build_dir: Path,
    calib_zip: Path,
    calibration_size: int,
    target: str,
    quantize: int | str,
) -> dict:
    """Build the Pulsar2 configuration for the requested ``quantize`` precision."""
    if quantize == 16:
        # 16-bit float: axnn 8.0 rejects `data_type: FP16` (absent from the quantizer dtype map and only BF16 is
        # accepted), so 16-bit float uses BF16 for both weights and activations.
        layer_configs = [
            {
                "start_tensor_names": ["DEFAULT"],
                "end_tensor_names": ["DEFAULT"],
                "data_type": "BF16",
            }
        ]
    elif quantize == "w8a16":
        # INT8 weights + INT16 activations. Weights default to S8, so only the activation type is set.
        layer_configs = [
            {
                "start_tensor_names": ["DEFAULT"],
                "end_tensor_names": ["DEFAULT"],
                "data_type": "U16",
            }
        ]
    else:  # 8: leave layer precision to the Pulsar2 default, i.e. U8 activations and S8 weights
        layer_configs = []
    return {
        "input": str(onnx_file),
        "output_dir": str(build_dir),
        "output_name": "compiled.axmodel",
        "model_type": "ONNX",
        "target_hardware": target,
        "npu_mode": AXERA_NPU_MODES[target],
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "DEFAULT",
                    "calibration_dataset": str(calib_zip),
                    "calibration_format": "Image",
                    "calibration_size": calibration_size,
                    "calibration_mean": [0, 0, 0],
                    "calibration_std": [255, 255, 255],
                }
            ],
            "layer_configs": layer_configs,
            "calibration_method": "MinMax",
        },
        "input_processors": [
            {
                "tensor_name": "DEFAULT",
                "tensor_format": "RGB",
                "tensor_layout": "NCHW",
                "src_format": "RGB",
                "src_layout": "NHWC",
                "src_dtype": "U8",
                "mean": [0, 0, 0],
                "std": [255, 255, 255],
            }
        ],
        "output_processors": [{"tensor_name": "DEFAULT", "output_dtype": "FP32"}],
    }


def _api_build(config: dict, prefix: str) -> bool:
    """Compile with the in-process axnn build API, returning False when the API is unavailable."""
    try:
        import axnn.yamain.config.build_config_pb2 as build_pb
        from axnn.yamain.command.build import build
        from google.protobuf.json_format import ParseDict
    except (ImportError, ModuleNotFoundError) as e:
        LOGGER.warning(f"{prefix} axnn build API unavailable: {e}")
        return False

    cfg = build_pb.BuildConfig()
    ParseDict(config, cfg)
    LOGGER.info(f"{prefix} starting export with axnn build API...")
    build(cfg)
    return True


def _cli_build(config_file: Path, prefix: str) -> None:
    """Compile with the pulsar2 CLI as a fallback when the API is unavailable."""
    executable = shutil.which("pulsar2")
    if not executable:
        raise FileNotFoundError("AXERA export requires the axnn build API or the pulsar2 CLI.")
    LOGGER.warning(f"{prefix} axnn build API unavailable, falling back to pulsar2 build --config.")
    subprocess.run([executable, "build", "--config", str(config_file)], check=True)


def _check_build(build_dir: Path, prefix: str) -> dict:
    """Validate Pulsar2 output and return its build context."""
    build_context = build_dir / "build_context.json"
    if not build_context.is_file():
        raise RuntimeError(f"{prefix} Pulsar2 did not create build_context.json.")
    context = json.loads(build_context.read_text())
    if context.get("code") != "Success":
        raise RuntimeError(f"{prefix} Pulsar2 build failed: {context.get('error_msg', context.get('code'))}")
    if not (build_dir / "compiled.axmodel").is_file():
        raise RuntimeError(f"{prefix} Pulsar2 did not create compiled.axmodel.")
    return context


def onnx2axera(
    onnx_file: str | Path,
    output_dir: str | Path,
    dataset: Any,
    metadata: dict | None = None,
    target: str = "AX650",
    quantize: int | str = 8,
    head: str | None = None,
    prefix: str = "",
) -> str:
    """Compile an ONNX model to an AXERA ``.axmodel`` directory with Pulsar2.

    Args:
        onnx_file (str | Path): Source ONNX model path.
        output_dir (str | Path): Directory for the final ``_axera_model`` artifact.
        dataset (Any): Calibration dataloader from ``Exporter.get_int8_calibration_dataloader``.
        metadata (dict | None): Ultralytics metadata saved beside the compiled model.
        target (str): Public chip name: ``"AX650"`` (INT8) or ``"AX8860"`` (16-bit float).
        quantize (int | str): ``8`` (INT8), ``16`` (16-bit float) or ``"w8a16"`` (INT8 weights + INT16 activations).
            ``Exporter`` fills in the per-chip default before calling this function.
        head (str | None): ONNX node-name scope of the model head, e.g. ``"/model.23/"``. Quantized detect, segment,
            pose and OBB builds stop at the head convs and keep the decode in ``tail.onnx`` for the host; semantic
            segmentation, depth, classification and 16-bit builds keep the whole graph.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_axera_model`` directory.
    """
    onnx_file = Path(onnx_file).resolve()
    output_dir = Path(output_dir).resolve()
    target = str(target).upper()
    if target not in AXERA_CHIPS:
        raise ValueError(f"Invalid AXERA target '{target}'. Valid targets are {sorted(AXERA_CHIPS)}.")
    if quantize not in {8, 16, "w8a16"}:
        raise ValueError(f"AXERA export requires quantize in {{8, 16, 'w8a16'}}; got {quantize!r}.")
    try:
        check_version("axnn", ">=8.0", name="axnn", hard=True)
    except ModuleNotFoundError as e:  # missing package or an older axnn
        raise ModuleNotFoundError(
            f"{str(e).strip()} (install the latest release from https://huggingface.co/AXERA-TECH/Pulsar2/tree/main)"
        ) from e

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    target_onnx = output_dir / "model.onnx"
    if onnx_file != target_onnx:
        shutil.move(str(onnx_file), target_onnx)
    onnx_file = target_onnx
    build_dir = output_dir

    images = _calibration_images(dataset)
    calib_dir = build_dir / "calib"
    calib_zip = build_dir / "calib.zip"
    calibration_size = _stage_calibration(images, calib_dir, calib_zip)

    body = _split_head(onnx_file, build_dir, head) if (quantize in {8, "w8a16"} and head) else None
    compiled_onnx = body or onnx_file
    output_names = _graph_output_names(compiled_onnx)

    config = _build_config(compiled_onnx, build_dir, calib_zip, calibration_size, target, quantize)
    config_file = output_dir / "config.json"
    config_file.write_text(json.dumps(config, indent=2) + "\n")

    if not _api_build(config, prefix):
        _cli_build(config_file, prefix)

    context = _check_build(build_dir, prefix)

    if metadata is not None:
        meta = dict(metadata)
        meta["dynamic"] = False
        meta["axera"] = {
            "target": target,
            "inputs": [{"name": "images", "dtype": "uint8", "layout": "NHWC", "range": [0, 255]}],
            "outputs": [{"name": n, "dtype": "float32", "order": i} for i, n in enumerate(output_names)],
            "quantize": quantize,
            "npu_mode": context.get("npu_mode_str", ""),
            "calibration": {"format": "Image", "size": calibration_size},
            "build_context": context.get("code", ""),
        }
        YAML.save(output_dir / "metadata.yaml", meta)

    return str(output_dir)
