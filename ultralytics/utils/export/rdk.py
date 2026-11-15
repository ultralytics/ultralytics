# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D-Robotics RDK X5 BPU export: compile a standard ONNX graph's neural body and ship its decode tail."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ultralytics.utils import ARM64, LINUX, LOGGER, YAML, colorstr

TOOLCHAIN_DOC = "https://developer.d-robotics.cc/rdk_x5"
_COMPILE_TIMEOUT = 3600  # seconds; a 640px O3 INT8 compile of a nano model takes minutes
_DESCRIBE_TIMEOUT = 600  # seconds for one rdkx5-sim --describe call


def _apply_names(model, mapping):
    """Rename graph tensors in place: every node input/output, initializer and graph I/O entry."""
    for node in model.graph.node:
        node.input[:] = [mapping.get(x, x) for x in node.input]
        node.output[:] = [mapping.get(x, x) for x in node.output]
    for group in (model.graph.initializer, model.graph.input, model.graph.output, model.graph.value_info):
        for entry in group:
            if entry.name in mapping:
                entry.name = mapping[entry.name]
    return model


def split_rdk_graph(onnx_file, head, body_file, tail_file):
    """Split a standard ONNX export into its RDK X5 BPU body and its host decode ``tail.onnx``.

    The cut follows ``onnx2rknn``: the head's terminal ``Conv`` outputs plus any further graph outputs (e.g. segment
    prototypes) end the BPU body, except the classifier which ends at its ``Gemm`` logits so the fully-connected layer
    stays on the BPU; everything after the cut is the decode tail. Boundary names containing "/" are renamed to
    ``outN`` in both graphs: the X5 runtime rejects "/" in output names.

    Args:
        onnx_file (str | Path): Standard ONNX export from `Exporter.export_onnx`.
        head (str | None): ONNX node-name scope of the model head, e.g. ``"/model.23/"``.
        body_file (str | Path): Destination for the compiled BPU body graph.
        tail_file (str | Path): Destination for the host decode tail graph.

    Returns:
        (list[str]): Boundary tensor names, shared by the body outputs and the tail inputs.

    Raises:
        ValueError: If the model type has no supported RDK export head or the graph carries no head
            `Conv`/`Gemm` boundary.
    """
    import onnx

    if not head:
        raise ValueError(
            "D-Robotics RDK export found no supported YOLO head in this model type (RT-DETR and foreign models "
            "have none); standard YOLO detect, segment, classify, pose and obb graphs are supported."
        )
    graph = onnx.load(str(onnx_file)).graph
    ends = {x for n in graph.node if n.op_type in {"Reshape", "Concat"} for x in n.input}
    ends -= {x for n in graph.node if n.op_type not in {"Reshape", "Concat"} for x in n.input}
    boundary = [  # head convs feed only these per-level ops, e.g. Concat in WorldDetect
        n.output[0]
        for n in graph.node
        if n.op_type == "Conv" and head in n.name and "/dfl/" not in n.name and n.output[0] in ends
    ]
    boundary += [n.output[0] for n in graph.node if n.op_type == "Gemm" and head in n.name]  # classifier logits
    if not boundary:
        raise ValueError(
            f"D-Robotics RDK export found no head Conv/Gemm boundary under '{head}'; standard YOLO detect, "
            "segment, classify, pose and obb graphs are supported."
        )
    boundary += [o.name for o in graph.output if o.name != "output0"]  # e.g. segment prototypes
    onnx.utils.extract_model(str(onnx_file), str(body_file), [i.name for i in graph.input], boundary)
    onnx.utils.extract_model(str(onnx_file), str(tail_file), boundary, [o.name for o in graph.output])
    mapping = {name: f"out{i}" for i, name in enumerate(boundary) if "/" in name}
    if mapping:
        for file in (body_file, tail_file):  # one flat name on both sides of the boundary
            onnx.save(_apply_names(onnx.load(str(file)), mapping), str(file))
    return [mapping.get(name, name) for name in boundary]


def _write_calibration_files(dataset, cal_dir, prefix=""):
    """Write calibration images as the X5 toolchain's expected RGB float32 NCHW files.

    The shared calibration loader applies the deployment geometry (detector-like LetterBox, classification resize +
    center crop) and emits 0-255 RGB; the compiled model itself applies the configured ``data_scale`` 1/255.
    """
    cal_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for batch in dataset:  # one epoch of the calibration loader; no sampling or count cap
        for image in batch["img"]:
            image.numpy()[None].astype(np.float32).tofile(cal_dir / f"calibration_{count:05d}.rgbchw")
            count += 1
    if not count:
        raise ValueError(f"{prefix} INT8 calibration produced no images from 'data'.")
    LOGGER.info(f"{prefix} wrote {count} calibration images (RGB float32, 0-255) to {cal_dir}")


def _describe_compiled(bin_file, prefix=""):
    """Describe a compiled .bin through Mapper's `rdkx5-sim` CLI, returning its inputs/outputs JSON or None."""
    if shutil.which("rdkx5-sim") is None:
        LOGGER.info(f"{prefix} rdkx5-sim not found on PATH; skipping compiled-model verification.")
        return None
    result = subprocess.run(
        ["rdkx5-sim", str(bin_file), "--describe"],
        capture_output=True,
        text=True,
        timeout=_DESCRIBE_TIMEOUT,
        check=False,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},  # child resolves its own SDK
    )
    if result.returncode != 0:
        raise RuntimeError(f"{prefix} rdkx5-sim --describe failed ({result.returncode}): {result.stderr.strip()}")
    return {k: v for k, v in json.loads(result.stdout).items() if k in {"inputs", "outputs"}}


def _verify_compiled(spec, expected, prefix=""):
    """Fail on any compiled-model I/O mismatch with the exported body/tail boundary."""
    inputs = spec["inputs"]
    if len(inputs) != 1 or inputs[0].get("pixel_format") != "nv12":
        raise ValueError(
            f"{prefix} compiled .bin exposes {len(inputs)} input(s) with formats "
            f"{[i.get('pixel_format') for i in inputs]}; expected the single compiled NV12 input."
        )
    names = [o["name"] for o in spec["outputs"]]
    if names != list(expected):
        raise ValueError(
            f"{prefix} compiled .bin outputs {names} do not match the exported boundary {list(expected)}; "
            "refusing to guess the binding. Re-export or file an issue with the logs."
        )
    for output in spec["outputs"]:
        if output.get("dtype") != "float32" or output.get("quanti_type") != "NONE":
            raise ValueError(
                f"{prefix} output '{output['name']}' deploys as "
                f"{output.get('dtype')}/{output.get('quanti_type')}; only "
                "dequantized float32 outputs feed the tail — re-export or use the toolchain runtime."
            )
        shape = [int(d) for d in output["shape"]]
        while len(shape) > len(expected[output["name"]]) and shape[-1] == 1:  # the BPU appends trailing unit dims
            shape.pop()
        if shape != expected[output["name"]]:
            raise ValueError(
                f"{prefix} output '{output['name']}' deploys as {output['shape']}, the exported boundary records "
                f"{expected[output['name']]}."
            )


def onnx2rdk(onnx_file, output_dir, dataset, metadata, head, prefix=""):
    """Compile a standard ONNX export's BPU body to an INT8 `.bin` with `hb_mapper` and publish its deployment.

    Args:
        onnx_file (str | Path): Standard ONNX export from `Exporter.export_onnx` (opset 11).
        output_dir (str | Path): Final `{model}_rdk_model` directory receiving the deployment.
        dataset (InfiniteDataLoader): INT8 calibration loader from `Exporter.get_int8_calibration_dataloader`.
        metadata (dict): Ultralytics export metadata; a `format: rdk` marker is added for `.bin` routing.
        head (str | None): ONNX node-name scope of the model head, e.g. ``"/model.23/"``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the final deployment directory.
    """
    import onnx

    assert LINUX and not ARM64, "RDK export is only supported on x86_64 Linux with the hb_mapper toolchain."
    prefix = prefix or colorstr("D-Robotics:")
    onnx_file, output_dir = Path(onnx_file), Path(output_dir)
    if shutil.which("hb_mapper") is None:
        raise RuntimeError(
            f"{prefix} 'hb_mapper' not found on PATH. Install the D-Robotics OpenExplore X5 toolchain (Python 3.10) "
            "and put its bin directory on PATH, e.g. 'PATH=/opt/explore/3.x.x/x86_64/gcc/bin:$PATH'. "
            f"See {TOOLCHAIN_DOC}. Dependencies are never installed automatically."
        )

    # The workspace lives beside output_dir so the final os.replace stays on one filesystem.
    stem = onnx_file.stem
    h, w = metadata["imgsz"]
    model_prefix = f"{stem}_bayese_{h}x{w}_nv12"
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".rdk_work_") as workspace:
        workspace = Path(workspace).resolve()  # absolute: hb_mapper resolves --config against its own cwd
        body_file, tail_file = workspace / "body.onnx", workspace / "tail.onnx"
        boundary = split_rdk_graph(onnx_file, head, body_file, tail_file)
        # the tail's own inputs are the runtime contract for the compiled outputs
        expected = {
            i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in onnx.load(str(tail_file)).graph.input
        }
        cal_dir = workspace / "calibration_data"
        bpu_dir = workspace / "bpu_model_output"
        bpu_dir.mkdir(parents=True)
        _write_calibration_files(dataset, cal_dir, prefix)
        config = workspace / "hb_mapper_config.yaml"
        YAML.save(  # safe YAML quoting for paths with spaces or apostrophes
            config,
            {
                "model_parameters": {
                    "onnx_model": str(body_file.resolve()),
                    "march": "bayes-e",
                    "layer_out_dump": False,
                    "working_dir": str(bpu_dir.resolve()),
                    "output_model_file_prefix": model_prefix,
                },
                "input_parameters": {
                    "input_name": "",
                    "input_type_rt": "nv12",
                    "input_type_train": "rgb",
                    "input_layout_train": "NCHW",
                    "norm_type": "data_scale",
                    "scale_value": 0.003921568627451,
                },
                "calibration_parameters": {
                    "cal_data_dir": str(cal_dir.resolve()),
                    "cal_data_type": "float32",
                    "calibration_type": "default",
                    "optimization": "set_Softmax_input_int8,set_Softmax_output_int8",
                },
                "compiler_parameters": {
                    "jobs": 16,
                    "compile_mode": "latency",
                    "debug": True,
                    "optimize_level": "O3",
                },
            },
        )
        LOGGER.info(f"{prefix} compiling {len(boundary)}-output body with hb_mapper makertbin (takes minutes)...")
        compile_log = workspace / "hb_mapper_makertbin.log"
        failure_log = output_dir.parent / f"{stem}_hb_mapper_makertbin_failure.log"
        try:
            result = subprocess.run(
                ["hb_mapper", "makertbin", "--config", str(config), "--model-type", "onnx"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=_COMPILE_TIMEOUT,
                check=False,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},  # child resolves its own SDK
            )
        except subprocess.TimeoutExpired:
            failure_log.write_text("hb_mapper timed out", encoding="utf-8")
            raise RuntimeError(
                f"{prefix} hb_mapper timed out after {_COMPILE_TIMEOUT}s; log retained at {failure_log}."
            )
        compile_log.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            shutil.copy2(compile_log, failure_log)  # keep the toolchain output: the workspace is temporary
            tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
            raise RuntimeError(
                f"{prefix} hb_mapper failed with exit code {result.returncode}:\n{tail}\n"
                f"Full log retained at {failure_log}."
            )

        compiled_bin = bpu_dir / f"{model_prefix}.bin"
        if not compiled_bin.is_file():
            shutil.copy2(compile_log, failure_log)
            raise FileNotFoundError(
                f"{prefix} hb_mapper finished but '{compiled_bin.name}' was not produced; log retained at "
                f"{failure_log}."
            )

        # verify the exact bytes that will be published
        spec = _describe_compiled(compiled_bin, prefix)
        if spec is not None:
            _verify_compiled(spec, expected, prefix)
        staging = workspace / "staged_deployment"
        staging.mkdir()
        shutil.copy2(compiled_bin, staging / f"{stem}.bin")
        shutil.copy2(tail_file, staging / "tail.onnx")
        quant_info = bpu_dir / f"{model_prefix}_quant_info.json"
        if quant_info.is_file():
            shutil.copy2(quant_info, staging)
        shutil.copy2(compile_log, staging)
        metadata["format"] = "rdk"  # routes an ambiguous .bin through the RDK backend
        YAML.save(staging / "metadata.yaml", metadata)

        # staged publish: move any previous deployment aside, swap in the new one, and
        # roll back to the old bytes if the swap itself fails
        backup = None
        if output_dir.exists():
            backup = workspace / "previous_deployment"
            shutil.move(str(output_dir), str(backup))
        try:
            os.replace(staging, output_dir)
        except OSError:
            if backup is not None:
                shutil.move(str(backup), str(output_dir))
            raise
        return str(output_dir)
