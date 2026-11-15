# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D-Robotics RDK X5 inference backend for deployments exported with `format="rdk"`."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.utils import ARM64, LINUX, LOGGER, YAML
from ultralytics.utils.checks import ONNXRUNTIME_BUILDS, check_requirements

from .base import BaseBackend

TOOLCHAIN_DOC = "https://developer.d-robotics.cc/rdk_doc_center"
_RE_EXPORT = (
    "Re-export with YOLO(model).export(format='rdk') so the deployment carries tail.onnx and a 'format: rdk' marker."
)
_DESCRIBE_TIMEOUT = 600  # seconds for one rdkx5-sim --describe call
_RUN_TIMEOUT = 600  # seconds for one rdkx5-sim run call (the native timeout stays the CLI's own)


def _check(condition, message):
    """Raise a ValueError with an actionable RDK message when the condition fails."""
    if not condition:
        raise ValueError(f"D-Robotics RDK: {message}")


def _normalized_shape(shape, expected):
    """Return `shape` matching `expected` with only BPU-appended trailing unit dims removed, else None."""
    shape = [int(d) for d in shape]
    while len(shape) > len(expected) and shape[-1] == 1:
        shape.pop()
    return shape if shape == [int(d) for d in expected] else None


def _rgb_to_nv12(image):
    """Convert one RGB uint8 (H, W, 3) image to the packed NV12 layout the compiled X5 model consumes."""
    h, w = image.shape[:2]  # even dims required by I420, guaranteed by the load-time NV12 input checks
    planar = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(-1)
    y, u, v = planar[: h * w], planar[h * w : h * w * 5 // 4], planar[h * w * 5 // 4 :]
    nv12 = np.concatenate((y, np.stack((u, v), -1).reshape(-1)))
    return np.ascontiguousarray(nv12, dtype=np.uint8).reshape(h * 3 // 2, w)


class RDKBackend(BaseBackend):
    """D-Robotics RDK X5 inference backend for `.bin` models compiled by `hb_mapper`."""

    def load_model(self, weight: str | Path) -> None:
        """Load an Ultralytics RDK X5 deployment directory or its compiled .bin file.

        Args:
            weight (str | Path): `{stem}_rdk_model/` directory, or the `.bin` inside one.

        Raises:
            ValueError: If the deployment is ambiguous, not an RDK export, or its runtime I/O disagrees with the tail.
            FileNotFoundError: If the compiled model file does not exist.
            RuntimeError: If the platform or runtime is unavailable, or simulator initialization fails.
        """
        w = Path(weight).resolve()
        if w.is_dir():
            bins = sorted(w.glob("*.bin"))  # top level only; never a recursive first-match
            _check(
                len(bins) == 1,
                f"'{w}' contains {len(bins)} top-level .bin files {sorted(p.name for p in bins)}; point at the exact "
                f"'{{stem}}_rdk_model' export directory or .bin file.",
            )
            w = bins[0]
        if not w.is_file():
            raise FileNotFoundError(f"D-Robotics RDK model file not found: '{w}'.")
        self._bin = w
        sidecar = w.parent / "metadata.yaml"
        _check(sidecar.is_file(), f"'{w}' has no metadata.yaml; a bare .bin is not routable.")
        try:
            metadata = YAML.load(sidecar)
        except ValueError as error:  # a valid-YAML non-mapping sidecar gets the actionable RDK error at load
            raise ValueError(f"D-Robotics RDK: {error}. {_RE_EXPORT}") from None
        _check(
            isinstance(metadata, dict) and metadata.get("format") == "rdk",
            f"'{sidecar}' carries no 'format: rdk' marker. {_RE_EXPORT}",
        )
        self.apply_metadata(metadata)
        tail_file = w.parent / "tail.onnx"
        _check(tail_file.is_file(), f"'{w}' has no tail.onnx decode graph. {_RE_EXPORT}")
        check_requirements([ONNXRUNTIME_BUILDS])
        import onnxruntime as ort

        self._tail = ort.InferenceSession(str(tail_file), providers=["CPUExecutionProvider"])
        # the tail's own inputs name and shape the compiled outputs this backend must feed it
        self._inputs = {i.name: list(i.shape) for i in self._tail.get_inputs()}
        self.dynamic = False  # static deployments; AutoBackend serializes larger batches
        if LINUX and not ARM64:
            self._load_simulator()
        elif LINUX and ARM64:
            self._load_board_runtime()
        else:
            raise RuntimeError(
                f"D-Robotics RDK X5 inference runs on Linux x86_64 (rdkx5-sim) or RDK boards (hbm_runtime); this "
                f"platform is unsupported. See {TOOLCHAIN_DOC}."
            )
        LOGGER.info(f"Loaded {w.name} for D-Robotics RDK X5 {self.task} inference.")

    def _validate_outputs(self, outputs, source):
        """Check actual runtime outputs (describe-schema entries) against the tail graph inputs."""
        names = [o["name"] for o in outputs]
        _check(
            names == list(self._inputs),
            f"{source} outputs {names} do not match the tail inputs {list(self._inputs)}. {_RE_EXPORT}",
        )
        for o in outputs:
            _check(
                o.get("dtype") == "float32" and o.get("quanti_type") == "NONE",
                f"{source} output '{o['name']}' deploys as {o.get('dtype')}/{o.get('quanti_type')}; only dequantized "
                f"float32 outputs feed the tail.",
            )
            _check(
                _normalized_shape(o["shape"], self._inputs[o["name"]]) is not None,
                f"{source} output '{o['name']}' deploys as {o['shape']}, the tail input expects "
                f"{self._inputs[o['name']]}.",
            )

    def _load_simulator(self):
        """Resolve the x86 simulator CLI and validate the model's actual I/O through it."""
        sim = shutil.which("rdkx5-sim")
        if sim is None:
            raise RuntimeError(
                "D-Robotics RDK: 'rdkx5-sim' not found on PATH. Use an rdkx5-yolo-mapper build with simulation "
                "support (Python 3.10) and put its bin directory on PATH, or run on an RDK X5 board. "
                f"See {TOOLCHAIN_DOC}. "
                "Dependencies are never installed automatically."
            )
        self._sim = sim
        try:
            result = subprocess.run(
                [sim, str(self._bin), "--describe"],
                capture_output=True,
                text=True,
                timeout=_DESCRIBE_TIMEOUT,
                check=False,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"D-Robotics RDK: rdkx5-sim --describe timed out after {_DESCRIBE_TIMEOUT}s.") from None
        if result.returncode != 0:
            raise RuntimeError(
                f"D-Robotics RDK: rdkx5-sim --describe failed with exit code {result.returncode}: "
                f"{result.stderr.strip()[-1000:]}"
            )
        try:
            spec = json.loads(result.stdout)
        except json.JSONDecodeError:
            detail = f"{result.stdout[:300]!r} {result.stderr[:300]!r}"
            raise ValueError(f"D-Robotics RDK: rdkx5-sim --describe returned invalid JSON: {detail}") from None
        inputs = spec.get("inputs")
        _check(
            isinstance(inputs, list) and len(inputs) == 1,
            f"rdkx5-sim --describe reports {len(inputs) if isinstance(inputs, list) else inputs} inputs; the "
            f"deployment expects the single compiled NV12 input.",
        )
        entry = inputs[0]
        shape = [int(d) for d in entry["shape"]]
        h, w = shape[-2:]
        storage = entry.get("storage_shape")
        _check(
            shape == [1, 3, h, w]
            and entry.get("dtype") == "uint8"
            and entry.get("pixel_format") == "nv12"
            and h % 2 == 0
            and w % 2 == 0
            and (storage is None or [int(d) for d in storage] == [h + h // 2, w]),
            f"compiled input {entry} is not the even-dimension uint8 (1, 3, H, W) NV12 input.",
        )
        self._validate_outputs(spec.get("outputs", []), "rdkx5-sim --describe")
        self._input_name, self._hw = entry["name"], (h, w)

    def _load_board_runtime(self):
        """Construct the board `hbm_runtime` session and validate its actual I/O."""
        try:
            import hbm_runtime
        except ImportError as e:
            raise RuntimeError(
                "D-Robotics RDK: hbm_runtime is required on RDK boards but not importable. Use the board's matching "
                f"Python environment (it is never auto-installed). See {TOOLCHAIN_DOC}."
            ) from e
        dt, qt = hbm_runtime.hbDNNDataType, hbm_runtime.hbDNNQuantiType
        self._runtime = hbm_runtime.HB_HBMRuntime(str(self._bin))
        names = self._runtime.model_names
        _check(len(names) == 1, f"hbm_runtime loaded {len(names)} models {names}; expected exactly one.")
        (model,) = names
        input_names = self._runtime.input_names[model]
        _check(
            len(input_names) == 1,
            f"board model exposes {len(input_names)} inputs; the deployment expects the single NV12 input.",
        )
        input_name = input_names[0]
        _check(
            str(self._runtime.input_dtypes[model][input_name]) == str(dt.NV12),
            f"board input '{input_name}' is not the NV12 input this backend feeds. {_RE_EXPORT}",
        )
        logical = [int(d) for d in self._runtime.input_shapes[model][input_name]]
        h, w = logical[-2:]
        _check(
            logical == [1, 3, h, w] and h % 2 == 0 and w % 2 == 0,
            f"board input shape {logical} is not an even-dimension (1, 3, H, W) NV12 input.",
        )
        _check(
            str(self._runtime.input_quants[model][input_name].quant_type) == str(qt.NONE),
            f"board input '{input_name}' carries input quantization; raw packed NV12 bytes would not be interpreted "
            f"correctly. {_RE_EXPORT}",
        )
        dtypes = self._runtime.output_dtypes[model]
        quants = self._runtime.output_quants[model]
        outputs = [  # the SDK's str() enums normalized to the describe schema _validate_outputs checks
            {
                "name": name,
                "dtype": "float32" if str(dtypes[name]) == str(dt.F32) else str(dtypes[name]),
                "quanti_type": "NONE" if str(quants[name].quant_type) == str(qt.NONE) else str(quants[name].quant_type),
                "shape": self._runtime.output_shapes[model][name],
            }
            for name in self._runtime.output_names[model]
        ]
        self._validate_outputs(outputs, "board runtime")
        self._model_name, self._input_name, self._hw = model, input_name, (h, w)

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """Run one RGB image batch through the compiled body and its decode tail.

        Args:
            im (torch.Tensor): Input tensor in BHWC format (AutoBackend permutes BCHW for this
                nhwc backend), RGB, float 0-1. Batch must be 1: the deployment is static
                batch-1 and `AutoBackend` serializes larger batches through repeated calls.

        Returns:
            (list[np.ndarray]): The tail's standard predictions — e.g. detect one2many
                `(B, 4 + nc, N)`, end-to-end `(B, max_det, 6 + extra)`, segment plus prototypes,
                classify `(B, nc)` probabilities.
        """
        if im.ndim != 4 or im.shape[-1] != 3 or im.shape[0] != 1:
            raise ValueError(
                f"D-Robotics RDK deployments are static batch-1 and take one RGB BHWC image, got {tuple(im.shape)}; "
                f"larger batches are split automatically by AutoBackend/predict/val."
            )
        image = (im.detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)[0]
        if image.shape[:2] != self._hw:
            raise ValueError(
                f"input image {tuple(image.shape[:2])} does not match the compiled model input {list(self._hw)}; "
                f"predict/val letterbox to imgsz={list(self._hw)}."
            )
        nv12 = _rgb_to_nv12(image)
        raw = self._run_simulator(nv12) if hasattr(self, "_sim") else self._run_board(nv12)
        feeds = {}
        for name, dims in self._inputs.items():
            value = np.array(raw[name], dtype=np.float32)  # one owned copy: SDK arrays are borrowed memory
            shape = _normalized_shape(value.shape, dims) or value.shape  # any real mismatch is ORT's to report
            feeds[name] = value.reshape(shape)
        return self._tail.run(None, feeds)

    def _run_simulator(self, nv12):
        """Execute one rdkx5-sim run call in a fresh scratch directory and return the outputs by name."""
        with tempfile.TemporaryDirectory(prefix="ultralytics_rdk_") as td:
            np.savez(Path(td) / "input.npz", **{self._input_name: nv12})
            try:
                result = subprocess.run(
                    [self._sim, str(self._bin), "--inputs", "input.npz", "--outputs", "output.npz"],
                    cwd=td,
                    capture_output=True,
                    text=True,
                    timeout=_RUN_TIMEOUT,
                    check=False,
                    env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"D-Robotics RDK: rdkx5-sim timed out after {_RUN_TIMEOUT}s.") from None
            if result.returncode != 0:
                raise RuntimeError(
                    f"D-Robotics RDK: rdkx5-sim failed with exit code {result.returncode}: "
                    f"{result.stderr.strip()[-1000:]}"
                )
            out = Path(td) / "output.npz"
            _check(out.is_file(), f"rdkx5-sim wrote no output file: {result.stderr.strip()[-500:]}")
            with np.load(out, allow_pickle=False) as npz:
                missing = [name for name in self._inputs if name not in npz.files]
                _check(not missing, f"rdkx5-sim outputs {npz.files} miss the tail inputs {missing}.")
                return {name: npz[name] for name in self._inputs}

    def _run_board(self, nv12):
        """Execute one board SDK run; the borrowed outputs are copied when they feed the tail."""
        return self._runtime.run({self._model_name: {self._input_name: nv12.reshape(-1)}})[self._model_name]
