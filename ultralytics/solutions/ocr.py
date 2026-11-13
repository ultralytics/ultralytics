# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import threading
import uuid
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LINUX, checks, ops
from ultralytics.utils.checks import check_python, check_requirements, check_tensorrt, rocm_is_available
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.torch_utils import select_device


class TextRecognizer:
    """PP-OCRv6 text-line recognizer on ONNX Runtime or TensorRT, reading BGR line crops in width-sorted batches.

    Examples:
        >>> recognize = TextRecognizer()
        >>> text, confidence = recognize([crop])[0]
        >>> recognize = TextRecognizer(device=0, engine=True)  # FP16 TensorRT on NVIDIA GPUs
    """

    def __init__(self, model: str = "PP-OCRv6_medium_rec.onnx", device: str | int | None = None, engine: bool = False):
        """Load the recognizer on the NVIDIA GPU `device` when available, else on CPU, with its embedded character set.

        With `engine=True` on an NVIDIA GPU it runs as an FP16 TensorRT engine, built once per TensorRT version and GPU
        architecture (about 30 s) and cached next to the ONNX model.
        """
        device = select_device("cpu" if device is None and not torch.cuda.is_available() else device, verbose=False)
        cuda = getattr(device, "type", "cpu") == "cuda" and not rocm_is_available()  # others, AMD too, read on CPU
        model = str(attempt_download_asset(model))
        self.session = None
        if engine and cuda:  # one TensorRT context, whose output buffer every batch reuses, so one batch at a time
            self.backend, self.lock = self.load_engine(model, device), threading.Lock()
            alphabet = self.backend.metadata["character"]
        else:
            # Interchangeable ONNX Runtime builds, as the ONNX backend checks them, so a GPU build is never replaced.
            # 1.26 hands outputs to torch via DLPack and needs Python 3.11+
            check_python("3.11")
            builds = (
                (checks.ONNXRUNTIME_CUDA, "onnxruntime-gpu", "onnxruntime-windowsml")  # Jetson lacks [cuda]
                if cuda
                else ("onnxruntime", "onnxruntime-gpu", "onnxruntime-vitisai", "onnxruntime-windowsml")
            )
            check_requirements([tuple(f"{b}{',' * ('<' in b)}>=1.26.0" for b in builds)])
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if cuda and "CUDAExecutionProvider" in ort.get_available_providers():
                if LINUX and hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()  # load pip-installed CUDA libraries that torch does not load itself
                # Heuristic cuDNN algorithms: the default exhaustive search re-benchmarks every new width and batch
                options = {"device_id": device.index or 0, "cudnn_conv_algo_search": "HEURISTIC"}
                providers.insert(0, ("CUDAExecutionProvider", options))
            self.session, self.lock = ort.InferenceSession(model, providers=providers), nullcontext()  # concurrent
            # Character probabilities stay on the recognizer's device, which returns only each step's best character
            output = self.session.get_outputs()[0].name
            self.output = output, "cuda" if len(providers) > 1 else "cpu", device.index or 0
            alphabet = self.session.get_modelmeta().custom_metadata_map["character"]
        self.characters = ["", *alphabet.split("\n"), " "]  # CTC alphabet: blank, the model's characters, then space

    @staticmethod
    def load_engine(model: str, device: torch.device):
        """Load the TensorRT engine for ONNX `model` on `device`, building it on first use for this TensorRT and GPU."""
        check_tensorrt()
        import tensorrt as trt

        from ultralytics.nn.backends.tensorrt import TensorRTBackend

        sm = "".join(map(str, torch.cuda.get_device_capability(device)))
        f = Path(model).with_name(f"{Path(model).stem}-trt{trt.__version__}-sm{sm}.engine")
        if not f.exists():  # dynamic batches of up to 8 words, 48 px high and up to 3200 px wide
            check_requirements(["onnx>=1.12.0,<2.0.0", "onnxconverter-common"])
            import onnx
            from onnxconverter_common import float16

            from ultralytics.utils.export.engine import onnx2engine

            tmp = f.with_suffix(f".{uuid.uuid4().hex}.tmp")  # concurrent builders, even in other containers
            # FP16 weights, which TensorRT builds as typed: the same reads ~40% faster than FP32, with no ModelOpt
            onnx_model, fp16 = onnx.load(model), tmp.with_suffix(".onnx")
            try:
                with warnings.catch_warnings():  # one warning per tiny weight clamped to the FP16 range
                    warnings.simplefilter("ignore")
                    onnx.save(float16.convert_float_to_float16(onnx_model, keep_io_types=True), fp16)
                metadata = {p.key: p.value for p in onnx_model.metadata_props}
                onnx2engine(str(fp16), tmp, dynamic=True, shape=(8, 3, 48, 1600), metadata=metadata)
            finally:
                fp16.unlink(missing_ok=True)
            os.replace(tmp, f)
        return TensorRTBackend(f, device)

    def __call__(self, crops: list[np.ndarray], batch: int = 8) -> list[tuple[str, float]]:
        """Read crops 48 px high, right-padded to at least 320 px as PaddleOCR does, returning text and confidence.

        Width-sorted batches of 8 keep padding close to each line's own width: batches of 32 merged words on a dense
        page.
        """
        lines = []
        for crop in crops:
            if crop.shape[0] >= 1.5 * crop.shape[1]:  # read tall crops sideways, as PaddleOCR does for steep text
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            width = min(3200, max(1, round(48 * crop.shape[1] / crop.shape[0])))
            lines.append(cv2.resize(crop, (width, 48)).transpose(2, 0, 1) / 127.5 - 1)
        order = sorted(range(len(lines)), key=lambda i: lines[i].shape[2])
        reads = [("", 0.0)] * len(lines)
        io = self.session and self.session.io_binding()
        for start in range(0, len(order), batch):
            indices = order[start : start + batch]
            x = np.zeros((len(indices), 3, 48, max(320, lines[indices[-1]].shape[2])), np.float32)
            for j, i in enumerate(indices):
                x[j, :, :, : lines[i].shape[2]] = lines[i]
            with self.lock:  # TensorRT: read this batch before the next one overwrites its output buffer
                if io is None:
                    with torch.cuda.device(self.backend.device):  # from any thread, on the engine's GPU
                        probs = self.backend.forward(torch.from_numpy(x).to(self.backend.device))[0]
                else:
                    io.bind_cpu_input("x", x)
                    io.bind_output(*self.output)
                    self.session.run_with_iobinding(io)
                    probs = torch.from_dlpack(io.get_outputs()[0])
                best = probs.max(-1)
                best = best.values.cpu().numpy(), best.indices.cpu().numpy()
            for i, scores, ids in zip(indices, *best):
                keep = (ids != 0) & np.r_[True, ids[1:] != ids[:-1]]  # CTC: drop blanks and repeats
                reads[i] = (
                    "".join(self.characters[c] for c in ids[keep]),
                    float(scores[keep].mean()) if keep.any() else 0.0,
                )
        return reads


class OCR(BaseSolution):
    """Detect words with a YOLO text detector, then read and order them at any page rotation.

    Examples:
        >>> ocr = OCR(model="yolo26x-text.pt")
        >>> result = ocr(im0)
        >>> print(result.texts)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize OCR with a YOLO text detector and the PP-OCRv6 recognizer or a custom crop-recognition callable."""
        kwargs["tracker"] = None
        kwargs.setdefault("model", "yolo26x-text.pt")
        kwargs.setdefault("imgsz", 1280)  # text is small; smaller images run at their own size, never upscaled
        kwargs.setdefault("max_det", 1000)  # a document page holds more words than the default 300 detections
        super().__init__(**kwargs)
        recognizer = self.CFG["recognizer"]
        if isinstance(recognizer, str):  # PaddleOCR model name, e.g. 'korean_PP-OCRv5_mobile_rec'
            model = "PP-OCRv6_medium_rec" if recognizer in {"paddleocr", "tensorrt"} else recognizer
            self.recognize = TextRecognizer(f"{model}.onnx", self.CFG["device"], engine=recognizer == "tensorrt")
        elif callable(recognizer):
            self.recognize = lambda crops: [(text, float(confidence)) for text, confidence in map(recognizer, crops)]
        else:
            raise TypeError("recognizer must be a backend or PaddleOCR model name, or a callable reading a BGR crop.")

    @staticmethod
    def lines(polygons: np.ndarray) -> list[list[int]]:
        """Group word polygons into lines in reading order: baseline bands, split at gaps wider than 1.5 text heights.

        A word joins a band within half the smaller of its and the band's heights, so a large heading and the small
        text beside it stay separate lines.
        """
        bands = []  # [mean center y, max height, indices] per band
        for i in sorted(range(len(polygons)), key=lambda i: (polygons[i][:, 1].mean(), polygons[i][:, 0].min())):
            y, h = float(polygons[i][:, 1].mean()), float(np.ptp(polygons[i][:, 1]))
            if bands and abs(y - bands[-1][0]) <= max(min(h, bands[-1][1]), 1.0) * 0.5:
                band = bands[-1]
                band[0] = (band[0] * len(band[2]) + y) / (len(band[2]) + 1)
                band[1] = max(band[1], h)
                band[2].append(i)
            else:
                bands.append([y, h, [i]])
        lines = []
        for _, h, indices in sorted(bands, key=lambda band: band[0]):
            indices.sort(key=lambda i: polygons[i][:, 0].min())
            lines.append(indices[:1])
            for i in indices[1:]:
                if polygons[i][:, 0].min() - polygons[lines[-1][-1]][:, 0].max() > 1.5 * h:
                    lines.append([i])
                else:
                    lines[-1].append(i)
        return lines

    @staticmethod
    def read_words(
        im0: np.ndarray, polygons: np.ndarray, recognize: Callable[[list[np.ndarray]], list[tuple[str, float]]]
    ) -> list[dict[str, Any]]:
        """Read word polygons (N, 4, 2) one by one at any page rotation, returning the words in reading order.

        The page's text direction (mod 180°) is the words' long-side angle weighted by elongation; words under 2:1 abstain
        (single characters are taller than wide), so a page of only those stays unrotated. The page is rotated so that
        direction runs left to right, where words are cropped upright and grouped into lines. Of that turn and its
        opposite, the one whose 8 widest words `recognize` reads more confidently wins, as upside-down text reads as
        low-confidence noise. Each word keeps its `detection_index` and `polygon` and gains its `line` number; words that
        read as nothing are dropped.
        """
        if not len(polygons):
            return []
        _, _, w, h, angle = ops.xyxyxyxy2xywhr(polygons).T  # w is the long side
        weight = np.where(w >= 2 * h, w - h, 0)
        angle = 0.5 * np.arctan2((weight * np.sin(2 * angle)).sum(), (weight * np.cos(2 * angle)).sum())
        turns = []
        for turn in (0, np.pi):
            ih, iw = im0.shape[:2]
            rotation = cv2.getRotationMatrix2D((iw / 2, ih / 2), float(np.degrees(angle + turn)), 1.0)
            cos, sin = abs(rotation[0, 0]), abs(rotation[0, 1])
            size = round(ih * sin + iw * cos), round(ih * cos + iw * sin)  # the rotated page's bounds
            rotation[:, 2] += (size[0] - iw) / 2, (size[1] - ih) / 2
            page = cv2.warpAffine(im0, rotation, size, borderValue=(255, 255, 255))
            points = polygons @ rotation[:, :2].T + rotation[:, 2]
            # Corners ordered so save_one_box rectifies each word upright in the rotated page
            rects = ops.xywhr2xyxyxyxy(ops.xyxyxyxy2xywhr(points))
            turns.append(
                (OCR.lines(points), [save_one_box(r, page, gain=1.0, pad=0, BGR=True, save=False) for r in rects])
            )
        # Read each turn's 8 widest words, keep the more confident turn, and reuse its reads
        widest = [
            sorted((i for i, c in enumerate(crops) if c.size), key=lambda i: crops[i].shape[1])[-8:]
            for _, crops in turns
        ]
        trials = [dict(zip(w, recognize([crops[i] for i in w]))) for w, (_, crops) in zip(widest, turns)]
        best = int(np.argmax([np.mean([0, *(c for _, c in t.values())]) for t in trials]))
        lines, crops = turns[best]
        rest = [i for i, crop in enumerate(crops) if crop.size and i not in trials[best]]
        reads = {**trials[best], **dict(zip(rest, recognize([crops[i] for i in rest])))}
        return [
            {
                "text": reads[i][0],
                "confidence": reads[i][1],
                "detection_index": i,
                "line": k,
                "polygon": polygons[i].tolist(),
            }
            for k, line in enumerate(lines)
            for i in line
            if reads.get(i, ("",))[0]
        ]

    def process(self, im0: np.ndarray) -> SolutionResults:
        """Detect words, then read them and return them in reading order with their line numbers."""
        if isinstance(self.CFG["imgsz"], int):  # never upscale: smaller images run at their own size
            self.track_add_args["imgsz"] = min(self.CFG["imgsz"], -(-max(im0.shape[:2]) // 32) * 32)
        self.extract_tracks(im0)
        if self.tracks.masks is not None:  # segmentation: each mask's minimum-area rectangle
            polygons = np.array([ops.xywhr2xyxyxyxy(ops.xyxyxyxy2xywhr(p[None]))[0] for p in self.tracks.masks.xy])
        elif len(self.boxes) and self.boxes.ndim == 3:  # OBB corners
            polygons = self.boxes.numpy()
        else:  # xyxy boxes
            polygons = np.asarray(self.boxes, np.float32).reshape(-1, 4)[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
        outlines = self.tracks.masks.xy if self.tracks.masks is not None else None  # masks keep their outline
        texts = [
            {
                **word,
                "detection_confidence": self.confs[word["detection_index"]],
                **({"polygon": outlines[word["detection_index"]].tolist()} if outlines is not None else {}),
            }
            for word in self.read_words(im0, polygons.astype(np.float32), self.recognize)
        ]
        annotator = SolutionAnnotator(
            im0, self.line_width, font_size=max(self.line_width * 6, 12), example="".join(t["text"] for t in texts)
        )
        for t in texts:
            label = (f"{t['text']} {t['confidence']:.2f}" if self.show_conf else t["text"]) if self.show_labels else ""
            annotator.box_label(np.array(t["polygon"]), label, color=(0, 255, 0), txt_color=(0, 0, 0))
        result = SolutionResults(plot_im=annotator.result(), texts=texts)
        self.display_output(result.plot_im)
        return result
