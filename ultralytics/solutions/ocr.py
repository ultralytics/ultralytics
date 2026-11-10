# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.plotting import save_one_box


class TextRecognizer:
    """PP-OCRv6 text-line recognizer on ONNX Runtime, reading BGR line crops in width-sorted batches.

    Examples:
        >>> recognize = TextRecognizer()
        >>> text, confidence = recognize([crop])[0]
    """

    def __init__(self, model: str = "PP-OCRv6_medium_rec.onnx", device: str | int | None = None) -> None:
        """Load the recognizer on CUDA when available and requested, else on CPU, with its embedded character set."""
        cuda = torch.cuda.is_available() and str(device) != "cpu"
        check_requirements("onnxruntime-gpu" if cuda else "onnxruntime")
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if cuda:
            ort.preload_dlls()  # CUDA and cuDNN from the pip packages PyTorch uses, which are not on the loader path
            # Heuristic cuDNN algorithms: the default exhaustive search re-benchmarks every new line width and batch size
            providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
        self.session = ort.InferenceSession(str(attempt_download_asset(model)), providers=providers)
        # CTC alphabet: blank, the model's embedded characters, then space
        self.characters = ["", *self.session.get_modelmeta().custom_metadata_map["character"].split("\n"), " "]

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
        for start in range(0, len(order), batch):
            indices = order[start : start + batch]
            x = np.zeros((len(indices), 3, 48, max(320, lines[indices[-1]].shape[2])), np.float32)
            for j, i in enumerate(indices):
                x[j, :, :, : lines[i].shape[2]] = lines[i]
            for i, probs in zip(indices, self.session.run(None, {"x": x})[0]):
                ids, scores = probs.argmax(1), probs.max(1)
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
        if recognizer == "paddleocr":
            self.recognize = TextRecognizer(device=self.CFG["device"])
        elif callable(recognizer):
            self.recognize = lambda crops: [(text, float(confidence)) for text, confidence in map(recognizer, crops)]
        else:
            raise ValueError("recognizer must be 'paddleocr' or a callable accepting a BGR image crop.")

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
        low-confidence noise. Each word keeps its `detection_index` and `polygon` and gains its `line` number.
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
        widest = [sorted((c for c in crops if c.size), key=lambda c: c.shape[1])[-8:] for _, crops in turns]
        lines, crops = turns[int(np.argmax([np.mean([0, *(c for _, c in recognize(w))]) for w in widest]))]
        kept = [i for i, crop in enumerate(crops) if crop.size]
        reads = dict(zip(kept, recognize([crops[i] for i in kept])))
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
        self.track_add_args["imgsz"] = min(self.CFG["imgsz"], -(-max(im0.shape[:2]) // 32) * 32)  # never upscale
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
