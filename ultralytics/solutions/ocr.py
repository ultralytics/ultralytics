# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator
from ultralytics.utils import ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.plotting import save_one_box


class OCR(BaseSolution):
    """Detect, recognize, order, and track scene text with a YOLO detector and a recognition backend.

    Examples:
        >>> ocr = OCR(model="scene-text-obb.pt", recognizer="paddleocr")
        >>> result = ocr(im0)
        >>> print(result.texts)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize OCR with a YOLO detector and a PaddleOCR backend or custom crop-recognition callable."""
        kwargs["tracker"] = kwargs.get("tracker", "bytetrack.yaml")
        super().__init__(**kwargs)
        self.recognizer = self.CFG["recognizer"]
        if not callable(self.recognizer) and self.recognizer != "paddleocr":
            raise ValueError("recognizer must be 'paddleocr' or a callable accepting a BGR image crop.")
        self.recognition_interval = max(1, self.CFG["recognition_interval"])
        self.text_cache = {}
        self.ocr_frame = 0
        self._paddle = None

    def forget_tracks(self, track_ids: list[int]) -> None:
        """Drop tracking and recognition history for retired track IDs."""
        super().forget_tracks(track_ids)
        for track_id in track_ids:
            self.text_cache.pop(track_id, None)

    @staticmethod
    def reading_order(polygons: list[np.ndarray]) -> list[int]:
        """Return detection indices in deterministic top-to-bottom, left-to-right reading order."""
        entries = sorted(
            enumerate(polygons),
            key=lambda item: (float(item[1][:, 1].mean()), float(item[1][:, 0].min()), item[0]),
        )
        lines = []
        for index, polygon in entries:
            center, height = float(polygon[:, 1].mean()), float(np.ptp(polygon[:, 1]))
            if lines and abs(center - lines[-1][0]) <= max(height, lines[-1][1], 1.0) * 0.5:
                line = lines[-1]
                count = len(line[2])
                line[0] = (line[0] * count + center) / (count + 1)
                line[1] = max(line[1], height)
                line[2].append((index, polygon))
            else:
                lines.append([center, height, [(index, polygon)]])
        return [
            index
            for _, _, line in sorted(lines, key=lambda line: line[0])
            for index, _ in sorted(line, key=lambda item: (float(item[1][:, 0].min()), item[0]))
        ]

    def _init_paddle(self) -> None:
        """Initialize PaddleOCR lazily for text-line recognition without duplicate text detection."""
        if not check_requirements(["paddlepaddle", "paddleocr>=3.0,<4"], install=False):
            raise ModuleNotFoundError(
                "PaddleOCR recognition requires PaddlePaddle and PaddleOCR 3.x. "
                "Install matching builds with 'pip install paddlepaddle paddleocr>=3.0,<4'."
            )
        from paddleocr import TextRecognition

        self._paddle = TextRecognition()

    def _recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """Recognize one BGR crop with the configured backend."""
        if callable(self.recognizer):
            text, confidence = self.recognizer(crop)
        else:
            if self._paddle is None:
                self._init_paddle()
            output = self._paddle.predict(crop)[0]
            text, confidence = output["rec_text"], output["rec_score"]
        if not isinstance(text, str) or not isinstance(confidence, (float, int, np.floating)):
            raise TypeError("recognizer must return a (text, confidence) tuple containing str and float values.")
        return text, float(confidence)

    def process(self, im0: np.ndarray) -> Results:
        """Run text detection, recognition, reading-order sorting, and optional track association on one image."""
        self.extract_tracks(im0)
        self.ocr_frame += 1
        texts = []
        polygons = []
        for i, box in enumerate(self.boxes):
            polygon = box.numpy().reshape(-1, 2)
            crop_box = box
            geometry = {"box": polygon.tolist()}
            if self.tracks.masks is not None:
                polygon = self.tracks.masks.xy[i].astype(np.float32)
                rectangle = cv2.minAreaRect(polygon)
                crop_box = ops.xywhr2xyxyxyxy(
                    torch.tensor([*rectangle[0], *rectangle[1], np.deg2rad(rectangle[2])], dtype=torch.float32)
                )
                geometry = {"polygon": polygon.tolist()}
            elif polygon.size == 4:
                x1, y1, x2, y2 = box.tolist()
                polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                geometry = {"box": polygon.tolist()}
            track_id = self.track_ids[i] if self.track_ids else None
            cached = self.text_cache.get(track_id) if track_id is not None else None
            if cached is None or self.ocr_frame - cached["frame"] >= self.recognition_interval:
                crop = save_one_box(crop_box, im0, gain=1.0, pad=0, BGR=True, save=False)
                text, confidence = self._recognize(crop) if crop.size else ("", 0.0)
                cached = {"text": text, "confidence": confidence, "frame": self.ocr_frame}
                if track_id is not None:
                    self.text_cache[track_id] = cached
            texts.append(
                {
                    "text": cached["text"],
                    "confidence": cached["confidence"],
                    "detection_confidence": self.confs[i],
                    "detection_index": i,
                    **geometry,
                    **({"track_id": track_id} if track_id is not None else {}),
                }
            )
            polygons.append(polygon)
        order = self.reading_order(polygons) if polygons else []
        result = self.tracks
        result.update(texts=[texts[i] for i in order])
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        for i in order:
            text = texts[i]
            points = np.array(text.get("polygon", text.get("box")), dtype=np.int32)
            annotator.draw_region(points.tolist(), (0, 255, 0), self.line_width)
            if self.show_labels:
                x1, y1 = points.min(0)
                x2, y2 = points.max(0)
                label = f"{text['track_id']} {text['text']}" if "track_id" in text else text["text"]
                annotator.adaptive_label((int(x1), int(y1), int(x2), int(y2)), label)
        result.plot_im = annotator.im
        self.display_output(result.plot_im)
        return result
