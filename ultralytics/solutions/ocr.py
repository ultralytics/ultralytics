# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import ops
from ultralytics.utils.checks import check_paddle, check_requirements
from ultralytics.utils.plotting import save_one_box


class OCR(BaseSolution):
    """Detect, recognize, and order scene text with a YOLO detector and a recognition backend.

    Examples:
        >>> ocr = OCR(model="scene-text-obb.pt", recognizer="paddleocr")
        >>> result = ocr(im0)
        >>> print(result.texts)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize OCR with a YOLO detector and a PaddleOCR backend or custom crop-recognition callable."""
        kwargs["tracker"] = None
        super().__init__(**kwargs)
        self.recognizer = self.CFG["recognizer"]
        if self.recognizer == "paddleocr":
            check_paddle(">=3.1.0", gpu=">=3.1.0")  # PP-OCRv6 requires PaddlePaddle 3.1+
            check_requirements("paddleocr>=3.7,<4")
            from paddleocr import TextLineOrientationClassification, TextRecognition

            self._paddle = TextRecognition()
            self._orientation = TextLineOrientationClassification()  # flags upside-down text lines
        elif not callable(self.recognizer):
            raise ValueError("recognizer must be 'paddleocr' or a callable accepting a BGR image crop.")

    @staticmethod
    def reading_order(polygons: list[np.ndarray]) -> list[int]:
        """Return detection indices in deterministic top-to-bottom, left-to-right reading order."""
        lines = []  # [mean center y, max height, indices] per text line
        for i in sorted(range(len(polygons)), key=lambda i: (polygons[i][:, 1].mean(), polygons[i][:, 0].min())):
            y, h = float(polygons[i][:, 1].mean()), float(np.ptp(polygons[i][:, 1]))
            if lines and abs(y - lines[-1][0]) <= max(h, lines[-1][1], 1.0) * 0.5:  # same line as previous text
                line = lines[-1]
                line[0] = (line[0] * len(line[2]) + y) / (len(line[2]) + 1)
                line[1] = max(line[1], h)
                line[2].append(i)
            else:
                lines.append([y, h, [i]])
        return [
            i
            for *_, idx in sorted(lines, key=lambda x: x[0])
            for i in sorted(idx, key=lambda i: (polygons[i][:, 0].min(), i))
        ]

    def _recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognize BGR crops with the configured backend, batching PaddleOCR inference."""
        if callable(self.recognizer):
            return [(text, float(confidence)) for text, confidence in map(self.recognizer, crops)]
        reads = [(r["rec_text"], float(r["rec_score"])) for r in self._paddle.predict(crops, batch_size=16)]
        # Boxes cannot tell 0 from 180 degrees: re-read crops classified upside down, keep the more confident reading
        orientations = self._orientation.predict(crops, batch_size=16)
        flip = [i for i, r in enumerate(orientations) if r["label_names"][0] == "180_degree"]
        flipped = self._paddle.predict([cv2.rotate(crops[i], cv2.ROTATE_180) for i in flip], batch_size=16)
        for i, r in zip(flip, flipped):
            if r["rec_score"] > reads[i][1]:
                reads[i] = (r["rec_text"], float(r["rec_score"]))
        return reads

    def process(self, im0: np.ndarray) -> SolutionResults:
        """Run text detection, recognition, and reading-order sorting on one image."""
        self.extract_tracks(im0)
        texts, polygons, pending = [], [], []
        for i, box in enumerate(self.boxes):
            if self.tracks.masks is not None:  # keep the polygon, crop its minimum-area rectangle
                polygon = self.tracks.masks.xy[i]
                box = ops.xywhr2xyxyxyxy(ops.xyxyxyxy2xywhr(polygon[None]))[0]
            elif box.ndim == 2:  # OBB corners
                polygon = box.numpy()
            else:  # xyxy box corners
                x1, y1, x2, y2 = box.tolist()
                polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            crop = save_one_box(box, im0, gain=1.0, pad=0, BGR=True, save=False)
            if crop.shape[0] >= 1.5 * crop.shape[1] > 0:  # read tall crops sideways like PaddleOCR for steep text
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if crop.size:
                pending.append((i, crop))
            texts.append(
                {
                    "text": "",
                    "confidence": 0.0,
                    "detection_confidence": self.confs[i],
                    "detection_index": i,
                    "polygon": polygon.tolist(),
                }
            )
            polygons.append(polygon)
        recognized = self._recognize([crop for _, crop in pending]) if pending else []
        for (i, _), (text, confidence) in zip(pending, recognized):
            texts[i].update(text=text, confidence=confidence)
        order = self.reading_order(polygons)
        annotator = SolutionAnnotator(
            im0, self.line_width, font_size=max(self.line_width * 6, 12), example="".join(t["text"] for t in texts)
        )
        for i in order:
            text, confidence = texts[i]["text"], texts[i]["confidence"]
            label = (f"{text} {confidence:.2f}" if self.show_conf else text) if self.show_labels else ""
            annotator.box_label(polygons[i], label, color=(0, 255, 0), txt_color=(0, 0, 0))
        result = SolutionResults(plot_im=annotator.result(), texts=[texts[i] for i in order])
        self.display_output(result.plot_im)
        return result
