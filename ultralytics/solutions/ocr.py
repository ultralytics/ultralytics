# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import ops
from ultralytics.utils.checks import check_requirements
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
        if not callable(self.recognizer) and self.recognizer != "paddleocr":
            raise ValueError("recognizer must be 'paddleocr' or a callable accepting a BGR image crop.")
        self._paddle = None

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

    def _recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognize BGR crops with the configured backend, batching PaddleOCR inference."""
        if callable(self.recognizer):
            return [(text, float(confidence)) for text, confidence in map(self.recognizer, crops)]
        if self._paddle is None:  # load lazily so custom recognizers never require Paddle
            check_requirements((("paddlepaddle>=3.0.0", "paddlepaddle-gpu>=3.0.0"), "paddleocr>=3.0,<4"))
            from paddleocr import TextRecognition

            self._paddle = TextRecognition()
        return [(r["rec_text"], float(r["rec_score"])) for r in self._paddle.predict(crops, batch_size=16)]

    def process(self, im0: np.ndarray) -> SolutionResults:
        """Run text detection, recognition, and reading-order sorting on one image."""
        self.extract_tracks(im0)
        texts, polygons, pending = [], [], []
        key = "box" if self.tracks.masks is None else "polygon"
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
            if crop.size:
                pending.append((i, crop))
            texts.append(
                {
                    "text": "",
                    "confidence": 0.0,
                    "detection_confidence": self.confs[i],
                    "detection_index": i,
                    key: polygon.tolist(),
                }
            )
            polygons.append(polygon)
        recognized = self._recognize([crop for _, crop in pending]) if pending else []
        for (i, _), (text, confidence) in zip(pending, recognized):
            texts[i].update(text=text, confidence=confidence)
        order = self.reading_order(polygons)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        if self.CFG["show_boxes"]:
            for i in order:
                annotator.draw_region(polygons[i].astype(np.int32).tolist(), (0, 255, 0), self.line_width)
        if self.show_labels:
            annotator = SolutionAnnotator(
                annotator.im,
                line_width=self.line_width,
                font_size=max(round(self.line_width * 6), 12),
                pil=True,
                example="".join(text["text"] for text in texts),
            )
            for i in order:
                text = texts[i]
                x1, y1 = polygons[i].min(0)
                label = text["text"]
                label += f" {text['confidence']:.2f}" if self.show_conf else ""
                width, height = annotator.font.getsize(label)
                x = min(max(int(x1), 0), max(im0.shape[1] - width - 4, 0))
                annotator.text([x, max(int(y1) - 2, height + 1)], label, anchor="bottom", box_color=(0, 0, 0))
        result = SolutionResults(plot_im=annotator.result(), texts=[texts[i] for i in order])
        self.display_output(result.plot_im)
        return result
