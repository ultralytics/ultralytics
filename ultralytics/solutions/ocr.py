# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults


class OCR(BaseSolution):
    """Run user-provided OCR on YOLO-detected regions in a static image.

    Args:
        recognizer (Callable[[np.ndarray], str]): Callable that receives one BGR image crop and returns recognized text.
        **kwargs (Any): Keyword arguments passed to :class:`BaseSolution`, including ``model``, ``classes``, ``conf``,
            ``iou``, ``device``, ``imgsz``, ``show``, ``show_boxes``, ``show_conf``, and ``show_labels``.

    Examples:
        Define an OCR callable and process an image:

        >>> import numpy as np
        >>> recognizer = lambda crop: "AB123CD"
        >>> ocr = OCR(model="text_detector.pt", recognizer=recognizer)
        >>> result = ocr(np.zeros((100, 100, 3), dtype=np.uint8))
    """

    def __init__(self, recognizer: Callable[[np.ndarray], str], **kwargs: Any) -> None:
        """Initialize the OCR solution."""
        if not callable(recognizer):
            raise TypeError("recognizer must be callable")
        super().__init__(**kwargs)
        self.recognizer = recognizer

    def process(self, im0: np.ndarray) -> SolutionResults:
        """Detect regions and run the configured recognizer on each crop.

        Args:
            im0 (np.ndarray): Input BGR image.

        Returns:
            (SolutionResults): Results containing the annotated image and aligned ``ocr_texts``, ``boxes``,
                ``classes``, and ``confidences`` lists.
        """
        with self.profilers[0]:
            results = self.model.predict(
                im0,
                classes=self.classes,
                conf=self.CFG["conf"],
                iou=self.CFG["iou"],
                device=self.CFG["device"],
                imgsz=self.CFG["imgsz"],
                max_det=self.CFG["max_det"],
                verbose=False,
            )[0]

        annotator = SolutionAnnotator(im0, self.line_width)
        texts, boxes, classes, confidences = [], [], [], []
        height, width = im0.shape[:2]

        detected_boxes = getattr(results, "boxes", None)
        if detected_boxes is not None:
            detections = zip(detected_boxes.xyxy, detected_boxes.cls, detected_boxes.conf)
        else:
            detections = ()

        for box, cls, conf in detections:
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, width), min(y2, height)
            if x1 >= x2 or y1 >= y2:
                continue

            text = self.recognizer(im0[y1:y2, x1:x2].copy())
            if not isinstance(text, str):
                raise TypeError(f"recognizer must return str, got {type(text).__name__}")

            box_list = [x1, y1, x2, y2]
            class_id, confidence = int(cls), float(conf)
            texts.append(text)
            boxes.append(box_list)
            classes.append(class_id)
            confidences.append(confidence)

            if self.CFG["show_boxes"]:
                label = None
                if self.show_labels:
                    label = f"{self.names[class_id]}: {text}"
                    if self.show_conf:
                        label += f" {confidence:.2f}"
                annotator.box_label(box_list, label=label)

        plot_im = annotator.result()
        self.display_output(plot_im)
        return SolutionResults(
            plot_im=plot_im,
            ocr_texts=texts,
            boxes=boxes,
            classes=classes,
            confidences=confidences,
        )
