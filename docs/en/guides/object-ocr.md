---
title: Detection to OCR with Ultralytics Solutions
description: Run a user-provided OCR callable on regions detected by a YOLO model.
keywords: Ultralytics, OCR, object detection, text recognition, Solution
---

# Detection to OCR

The `OCR` Solution combines a user-supplied YOLO detector with a user-supplied OCR callable. The detector locates each
region, and the callable receives one BGR `numpy.ndarray` crop at a time and returns its recognized text.

Ultralytics does not install or select an OCR backend for this Solution. You can adapt Tesseract, PaddleOCR, a hosted
service, or a custom model to the callable interface:

```python
import cv2
import numpy as np
import pytesseract


def recognizer(crop: np.ndarray) -> str:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 7")
    return " ".join(text.split())
```

Pass that callable together with a detector trained for the regions you want to read:

```python
from ultralytics import solutions

ocr = solutions.OCR(model="text_detector.pt", recognizer=recognizer)
result = ocr(cv2.imread("image.jpg"))

print(result.ocr_texts)
print(result.boxes)
```

`result.ocr_texts`, `result.boxes`, `result.classes`, and `result.confidences` are aligned by detection index. The
annotated image is available as `result.plot_im`. Set `show=False` (the default) to skip display, or use
`show_boxes=False` and `show_labels=False` to control annotations.

The shared `BaseSolution` timing output stores the model inference time under `result.speed["track"]`, even though this
Solution uses prediction rather than tracking. The other entry, `result.speed["solution"]`, is the OCR and annotation
time.

This Solution processes one image per call. It does not track objects, combine text across frames, or provide an OCR
backend. The YOLO model must detect the regions that are passed to the recognizer; a general object model is not
automatically a text detector.
