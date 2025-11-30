# Human Posture Detection (Standing / Sitting / Unknown)

This example demonstrates how to use **YOLOv8-Pose** to classify human posture
from webcam input, using geometric relationships between detected body keypoints.

---

## 🧠 Overview

The script detects a person in real time through the webcam and estimates their posture
as **Standing**, **Sitting**, or **Unknown** based on:

- The **knee–hip–shoulder angle** (side view)
- The **leg/bust ratio** (front view)

It uses **YOLOv8-Pose** from the `ultralytics` package and simple geometric heuristics.

---

## ⚙️ Requirements

Make sure you have the following installed:

```bash
pip install ultralytics opencv-python torch numpy
```
