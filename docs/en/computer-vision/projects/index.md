---
comments: true
title: Computer Vision Projects — Beginner to Advanced with YOLO
description: A curated library of 25+ computer vision projects from beginner to advanced. Each project includes working Python code, difficulty rating, techniques used, and links to full Ultralytics implementation guides.
keywords: computer vision projects, computer vision projects for beginners, YOLO projects, object detection projects, image segmentation projects, deep learning projects, computer vision examples, computer vision project ideas, best computer vision projects, OpenCV projects
---

# Computer Vision Projects

A curated library of 25+ computer vision projects built with [Ultralytics YOLO](https://www.ultralytics.com/yolo) — organised by difficulty from first-time experiments to production-ready systems. Each project includes working Python code, the core techniques involved, and a link to the full implementation guide.

Install Ultralytics to get started with any project on this page:

```bash
pip install ultralytics
```

---

## All Projects at a Glance

| Project | Difficulty | Technique |
|---|---|---|
| [Security Alarm System](#1-security-alarm-system) | Beginner | Object detection |
| [Object Counting](#2-object-counting) | Beginner | Detection + counting |
| [Live Inference App](#3-live-inference-app) | Beginner | Detection, Streamlit |
| [Object Blurring & Anonymisation](#4-object-blurring--anonymisation) | Beginner | Segmentation |
| [Image Classification](#5-image-classification) | Beginner | Classification |
| [Object Cropping](#6-object-cropping) | Beginner | Detection, region extraction |
| [Visual Similarity Search](#7-visual-similarity-search) | Beginner | Feature embeddings |
| [Workout & Rep Counter](#8-workout--rep-counter) | Intermediate | Pose estimation |
| [Parking Management System](#9-parking-management-system) | Intermediate | Detection, occupancy logic |
| [Queue Management System](#10-queue-management-system) | Intermediate | Detection + tracking |
| [Speed Estimation](#11-speed-estimation) | Intermediate | Tracking, velocity |
| [Heatmap Generation](#12-heatmap-generation) | Intermediate | Tracking, density mapping |
| [Region & Zone Counting](#13-region--zone-counting) | Intermediate | Tracking, region logic |
| [Distance Calculation](#14-distance-calculation) | Intermediate | Detection, spatial reasoning |
| [Instance Segmentation with Tracking](#15-instance-segmentation-with-tracking) | Intermediate | Segmentation + tracking |
| [Analytics Dashboard](#16-analytics-dashboard) | Intermediate | Detection, visualisation |
| [Vision Eye — Perspective Mapping](#17-vision-eye--perspective-mapping) | Intermediate | Detection, homography |
| [Pose Estimation & Ergonomics](#18-pose-estimation--ergonomics) | Intermediate | Pose estimation |
| [Rotated Object Detection](#19-rotated-object-detection-obb) | Advanced | OBB, aerial imagery |
| [Tiled Inference for Large Images](#20-tiled-inference-for-large-images) | Advanced | SAHI, sliced inference |
| [Open-Vocabulary Detection](#21-open-vocabulary-detection) | Advanced | YOLO-World, text prompts |
| [Zero-Shot Segmentation](#22-zero-shot-segmentation) | Advanced | SAM 2, promptable masks |
| [Defect & Anomaly Detection](#23-defect--anomaly-detection) | Advanced | Custom training |
| [Edge Deployment — Jetson & Raspberry Pi](#24-edge-deployment--jetson--raspberry-pi) | Advanced | TensorRT, ONNX, embedded |
| [Fine-Tune on Custom Dataset](#25-fine-tune-on-custom-dataset) | Advanced | Transfer learning |

---

## Beginner Projects

Minimal setup, a single core technique, and immediate results. Ideal if this is your first computer vision project.

### 1. Security Alarm System

Detect people or vehicles entering a restricted zone in real time and trigger an alert. A practical introduction to object detection, regions of interest, and integrating model output with external actions.

```python
from ultralytics import solutions

alarm = solutions.SecurityAlarm(
    model="yolo26n.pt",
    show=True,
)

alarm.monitor(source="path/to/video.mp4")
```

**Guide:** [Security Alarm System](../../guides/security-alarm-system.md) · **Task:** [Object Detection](../../tasks/detect.md)

---

### 2. Object Counting

Count objects passing through a frame or crossing a defined line. One of the most common real-world computer vision applications — used in retail, manufacturing, and traffic monitoring.

```python
from ultralytics import solutions

counter = solutions.ObjectCounter(
    model="yolo26n.pt",
    show=True,
    region=[(20, 400), (1080, 400)],  # define counting line
)

counter.count(source="path/to/video.mp4")
```

**Guide:** [Object Counting](../../guides/object-counting.md) · **Task:** [Object Detection](../../tasks/detect.md)

---

### 3. Live Inference App

Build and deploy a browser-based live inference app in under 20 lines of code using Streamlit. Ideal for demos, prototypes, and sharing results without a complex deployment pipeline.

```bash
yolo streamlit-predict
```

**Guide:** [Streamlit Live Inference](../../guides/streamlit-live-inference.md) · **Mode:** [Predict](../../modes/predict.md)

---

### 4. Object Blurring & Anonymisation

Automatically blur faces, license plates, or any detected object class to anonymise video or image data. Uses segmentation masks for pixel-precise blurring rather than bounding-box blurring.

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo26n-seg.pt")
results = model("path/to/image.jpg")

for result in results:
    img = result.orig_img.copy()
    for mask in result.masks.xy:
        # apply blur within mask region
        pass
    cv2.imshow("Anonymised", img)
```

**Guide:** [Object Blurring](../../guides/object-blurring.md) · **Task:** [Instance Segmentation](../../tasks/segment.md)

---

### 5. Image Classification

Classify an entire image into a category — the simplest computer vision task and a solid foundation before moving to detection or segmentation. Fine-tune on your own dataset with a few lines of code.

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
results = model("path/to/image.jpg")
print(results[0].probs.top5)  # top-5 class predictions
```

**Task:** [Image Classification](../../tasks/classify.md) · **Mode:** [Train](../../modes/train.md)

---

### 6. Object Cropping

Detect objects and automatically crop each one into a separate image file. Useful for building datasets, extracting product images, or feeding crops into a downstream classifier.

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")
results = model("path/to/image.jpg")

for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    crop = results[0].orig_img[y1:y2, x1:x2]
    cv2.imwrite(f"crop_{i}.jpg", crop)
```

**Guide:** [Object Cropping](../../guides/object-cropping.md) · **Task:** [Object Detection](../../tasks/detect.md)

---

### 7. Visual Similarity Search

Find visually similar images in a dataset using YOLO feature embeddings and nearest-neighbour search. Useful for duplicate detection, product search, and dataset curation.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.embed(["image1.jpg", "image2.jpg", "image3.jpg"])
# results contains feature vectors for similarity comparison
```

**Guide:** [Similarity Search](../../guides/similarity-search.md)

---

## Intermediate Projects

Projects that combine multiple techniques, process video streams, or require some domain knowledge. Suitable for developers with basic Python and ML experience.

### 8. Workout & Rep Counter

Count exercise repetitions in real time by tracking joint angles from pose keypoints. Detects squats, push-ups, pull-ups, and other movements without any custom training.

```python
from ultralytics import solutions

gym = solutions.AIGym(
    model="yolo26n-pose.pt",
    kpts=[6, 8, 10],  # shoulder, elbow, wrist keypoints
    show=True,
)

gym.monitor(source="path/to/video.mp4")
```

**Guide:** [Workouts Monitoring](../../guides/workouts-monitoring.md) · **Task:** [Pose Estimation](../../tasks/pose.md)

---

### 9. Parking Management System

Monitor a car park in real time — draw parking zones, detect vehicles, and track occupancy state for each space. Deployable on a single camera covering an entire car park.

```python
from ultralytics import solutions

parking = solutions.ParkingManagement(
    model="yolo26n.pt",
    json_file="parking_regions.json",  # define parking zones
    show=True,
)

parking.monitor(source="path/to/video.mp4")
```

**Guide:** [Parking Management](../../guides/parking-management.md) · **Task:** [Object Detection](../../tasks/detect.md)

---

### 10. Queue Management System

Count and monitor people queuing in a defined region. Estimate queue length and waiting time from a single overhead or side-angle camera — no hardware sensors required.

```python
from ultralytics import solutions

queue = solutions.QueueManager(
    model="yolo26n.pt",
    region=[(50, 50), (500, 50), (500, 500), (50, 500)],
    show=True,
)

queue.monitor(source="path/to/video.mp4")
```

**Guide:** [Queue Management](../../guides/queue-management.md) · **Task:** [Object Detection](../../tasks/detect.md)

---

### 11. Speed Estimation

Estimate vehicle speed from a video stream without radar or physical sensors. Tracks objects across frames, measures pixel displacement against a real-world reference distance, and outputs speed in km/h or mph.

```python
from ultralytics import solutions

speed = solutions.SpeedEstimator(
    model="yolo26n.pt",
    region=[(0, 360), (1280, 360)],  # reference line
    show=True,
)

speed.estimate_speed(source="path/to/video.mp4")
```

**Guide:** [Speed Estimation](../../guides/speed-estimation.md) · **Mode:** [Track](../../modes/track.md)

---

### 12. Heatmap Generation

Generate density heatmaps from object movement data across video frames. Visualise where objects spend the most time — useful for retail analytics, crowd management, and sports analysis.

```python
from ultralytics import solutions

heatmap = solutions.Heatmap(
    model="yolo26n.pt",
    colormap=2,  # cv2.COLORMAP_JET
    show=True,
)

heatmap.generate(source="path/to/video.mp4")
```

**Guide:** [Heatmaps](../../guides/heatmaps.md) · **Mode:** [Track](../../modes/track.md)

---

### 13. Region & Zone Counting

Count objects entering or exiting custom polygonal regions. Useful for restricted area monitoring, entrance counting, and multi-zone analytics from a single camera.

```python
from ultralytics import solutions

region_counter = solutions.RegionCounter(
    model="yolo26n.pt",
    region=[(200, 200), (600, 200), (600, 500), (200, 500)],
    show=True,
)

region_counter.count(source="path/to/video.mp4")
```

**Guides:** [Region Counting](../../guides/region-counting.md) · [Track Zone](../../guides/trackzone.md)

---

### 14. Distance Calculation

Calculate the real-world distance between any two detected objects in a frame. Useful for social distancing monitoring, industrial safety, and robotics proximity detection.

```python
from ultralytics import solutions

distance = solutions.DistanceCalculation(
    model="yolo26n.pt",
    show=True,
)

distance.calculate(source="path/to/video.mp4")
```

**Guide:** [Distance Calculation](../../guides/distance-calculation.md) · **Task:** [Object Detection](../../tasks/detect.md)

---

### 15. Instance Segmentation with Tracking

Combine pixel-level segmentation masks with object tracking to follow the exact shape of each object across video frames. Useful for precise motion analysis and object isolation in video.

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
results = model.track(source="path/to/video.mp4", persist=True, show=True)
```

**Guide:** [Instance Segmentation with Tracking](../../guides/instance-segmentation-and-tracking.md) · **Task:** [Segmentation](../../tasks/segment.md)

---

### 16. Analytics Dashboard

Generate bar charts, line graphs, and pie charts from detection data in real time. Visualise object counts, class distributions, and trends directly from a video stream.

```python
from ultralytics import solutions

analytics = solutions.Analytics(
    analytics_type="line",
    model="yolo26n.pt",
    show=True,
)

analytics.process(source="path/to/video.mp4")
```

**Guide:** [Analytics](../../guides/analytics.md) · **Task:** [Object Detection](../../tasks/detect.md)

---

### 17. Vision Eye — Perspective Mapping

Map detected object positions from camera perspective onto a bird's-eye view using homography. Useful for traffic management, sports analytics, and any application where top-down spatial reasoning matters.

```python
from ultralytics import solutions

vision_eye = solutions.VisionEye(
    model="yolo26n.pt",
    show=True,
)

vision_eye.process(source="path/to/video.mp4")
```

**Guide:** [Vision Eye](../../guides/vision-eye.md) · **Task:** [Object Detection](../../tasks/detect.md)

---

### 18. Pose Estimation & Ergonomics

Detect human body keypoints and analyse joint angles to monitor posture, flag unsafe working positions, or track athletic performance. Works on individuals or groups without any wearables.

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
results = model("path/to/image.jpg")

for result in results:
    keypoints = result.keypoints.xy  # (x, y) for each keypoint
    print(keypoints)
```

**Task:** [Pose Estimation](../../tasks/pose.md) · **Guide:** [Workouts Monitoring](../../guides/workouts-monitoring.md)

---

## Advanced Projects

Production-grade implementations that require custom datasets, model fine-tuning, or specialised hardware. Suitable for engineers and researchers building deployable systems.

### 19. Rotated Object Detection (OBB)

Detect objects at arbitrary rotation angles using oriented bounding boxes — critical for aerial and satellite imagery where objects appear at any orientation. Significantly more accurate than axis-aligned boxes for elongated objects.

```python
from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")
results = model("path/to/aerial-image.jpg")
results[0].show()  # displays rotated bounding boxes
```

**Task:** [OBB Detection](../../tasks/obb.md)

---

### 20. Tiled Inference for Large Images

Run inference on very large images (satellite imagery, medical scans, industrial inspection) by slicing them into overlapping tiles, running inference on each tile, and merging results. Uses SAHI integration.

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.3,
)

result = get_sliced_prediction(
    "path/to/large-image.jpg",
    model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

**Guide:** [SAHI Tiled Inference](../../guides/sahi-tiled-inference.md)

---

### 21. Open-Vocabulary Detection

Detect any object described in natural language — no retraining required. YOLO-World accepts text prompts at inference time, making it ideal for zero-shot detection of novel categories.

```python
from ultralytics import YOLOWorld

model = YOLOWorld("yoloworld-l.pt")
model.set_classes(["person", "forklift", "hard hat"])  # any text prompt

results = model("path/to/image.jpg")
results[0].show()
```

**Model:** [YOLO-World](../../models/yolo-world.md)

---

### 22. Zero-Shot Segmentation

Segment any object using point, box, or text prompts — no training on your specific objects required. SAM 2 produces high-quality masks interactively or in batch mode.

```python
from ultralytics import SAM

model = SAM("sam2.1_l.pt")

# prompt with bounding box
results = model("path/to/image.jpg", bboxes=[[100, 100, 400, 400]])
results[0].show()
```

**Model:** [SAM 2](../../models/sam-2.md) · [SAM](../../models/sam.md)

---

### 23. Defect & Anomaly Detection

Train a custom YOLO model to detect defects, damage, or anomalies specific to your product or environment. Requires a labelled dataset but can reach production accuracy with as few as 200-500 images.

```python
from ultralytics import YOLO

# train on your custom defect dataset
model = YOLO("yolo26n.pt")
model.train(
    data="defects.yaml",
    epochs=100,
    imgsz=640,
)

# run inference
results = model("path/to/product-image.jpg")
```

**Guide:** [Steps of a CV Project](../../guides/steps-of-a-cv-project.md) · **Mode:** [Train](../../modes/train.md)

---

### 24. Edge Deployment — Jetson & Raspberry Pi

Export a trained YOLO model to TensorRT (Jetson) or ONNX/TFLite (Raspberry Pi) and run real-time inference on embedded hardware. Achieves 30+ FPS on Jetson Orin with TensorRT.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# export for Jetson (TensorRT)
model.export(format="engine", device=0)

# export for Raspberry Pi (ONNX)
model.export(format="onnx")
```

**Guides:** [NVIDIA Jetson](../../guides/nvidia-jetson.md) · [Raspberry Pi](../../guides/raspberry-pi.md) · **Mode:** [Export](../../modes/export.md)

---

### 25. Fine-Tune on Custom Dataset

Adapt a pretrained YOLO26 model to detect your own object classes using transfer learning. Works with datasets from Roboflow, Label Studio, or any YOLO-format annotation tool.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # start from pretrained weights

model.train(
    data="custom-dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.01,
)

# validate on test set
metrics = model.val()
print(metrics.box.map)  # mAP50-95
```

**Guide:** [Steps of a CV Project](../../guides/steps-of-a-cv-project.md) · **Mode:** [Train](../../modes/train.md)

---

## Choose by Technique

| Technique | Beginner project | Intermediate project | Advanced project |
|---|---|---|---|
| Object Detection | [Object Counting](#2-object-counting) | [Speed Estimation](#11-speed-estimation) | [Defect Detection](#23-defect--anomaly-detection) |
| Instance Segmentation | [Object Blurring](#4-object-blurring--anonymisation) | [Segmentation + Tracking](#15-instance-segmentation-with-tracking) | [Tiled Inference](#20-tiled-inference-for-large-images) |
| Image Classification | [Image Classification](#5-image-classification) | — | [Fine-Tune on Custom Data](#25-fine-tune-on-custom-dataset) |
| Pose Estimation | [Pose Estimation](#18-pose-estimation--ergonomics) | [Workout Counter](#8-workout--rep-counter) | — |
| Object Tracking | [Object Counting](#2-object-counting) | [Heatmap Generation](#12-heatmap-generation) | — |
| OBB Detection | — | — | [Rotated Object Detection](#19-rotated-object-detection-obb) |
| Foundation Models | — | — | [Open-Vocabulary Detection](#21-open-vocabulary-detection) · [SAM 2](#22-zero-shot-segmentation) |

---

## FAQ

??? question "What is the easiest computer vision project for beginners?"

    [Object Counting](#2-object-counting) and the [Security Alarm System](#1-security-alarm-system) are the best starting points — both run with a single pretrained model, require no training, and produce visible results immediately. Install Ultralytics, run the code snippet, and you have a working project in under 5 minutes.

??? question "Do I need a GPU to run these projects?"

    No. All beginner projects run on CPU. A GPU significantly improves speed for real-time video inference (30+ FPS vs 5-10 FPS on CPU). For edge deployment, Jetson and Raspberry Pi projects are designed specifically for embedded hardware. See the [NVIDIA Jetson guide](../../guides/nvidia-jetson.md) and [Raspberry Pi guide](../../guides/raspberry-pi.md).

??? question "What is the best YOLO model to start with?"

    [YOLO26n](../../models/yolo26.md) (nano) — fastest, smallest, runs on CPU and edge devices. Scale up to YOLO26s or YOLO26m when you need more accuracy. See the [model comparison](../../models/index.md) for full benchmarks across speed and accuracy.

??? question "How many images do I need to train a custom model?"

    As few as 50-100 images per class for simple objects in consistent conditions. For production accuracy across varied conditions, aim for 500-1000 images per class. The [steps of a CV project guide](../../guides/steps-of-a-cv-project.md) covers data collection, annotation, and training configuration.

??? question "What are good computer vision project ideas for beginners?"

    Start with projects that use a pretrained model and produce immediate visual output — [object counting](#2-object-counting), [security alarm system](#1-security-alarm-system), [object blurring](#4-object-blurring--anonymisation), or the [live inference app](#3-live-inference-app). These require no dataset preparation or training and give you hands-on experience with detection pipelines, model outputs, and real-time inference before you tackle custom training.

??? question "How do I deploy a model to production?"

    See the [model deployment options guide](../../guides/model-deployment-options.md) for a comparison of TensorRT, ONNX, TFLite, CoreML, and other export formats. For scalable serving, see the [Triton Inference Server guide](../../guides/triton-inference-server.md). For embedded deployment, see [NVIDIA Jetson](../../guides/nvidia-jetson.md) and [Raspberry Pi](../../guides/raspberry-pi.md).
