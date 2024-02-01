---
comments: true
description: Object Counting in Different Region using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Object Counting, Object Tracking, Notebook, IPython Kernel, CLI, Python SDK
---

# Object Counting in Different Regions using Ultralytics YOLOv8 🚀

## What is Object Counting in Regions?

[Object counting](https://docs.ultralytics.com/guides/object-counting/) in regions with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) involves precisely determining the number of objects within specified areas using advanced computer vision. This approach is valuable for optimizing processes, enhancing security, and improving efficiency in various applications.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/okItf1iHlV8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLOv8 Object Counting in Multiple & Movable Regions
</p>

## Advantages of Object Counting in Regions?

- **Precision and Accuracy:** Object counting in regions with advanced computer vision ensures precise and accurate counts, minimizing errors often associated with manual counting.
- **Efficiency Improvement:** Automated object counting enhances operational efficiency, providing real-time results and streamlining processes across different applications.
- **Versatility and Application:** The versatility of object counting in regions makes it applicable across various domains, from manufacturing and surveillance to traffic monitoring, contributing to its widespread utility and effectiveness.

## Real World Applications

|                                                                               Retail                                                                               |                                                                          Market Streets                                                                           |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![People Counting in Different Region using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/5ab3bbd7-fd12-4849-928e-5f294d6c3fcf) | ![Crowd Counting in Different Region using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/e7c1aea7-474d-4d78-8d48-b50854ffe1ca) |
|                                                    People Counting in Different Region using Ultralytics YOLOv8                                                    |                                                    Crowd Counting in Different Region using Ultralytics YOLOv8                                                    |

## Steps to Run

### Step 1: Install Required Libraries

Begin by cloning the Ultralytics repository, installing dependencies, and navigating to the local directory using the provided commands in Step 2.

```bash
# Clone Ultralytics repo
git clone https://github.com/ultralytics/ultralytics

# Navigate to the local directory
cd ultralytics/examples/YOLOv8-Region-Counter
```

### Step 2: Run Region Counting Using Ultralytics YOLOv8

Execute the following basic commands for inference.

???+ tip "Region is Movable"

    During video playback, you can interactively move the region within the video by clicking and dragging using the left mouse button.

```bash
# Save results
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img

# Run model on CPU
python yolov8_region_counter.py --source "path/to/video.mp4" --device cpu

# Change model file
python yolov8_region_counter.py --source "path/to/video.mp4" --weights "path/to/model.pt"

# Detect specific classes (e.g., first and third classes)
python yolov8_region_counter.py --source "path/to/video.mp4" --classes 0 2

# View results without saving
python yolov8_region_counter.py --source "path/to/video.mp4" --view-img
```

### Optional Arguments

| Name                 | Type   | Default      | Description                                |
|----------------------|--------|--------------|--------------------------------------------|
| `--source`           | `str`  | `None`       | Path to video file, for webcam 0           |
| `--line_thickness`   | `int`  | `2`          | Bounding Box thickness                     |
| `--save-img`         | `bool` | `False`      | Save the predicted video/image             |
| `--weights`          | `str`  | `yolov8n.pt` | Weights file path                          |
| `--classes`          | `list` | `None`       | Detect specific classes i.e. --classes 0 2 |
| `--region-thickness` | `int`  | `2`          | Region Box thickness                       |
| `--track-thickness`  | `int`  | `2`          | Tracking line thickness                    |
