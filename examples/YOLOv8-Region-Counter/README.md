# Regions Counting Using Ultralytics YOLOv8 (Inference on Video)

> **Note:** Region Counter is now part of **[Ultralytics Solutions](https://docs.ultralytics.com/solutions/)**, offering enhanced features and ongoing updates.
>
> 🔗 **Explore the official [Region Counting Guide](https://docs.ultralytics.com/guides/region-counting/) for the latest implementation.**

> 🔔 **Notice:**
>
> This GitHub example (`ultralytics/examples/YOLOv8-Region-Counter/`) will remain available but **is no longer actively maintained**. For the most current features, updates, and support, please refer to the official [Region Counting guide](https://docs.ultralytics.com/guides/region-counting/) within the Ultralytics documentation. Thank you!

Region counting is a technique used to count objects within predefined areas or zones in a video feed. This allows for more detailed analysis, especially when monitoring multiple distinct areas simultaneously. Users can interactively adjust these regions by clicking and dragging with the left mouse button, enabling real-time counting tailored to specific needs and layouts. This method is valuable in various [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications, from traffic analysis to retail analytics.

<div>
<p align="center">
  <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/5ab3bbd7-fd12-4849-928e-5f294d6c3fcf" width="45%" alt="YOLOv8 region counting visual 1">
  <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/e7c1aea7-474d-4d78-8d48-b50854ffe1ca" width="45%" alt="YOLOv8 region counting visual 2">
</p>
</div>

## 📚 Table of Contents

- [Step 1: Install the Required Libraries](#-step-1-install-the-required-libraries)
- [Step 2: Run Region Counting with Ultralytics YOLOv8](#-step-2-run-region-counting-with-ultralytics-yolov8)
- [Usage Options](#-usage-options)
- [Frequently Asked Questions (FAQ)](#-frequently-asked-questions-faq)
- [Contributing](#-contributing)

## ⚙️ Step 1: Install the Required Libraries

First, clone the Ultralytics repository and navigate to the example directory. Ensure you have Python installed along with necessary dependencies like [PyTorch](https://pytorch.org/) and [OpenCV](https://opencv.org/).

```bash
# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Navigate to the example directory
cd ultralytics/examples/YOLOv8-Region-Counter

# Install required packages (if not already installed)
pip install ultralytics shapely
```

## ▶️ Step 2: Run Region Counting with Ultralytics YOLOv8

Execute the script using the following commands. You can customize the source, model, device, and other parameters.

### Note

Once the video starts playing, you can dynamically reposition the counting regions within the video frame by clicking and dragging them with your left mouse button.

```bash
# Run inference on a video source, saving results and viewing output
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --view-img

# Run inference using the CPU
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --view-img --device cpu

# Use a specific Ultralytics YOLOv8 model file
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --weights "path/to/yolov8n.pt"

# Detect only specific classes (e.g., class 0 and class 2)
python yolov8_region_counter.py --source "path/to/video.mp4" --classes 0 2 --weights "path/to/yolov8m.pt"

# Run inference without saving the output video/images
python yolov8_region_counter.py --source "path/to/video.mp4" --view-img
```

Learn more about inference arguments in the Ultralytics [Predict Mode documentation](https://docs.ultralytics.com/modes/predict/).

## 🛠️ Usage Options

The script accepts several command-line arguments for customization:

- `--source`: Path to the input video file.
- `--device`: Computation device (`cpu` or GPU ID like `0`).
- `--save-img`: Boolean flag to save output frames with detections.
- `--view-img`: Boolean flag to display the output video stream in real-time.
- `--weights`: Path to the [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) model file (`.pt`). Defaults typically use a standard model like `yolov8n.pt`.
- `--classes`: Filter detections by specific class IDs (e.g., `--classes 0 2 3` to detect classes 0, 2, and 3).
- `--line-thickness`: Thickness of the [bounding box](https://www.ultralytics.com/glossary/bounding-box) lines.
- `--region-thickness`: Thickness of the lines defining the counting regions.
- `--track-thickness`: Thickness of the object tracking lines.

Explore different models and training options in the [Ultralytics documentation](https://docs.ultralytics.com/).

## ❓ Frequently Asked Questions (FAQ)

### What is Region Counting?

Region counting is a process in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) used to determine the number of objects within predefined areas of an image or video frame. It's commonly applied in fields like [image processing](https://en.wikipedia.org/wiki/Image_processing) and [pattern recognition](https://en.wikipedia.org/wiki/Pattern_recognition) for analyzing spatial distributions and segmenting objects based on location.

### Does the Region Counter Support Custom Region Shapes?

Yes, the Region Counter allows defining regions using polygons (including rectangles). You can customize region properties like coordinates, names, and colors directly in the script. The `shapely` library is used for polygon definitions. See the [Shapely User Manual](https://shapely.readthedocs.io/en/stable/manual.html#polygons) for more details on polygon creation.

```python
from shapely.geometry import Polygon

# Example definition of counting regions
counting_regions = [
    {
        "name": "Region 1 (Pentagon)",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # 5-point polygon
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR color for region
        "text_color": (255, 255, 255),  # BGR color for text
    },
    {
        "name": "Region 2 (Rectangle)",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # 4-point polygon (rectangle)
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR color for region
        "text_color": (0, 0, 0),  # BGR color for text
    },
]
```

### Why Combine Region Counting with YOLOv8?

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) excels at [object detection](https://www.ultralytics.com/glossary/object-detection) and [tracking](https://www.ultralytics.com/glossary/object-tracking) in video streams. Integrating region counting enhances its capabilities by enabling object quantification within specific zones, making it useful for applications like crowd monitoring, traffic flow analysis, and retail footfall counting. Check out our blog post on [Object Detection and Tracking with Ultralytics YOLOv8](https://www.ultralytics.com/blog/object-detection-and-tracking-with-ultralytics-yolov8).

### How Can I Troubleshoot Issues?

For debugging, you can enable more verbose output. While this specific script doesn't have a dedicated `--debug` flag, you can add print statements within the code to inspect variables or use standard Python debugging tools. Ensure your video path and model weights path are correct. For common issues, refer to the [Ultralytics FAQ](https://docs.ultralytics.com/help/FAQ/).

### Can I Use Other YOLO Versions or Custom Models?

Yes, you can use different Ultralytics YOLO model versions (like YOLOv5, YOLOv9, YOLOv10, YOLO11) or your own custom-trained models by specifying the path to the `.pt` file using the `--weights` argument. Ensure the model is compatible with the Ultralytics framework. Find more about training custom models in the [Model Training guide](https://docs.ultralytics.com/modes/train/).

## 🤝 Contributing

Contributions to improve this example or add new features are welcome! Please feel free to submit Pull Requests or open Issues on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics). Remember to check the official [Region Counting guide](https://docs.ultralytics.com/guides/region-counting/) for the latest maintained version.
