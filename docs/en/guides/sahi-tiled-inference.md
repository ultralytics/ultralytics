---
comments: true
description: A comprehensive guide on how to use YOLOv8 with SAHI for standard and sliced inference in object detection tasks.
keywords: YOLOv8, SAHI, Sliced Inference, Object Detection, Ultralytics, Large Scale Image Analysis, High-Resolution Imagery
---

# Ultralytics Docs: Using YOLOv8 with SAHI for Sliced Inference

Welcome to the Ultralytics documentation on how to use YOLOv8 with [SAHI](https://github.com/obss/sahi) (Slicing Aided Hyper Inference). This comprehensive guide aims to furnish you with all the essential knowledge you'll need to implement SAHI alongside YOLOv8. We'll deep-dive into what SAHI is, why sliced inference is critical for large-scale applications, and how to integrate these functionalities with YOLOv8 for enhanced object detection performance.

<p align="center">
  <img width="1024" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif" alt="SAHI Sliced Inference Overview">
</p>

## Introduction to SAHI

SAHI (Slicing Aided Hyper Inference) is an innovative library designed to optimize object detection algorithms for large-scale and high-resolution imagery. Its core functionality lies in partitioning images into manageable slices, running object detection on each slice, and then stitching the results back together. SAHI is compatible with a range of object detection models, including the YOLO series, thereby offering flexibility while ensuring optimized use of computational resources.

### Key Features of SAHI

- **Seamless Integration**: SAHI integrates effortlessly with YOLO models, meaning you can start slicing and detecting without a lot of code modification.
- **Resource Efficiency**: By breaking down large images into smaller parts, SAHI optimizes the memory usage, allowing you to run high-quality detection on hardware with limited resources.
- **High Accuracy**: SAHI maintains the detection accuracy by employing smart algorithms to merge overlapping detection boxes during the stitching process.

## What is Sliced Inference?

Sliced Inference refers to the practice of subdividing a large or high-resolution image into smaller segments (slices), conducting object detection on these slices, and then recompiling the slices to reconstruct the object locations on the original image. This technique is invaluable in scenarios where computational resources are limited or when working with extremely high-resolution images that could otherwise lead to memory issues.

### Benefits of Sliced Inference

- **Reduced Computational Burden**: Smaller image slices are faster to process, and they consume less memory, enabling smoother operation on lower-end hardware.

- **Preserved Detection Quality**: Since each slice is treated independently, there is no reduction in the quality of object detection, provided the slices are large enough to capture the objects of interest.

- **Enhanced Scalability**: The technique allows for object detection to be more easily scaled across different sizes and resolutions of images, making it ideal for a wide range of applications from satellite imagery to medical diagnostics.

<table border="0">
  <tr>
    <th>YOLOv8 without SAHI</th>
    <th>YOLOv8 with SAHI</th>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/26833433/266123241-260a9740-5998-4e9a-ad04-b39b7767e731.png" alt="YOLOv8 without SAHI" width="640"></td>
    <td><img src="https://user-images.githubusercontent.com/26833433/266123245-55f696ad-ec74-4e71-9155-c211d693bb69.png" alt="YOLOv8 with SAHI" width="640"></td>
  </tr>
</table>

## Installation and Preparation

### Installation

To get started, install the latest versions of SAHI and Ultralytics:

```bash
pip install -U ultralytics sahi
```

### Import Modules and Download Resources

Here's how to import the necessary modules and download a YOLOv8 model and some test images:

```python
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
from IPython.display import Image

# Download YOLOv8 model
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

# Download test images
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')
```

## Standard Inference with YOLOv8

### Instantiate the Model

You can instantiate a YOLOv8 model for object detection like this:

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)
```

### Perform Standard Prediction

Perform standard inference using an image path or a numpy image.

```python
# With an image path
result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)

# With a numpy image
result = get_prediction(read_image("demo_data/small-vehicles1.jpeg"), detection_model)
```

### Visualize Results

Export and visualize the predicted bounding boxes and masks:

```python
result.export_visuals(export_dir="demo_data/")
Image("demo_data/prediction_visual.png")
```

## Sliced Inference with YOLOv8

Perform sliced inference by specifying the slice dimensions and overlap ratios:

```python
result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
```

## Handling Prediction Results

SAHI provides a `PredictionResult` object, which can be converted into various annotation formats:

```python
# Access the object prediction list
object_prediction_list = result.object_prediction_list

# Convert to COCO annotation, COCO prediction, imantics, and fiftyone formats
result.to_coco_annotations()[:3]
result.to_coco_predictions(image_id=1)[:3]
result.to_imantics_annotations()[:3]
result.to_fiftyone_detections()[:3]
```

## Batch Prediction

For batch prediction on a directory of images:

```python
predict(
    model_type="yolov8",
    model_path="path/to/yolov8n.pt",
    model_device="cpu",  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source="path/to/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

That's it! Now you're equipped to use YOLOv8 with SAHI for both standard and sliced inference.

## Citations and Acknowledgments

If you use SAHI in your research or development work, please cite the original SAHI paper and acknowledge the authors:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{akyon2022sahi,
          title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
          author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
          journal={2022 IEEE International Conference on Image Processing (ICIP)},
          doi={10.1109/ICIP46576.2022.9897990},
          pages={966-970},
          year={2022}
        }
        ```

We extend our thanks to the SAHI research group for creating and maintaining this invaluable resource for the computer vision community. For more information about SAHI and its creators, visit the [SAHI GitHub repository](https://github.com/obss/sahi).
