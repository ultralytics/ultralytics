---
comments: true
description: Learn how to implement YOLO11 with SAHI for sliced inference. Optimize memory usage and enhance detection accuracy for large-scale applications.
keywords: YOLO11, SAHI, Sliced Inference, Object Detection, Ultralytics, High-resolution Images, Computational Efficiency, Integration Guide
---

# Ultralytics Docs: Using YOLO11 with SAHI for Sliced Inference

Welcome to the Ultralytics documentation on how to use YOLO11 with [SAHI](https://github.com/obss/sahi) (Slicing Aided Hyper Inference). This comprehensive guide aims to furnish you with all the essential knowledge you'll need to implement SAHI alongside YOLO11. We'll deep-dive into what SAHI is, why sliced inference is critical for large-scale applications, and how to integrate these functionalities with YOLO11 for enhanced [object detection](https://www.ultralytics.com/glossary/object-detection) performance.

<p align="center">
  <img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/sahi-sliced-inference-overview.avif" alt="SAHI Sliced Inference Overview">
</p>

## Introduction to SAHI

SAHI (Slicing Aided Hyper Inference) is an innovative library designed to optimize object detection algorithms for large-scale and high-resolution imagery. Its core functionality lies in partitioning images into manageable slices, running object detection on each slice, and then stitching the results back together. SAHI is compatible with a range of object detection models, including the YOLO series, thereby offering flexibility while ensuring optimized use of computational resources.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/tq3FU_QczxE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Inference with SAHI (Slicing Aided Hyper Inference) using Ultralytics YOLO11
</p>

### Key Features of SAHI

- **Seamless Integration**: SAHI integrates effortlessly with YOLO models, meaning you can start slicing and detecting without a lot of code modification.
- **Resource Efficiency**: By breaking down large images into smaller parts, SAHI optimizes the memory usage, allowing you to run high-quality detection on hardware with limited resources.
- **High [Accuracy](https://www.ultralytics.com/glossary/accuracy)**: SAHI maintains the detection accuracy by employing smart algorithms to merge overlapping detection boxes during the stitching process.

## What is Sliced Inference?

Sliced Inference refers to the practice of subdividing a large or high-resolution image into smaller segments (slices), conducting object detection on these slices, and then recompiling the slices to reconstruct the object locations on the original image. This technique is invaluable in scenarios where computational resources are limited or when working with extremely high-resolution images that could otherwise lead to memory issues.

### Benefits of Sliced Inference

- **Reduced Computational Burden**: Smaller image slices are faster to process, and they consume less memory, enabling smoother operation on lower-end hardware.

- **Preserved Detection Quality**: Since each slice is treated independently, there is no reduction in the quality of object detection, provided the slices are large enough to capture the objects of interest.

- **Enhanced Scalability**: The technique allows for object detection to be more easily scaled across different sizes and resolutions of images, making it ideal for a wide range of applications from satellite imagery to medical diagnostics.

<table border="0">
  <tr>
    <th>YOLO11 without SAHI</th>
    <th>YOLO11 with SAHI</th>
  </tr>
  <tr>
    <td><img src="https://github.com/ultralytics/docs/releases/download/0/yolov8-without-sahi.avif" alt="YOLO11 without SAHI" width="640"></td>
    <td><img src="https://github.com/ultralytics/docs/releases/download/0/yolov8-with-sahi.avif" alt="YOLO11 with SAHI" width="640"></td>
  </tr>
</table>

## Installation and Preparation

### Installation

To get started, install the latest versions of SAHI and Ultralytics:

```bash
pip install -U ultralytics sahi
```

### Import Modules and Download Resources

Here's how to import the necessary modules and download a YOLO11 model and some test images:

```python
from sahi.utils.file import download_from_url
from sahi.utils.ultralytics import download_yolo11n_model

# Download YOLO11 model
model_path = "models/yolo11n.pt"
download_yolo11n_model(model_path)

# Download test images
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    "demo_data/terrain2.png",
)
```

## Standard Inference with YOLO11

### Instantiate the Model

You can instantiate a YOLO11 model for object detection like this:

```python
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)
```

### Perform Standard Prediction

Perform standard inference using an image path or a numpy image.

```python
from sahi.predict import get_prediction

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

## Sliced Inference with YOLO11

Perform sliced inference by specifying the slice dimensions and overlap ratios:

```python
from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
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
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="path/to/yolo11n.pt",
    model_device="cpu",  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source="path/to/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

That's it! Now you're equipped to use YOLO11 with SAHI for both standard and sliced inference.

## Citations and Acknowledgments

If you use SAHI in your research or development work, please cite the original SAHI paper and acknowledge the authors:

!!! quote ""

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

We extend our thanks to the SAHI research group for creating and maintaining this invaluable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about SAHI and its creators, visit the [SAHI GitHub repository](https://github.com/obss/sahi).

## FAQ

### How can I integrate YOLO11 with SAHI for sliced inference in object detection?

Integrating Ultralytics YOLO11 with SAHI (Slicing Aided Hyper Inference) for sliced inference optimizes your object detection tasks on high-resolution images by partitioning them into manageable slices. This approach improves memory usage and ensures high detection accuracy. To get started, you need to install the ultralytics and sahi libraries:

```bash
pip install -U ultralytics sahi
```

Then, download a YOLO11 model and test images:

```python
from sahi.utils.file import download_from_url
from sahi.utils.ultralytics import download_yolo11n_model

# Download YOLO11 model
model_path = "models/yolo11n.pt"
download_yolo11n_model(model_path)

# Download test images
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
```

For more detailed instructions, refer to our [Sliced Inference guide](#sliced-inference-with-yolo11).

### Why should I use SAHI with YOLO11 for object detection on large images?

Using SAHI with Ultralytics YOLO11 for object detection on large images offers several benefits:

- **Reduced Computational Burden**: Smaller slices are faster to process and consume less memory, making it feasible to run high-quality detections on hardware with limited resources.
- **Maintained Detection Accuracy**: SAHI uses intelligent algorithms to merge overlapping boxes, preserving the detection quality.
- **Enhanced Scalability**: By scaling object detection tasks across different image sizes and resolutions, SAHI becomes ideal for various applications, such as satellite imagery analysis and medical diagnostics.

Learn more about the [benefits of sliced inference](#benefits-of-sliced-inference) in our documentation.

### Can I visualize prediction results when using YOLO11 with SAHI?

Yes, you can visualize prediction results when using YOLO11 with SAHI. Here's how you can export and visualize the results:

```python
from IPython.display import Image

result.export_visuals(export_dir="demo_data/")
Image("demo_data/prediction_visual.png")
```

This command will save the visualized predictions to the specified directory, and you can then load the image to view it in your notebook or application. For a detailed guide, check out the [Standard Inference section](#visualize-results).

### What features does SAHI offer for improving YOLO11 object detection?

SAHI (Slicing Aided Hyper Inference) offers several features that complement Ultralytics YOLO11 for object detection:

- **Seamless Integration**: SAHI easily integrates with YOLO models, requiring minimal code adjustments.
- **Resource Efficiency**: It partitions large images into smaller slices, which optimizes memory usage and speed.
- **High Accuracy**: By effectively merging overlapping detection boxes during the stitching process, SAHI maintains high detection accuracy.

For a deeper understanding, read about SAHI's [key features](#key-features-of-sahi).

### How do I handle large-scale inference projects using YOLO11 and SAHI?

To handle large-scale inference projects using YOLO11 and SAHI, follow these best practices:

1. **Install Required Libraries**: Ensure that you have the latest versions of ultralytics and sahi.
2. **Configure Sliced Inference**: Determine the optimal slice dimensions and overlap ratios for your specific project.
3. **Run Batch Predictions**: Use SAHI's capabilities to perform batch predictions on a directory of images, which improves efficiency.

Example for batch prediction:

```python
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="path/to/yolo11n.pt",
    model_device="cpu",  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source="path/to/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

For more detailed steps, visit our section on [Batch Prediction](#batch-prediction).
