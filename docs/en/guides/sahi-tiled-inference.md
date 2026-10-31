---
title: YOLO26 Sliced Inference with SAHI
comments: true
description: Run Ultralytics YOLO26 with SAHI sliced inference to detect small objects in high-resolution images, with slicing parameters and batch prediction examples.
keywords: YOLO26, SAHI, Sliced Inference, Object Detection, Ultralytics, High-resolution Images, Computational Efficiency, Integration Guide
---

# Using YOLO26 with SAHI for Sliced Inference

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-ultralytics-yolo-with-sahi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open SAHI for Sliced Inference In Colab"></a>

This guide shows how to use YOLO26 with [SAHI](https://github.com/obss/sahi) (Slicing Aided Hyper Inference). It covers what SAHI is, why sliced inference helps [detect small objects](https://www.ultralytics.com/blog/exploring-small-object-detection-with-ultralytics-yolo11) in large, high-resolution images, and how to run it with YOLO26 for better [object detection](https://www.ultralytics.com/glossary/object-detection) results.

<p align="center">
  <video width="1024" src="https://cdn.ul.run/v/875e435c815f7be5e42d0d1af5627644.mp4" autoplay loop muted playsinline aria-label="SAHI tiled inference for small objects"></video>
</p>

## Introduction to SAHI

SAHI (Slicing Aided Hyper Inference) is a library designed to improve object detection on large-scale and high-resolution imagery. It partitions images into manageable slices, runs object detection on each slice, and then stitches the results back together. SAHI is compatible with a range of object detection models, including the YOLO series.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Kpl6cOUKbb8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use SAHI with Ultralytics YOLO26 to Detect Small Objects | Slicing Aided Hyper Inference 🚀
</p>

### Key Features of SAHI

- **Simple Integration**: SAHI works with Ultralytics YOLO models through `AutoDetectionModel`, so you can start slicing and detecting with little code modification.
- **Resource Efficiency**: By breaking down large images into smaller parts, SAHI keeps memory usage low, allowing you to run high-quality detection on hardware with limited resources.
- **High [Accuracy](https://www.ultralytics.com/glossary/accuracy)**: SAHI merges overlapping detections from adjacent slices during stitching, using postprocessing such as [non-maximum suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) or non-maximum merging.

## What is Sliced Inference?

Sliced Inference refers to the practice of subdividing a large or high-resolution image into smaller segments (slices), conducting object detection on these slices, and then mapping the detections back to object locations on the original image. This technique is useful when computational resources are limited, when working with extremely high-resolution images that could otherwise lead to memory issues, or when objects are too small to detect after the full image is downscaled to the model input size.

### Benefits of Sliced Inference

- **Reduced Memory Burden**: Each slice is small, so peak memory stays low even for very large images, enabling smoother operation on lower-end hardware.

- **Better Small-Object Detection**: Each slice is processed at the model's input resolution, so small objects occupy more pixels and are easier to detect, provided the slices are large enough to capture the objects of interest.

- **Enhanced Scalability**: The technique allows for object detection to be more easily scaled across different sizes and resolutions of images, making it suitable for a wide range of applications from [satellite imagery](https://www.ultralytics.com/glossary/satellite-image-analysis) to medical diagnostics.

<table border="0">
  <tr>
    <th>YOLO26 without SAHI</th>
    <th>YOLO26 with SAHI</th>
  </tr>
  <tr>
    <td><img src="https://cdn.ul.run/i/1865c0db3cd5e950fa75d07c6f5f38bb.avif" alt="YOLO26 detections on a high-resolution image without SAHI slicing" width="640"></td>
    <td><img src="https://cdn.ul.run/i/9f337a31d7085922645a191325a4a419.avif" alt="YOLO26 detections on a high-resolution image with SAHI sliced inference" width="640"></td>
  </tr>
</table>

## Installation and Preparation

### Installation

To get started, install the latest versions of SAHI and Ultralytics:

```bash
pip install -U ultralytics sahi
```

### Import Modules and Download Resources

Here's how to download some test images:

```python
from sahi.utils.file import download_from_url

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

## Standard Inference with YOLO26

### Instantiate the Model

You can instantiate a YOLO26 model for object detection like this:

```python
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)
```

### Perform Standard Prediction

Perform standard inference using an image path.

```python
from sahi.predict import get_prediction

result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)

result.export_visuals(export_dir="demo_data/", hide_conf=True)
```

### Visualize Results

Export and visualize the predicted bounding boxes and masks:

```python
from PIL import Image

# Open the predicted image
processed_image = Image.open("demo_data/prediction_visual.png")

# Display the predicted image
processed_image.show()
```

## Sliced Inference with YOLO26

Perform sliced inference by specifying the slice dimensions and overlap ratios:

```python
from PIL import Image
from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Export results
result.export_visuals(export_dir="demo_data/", hide_conf=True)

# Open the predicted image
processed_image = Image.open("demo_data/prediction_visual.png")

# Display the predicted image
processed_image.show()
```

## Handling Prediction Results

SAHI provides a `PredictionResult` object, which can be converted into various annotation formats:

```python
# Access the object prediction list
object_prediction_list = result.object_prediction_list

# Convert to COCO annotation and COCO prediction formats
result.to_coco_annotations()[:3]
result.to_coco_predictions(image_id=1)[:3]
```

!!! tip "Additional export formats"

    `PredictionResult` can also convert detections to [imantics](https://github.com/jsbroks/imantics) and [FiftyOne](https://github.com/voxel51/fiftyone) objects with `result.to_imantics_annotations()` and `result.to_fiftyone_detections()`. These methods require the respective packages, so install them first with `pip install imantics fiftyone`.

## Batch Prediction

For batch prediction on a directory of images:

```python
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    model_device="cpu",  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source="path/to/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

You are now ready to use YOLO26 with SAHI for both standard and sliced inference.

## Citations and Acknowledgments

If you use SAHI in your research or development work, please cite the original SAHI paper and acknowledge the authors:

!!! quote "Citation"

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

We extend our thanks to the SAHI research group for creating and maintaining this resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about SAHI and its creators, visit the [SAHI GitHub repository](https://github.com/obss/sahi).

## FAQ

### How can I integrate YOLO26 with SAHI for sliced inference in object detection?

Integrating Ultralytics YOLO26 with SAHI (Slicing Aided Hyper Inference) for [sliced inference](#sliced-inference-with-yolo26) improves object detection on high-resolution images by partitioning them into manageable slices. This approach helps detect small objects while keeping memory usage manageable. To get started, you need to install the ultralytics and sahi libraries:

```bash
pip install -U ultralytics sahi
```

Then, download test images:

```python
from sahi.utils.file import download_from_url

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

### Why should I use SAHI with YOLO26 for object detection on large images?

Using SAHI with Ultralytics YOLO26 for object detection on large images offers several benefits:

- **Reduced Memory Burden**: Smaller slices consume less memory, making it feasible to run high-quality detections on hardware with limited resources.
- **Better Small-Object Detection**: Slices keep small objects large relative to the model input, and SAHI merges overlapping boxes from adjacent slices.
- **Enhanced Scalability**: By scaling object detection tasks across different image sizes and resolutions, SAHI is suitable for various applications, such as satellite imagery analysis and medical diagnostics.

Learn more about the [benefits of sliced inference](#benefits-of-sliced-inference) in our documentation.

### Can I visualize prediction results when using YOLO26 with SAHI?

Yes, you can visualize prediction results when using YOLO26 with SAHI. Here's how you can export and visualize the results:

```python
from PIL import Image

result.export_visuals(export_dir="demo_data/", hide_conf=True)

processed_image = Image.open("demo_data/prediction_visual.png")

processed_image.show()
```

This command will save the visualized predictions to the specified directory, and you can then load the image to view it in your notebook or application. For a detailed guide, check out the [Standard Inference section](#visualize-results).

### What features does SAHI offer for improving YOLO26 object detection?

SAHI (Slicing Aided Hyper Inference) offers several features that complement Ultralytics YOLO26 for object detection:

- **Simple Integration**: SAHI works with YOLO models with minimal code adjustments.
- **Resource Efficiency**: It partitions large images into smaller slices, which keeps memory usage low.
- **High Accuracy**: By merging overlapping detection boxes during the stitching process, SAHI avoids duplicate detections at slice boundaries.

For a deeper understanding, read about SAHI's [key features](#key-features-of-sahi).

### How do I handle large-scale inference projects using YOLO26 and SAHI?

To handle large-scale inference projects using YOLO26 and SAHI, follow these best practices:

1. **Install Required Libraries**: Ensure that you have the latest versions of ultralytics and sahi.
2. **Configure Sliced Inference**: Determine the optimal slice dimensions and overlap ratios for your specific project.
3. **Run Batch Predictions**: Use SAHI's `predict` function to run batch predictions on a directory of images.

Example for batch prediction:

```python
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="path/to/yolo26n.pt",
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
