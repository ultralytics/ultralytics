---
comments: true
description: A comprehensive guide on how to use Patch-Based-Inference in instance segmentation and object detection tasks.
keywords: Patch-Based-Inference, patched_yolo_infer, YOLOv8, YOLOv8-seg, YOLOv9, YOLOv9-seg, FastSAM, RTDETR, SAHI, Sliced Inference, Instance Segmentation, Object Detection, Ultralytics, Large Scale Image Analysis, Small Object Segmentation
---

# Using Patch-Based-Inference for Small Objects in Object Detection or Segmentation

<p align="center">
  <img width="800" src="https://github.com/ultralytics/ultralytics/assets/62214284/20067e26-1a37-44d1-ae6e-2fd8084e703a" alt="Segmentation Example 2">
</p>

Welcome to the Ultralytics documentation on how to use the Ultralytics Community built [Patch-Based-Inference](https://github.com/Koldim2001/YOLO-Patch-Based-Inference) library. This library simplifies [SAHI](sahi-tiled-inference.md)-like inference for [instance segmentation](../tasks/segment.md) tasks, enabling the detection of small objects in images. It includes functionality for both [object detection](../tasks/detect.md) and [instance segmentation](../tasks/segment.md) tasks, supporting a wide range of [Ultralytics models](../models/index.md).

**Model Compatibility:** The library provides support for various Ultralytics deep learning models, including:

| Model                             | Task                           | Supported |
| --------------------------------- | ------------------------------ | --------- |
| [YOLOv8](../models/yolov8.md)     | [Detect](../tasks/detect.md)   | ✅        |
| [YOLOv9](../models/yolov9.md)     | [Detect](../tasks/detect.md)   | ✅        |
| [RTDETR](../models/rtdetr.md)     | [Detect](../tasks/detect.md)   | ✅        |
| [YOLOv8-seg](../models/yolov8.md) | [Segment](../tasks/segment.md) | ✅        |
| [YOLOv9-seg](../models/yolov9.md) | [Segment](../tasks/segment.md) | ✅        |
| [FastSAM](../models/fast-sam.md)  | [Segment](../tasks/segment.md) | ✅        |

Users may use pre-trained models or use custom-trained models to best suit their project requirements. The `patched_yolo_infer` library also provides sleek customization for the visualization of the inference results from any supported model.

## Installation

To install the `patched_yolo_infer` library, use `pip` to get the package from PyPi [![PyPI Version](https://img.shields.io/pypi/v/patched-yolo-infer.svg)](https://pypi.org/project/patched-yolo-infer/)

```bash
pip install patched_yolo_infer
```

!!! note
If a [CUDA enabled GPU](https://nvidia.com/) is available, it's recommended to pre-install [PyTorch](https://pytorch.org/) with CUDA support before installing `patched_yolo_infer`. Otherwise, the CPU version will be installed by default. See our [quickstart guide](../quickstart.md) on more information regarding setting up an Ultralytics environment.

## Notebooks

These interactive notebooks are provided by the authors of the `patched_yolo_infer` library to showcase its functionality. These notebooks cover batch-inference procedures for detection, instance segmentation, inference custom visualization, and more. Each notebook is paired with a tutorial on YouTube, making it easy to learn and implement features.

| **Topic**                                                                                                                                  | **Notebook**                               | **YouTube**                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Patch-Based-Inference Example][nb_example1]                                                                                               | [![Open In Colab][colab_badge]][colab_ex1] | <div align="center">[<img width=30% alt="Youtube Video" src=https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png>][yt_link1] |
| [Example of utilizing a function to visualize basic Ultralytics model inference results and managing overlapping image crops][nb_example2] | [![Open In Colab][colab_badge]][colab_ex2] | <div align="center">[<img width=30% alt="Youtube Video" src=https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png>][yt_link2] |

## Examples

### Detection example:

<p align="center">
  <img width="800" src="https://github.com/ultralytics/ultralytics/assets/62214284/5a2f682c-d8de-4dfd-9493-d20c51fb0fb3" alt="Detection example">
</p>

### Instance Segmentation:

<p align="center">
  <img width="800" src="https://github.com/ultralytics/ultralytics/assets/62214284/682e3db6-bd72-4a39-95f1-50c10eb41383" alt="Segmentation Example 1">
</p>

## Usage

For using `patched_yolo_infer` library with Ultralytics, here is a basic example.

1. Start with importing the required libraries

    ```python
    import cv2
    from patched_yolo_infer import MakeCropsDetectThem, CombineDetections, visualize_results
    ```

2. Loading the image:

    ```python
    img_path = "path/to/image.jpg"
    img = cv2.imread(img_path)
    ```

3. Next create an instance of the `MakeCropsDetectThem` class using your Ultralytics [detection](../tasks/detect.md) or [segmentation](../tasks/segment.md) model of choice.

    ```python
    element_crops = MakeCropsDetectThem(
        image=img,
        model_path="yolov8m.pt",
        segment=False,
        shape_x=640,
        shape_y=640,
        overlap_x=50,
        overlap_y=50,
        conf=0.5,
        iou=0.7,
        resize_initial_size=True,
    )
    ```

4. Next, use the `CombineDetections` class, that implements the combination of masks or boxes from multiple crops + NMS (Non-maximal suppression), to mesh all model patch-inference results. Select a match metric of [IOU](./yolo-performance-metrics.md#object-detection-metrics) or IOS (intersection over smaller). IOS is a custom metric defined by the `patched_yolo_infer` library.

    ```python
    result = CombineDetections(element_crops, nms_threshold=0.25, match_metric="IOS")
    ```

5. Access data from the `result` object

    ```python
    # Final Results:
    img = result.image
    confidences = result.filtered_confidences
    boxes = result.filtered_boxes
    polygons = result.filtered_polygons
    classes_ids = result.filtered_classes_id
    classes_names = result.filtered_classes_names
    ```

6. Visualizes custom results of object detection or segmentation on an image.

    ```python
    # Visualizing the results using the visualize_results function
    visualize_results(
        img=result.image,
        confidences=result.filtered_confidences,
        boxes=result.filtered_boxes,
        polygons=result.filtered_polygons,
        classes_ids=result.filtered_classes_id,
        classes_names=result.filtered_classes_names,
        segment=False,
    )
    ```

??? example "Full example code"

    ```python
    import cv2
    from patched_yolo_infer import CombineDetections, MakeCropsDetectThem, visualize_results

    img_path = "path/to/image.jpg"
    img = cv2.imread(img_path)

    result = CombineDetections(element_crops, nms_threshold=0.25, match_metric="IOS")

    element_crops = MakeCropsDetectThem(
        image=img,
        model_path="yolov8m.pt",
        segment=False,
        shape_x=640,
        shape_y=640,
        overlap_x=50,
        overlap_y=50,
        conf=0.5,
        iou=0.7,
        resize_initial_size=True,
    )

    # Visualizing the results using the visualize_results function
    visualize_results(
        img=result.image,
        confidences=result.filtered_confidences,
        boxes=result.filtered_boxes,
        polygons=result.filtered_polygons,
        classes_ids=result.filtered_classes_id,
        classes_names=result.filtered_classes_names,
        segment=False,
    )
    ```

### `CombineDetections` instance attributes

| Attribute              | Type                  | Description                                                                                              |
| ---------------------- | --------------------- | -------------------------------------------------------------------------------------------------------- |
| image                  | `numpy.ndarray`       | The original image on which the inference was performed.                                                 |
| filtered_confidences   | `list[numpy.float32]` | The confidence scores associated with each detected object.                                              |
| filtered_boxes         | `list[list[int]]`     | List of lists of bounding box coordinates. Box coordinates are in `[x_min, y_min, x_max, y_max]` format. |
| filtered_polygons      | `list[numpy.ndarray]` | List of segmentation contour coordinates (segment models only).                                          |
| filtered_classes_id    | `list[int]`           | Class indices for each detection.                                                                        |
| filtered_classes_names | `list[str]`           | These are the human-readable names corresponding to the class indices.                                   |

!!! tip

    See the GitHub repository for `patched_yolo_infer` for additional arguments, attributes, and methods available.

## Community Project Thanks

We appreciate all contributions from the Ultralytics Community and appreciate the authors of `patched_yolo_infer` for opening a Pull Request to include their guide. For more information about the Patch-Based-Inference project and its creators, visit the [GitHub repository](https://github.com/Koldim2001/YOLO-Patch-Based-Inference).

[nb_example1]: https://nbviewer.org/github/Koldim2001/YOLO-Patch-Based-Inference/blob/main/examples/example_patch_based_inference.ipynb
[colab_badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab_ex1]: https://colab.research.google.com/drive/1XCpIYLMFEmGSO0XCOkSD7CcD9SFHSJPA?usp=sharing
[yt_link1]: https://www.youtube.com/watch?v=kMfzWd8GK5Y
[nb_example2]: https://nbviewer.org/github/Koldim2001/YOLO-Patch-Based-Inference/blob/main/examples/example_extra_functions.ipynb
[colab_ex2]: https://colab.research.google.com/drive/1eM4o1e0AUQrS1mLDpcgK9HKInWEvnaMn?usp=sharing
[yt_link2]: https://youtu.be/nBQuWa63188
