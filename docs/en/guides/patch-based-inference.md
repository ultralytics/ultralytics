comments: true
description: A comprehensive guide on how to use SAHI-like tool for instance segmentation and detection with support of YOLOv8, YOLOv9, FastSAM, and RTDETR
keywords: YOLOv8-seg, SAHI, Sliced Inference, Object Detection, Ultralytics, Large Scale Image Analysis, High-Resolution Imagery, 

---

# Ultralytics Docs: Using Patch-Based-Inference for segmenting and detecting small objects in images. 

Welcome to the Ultralytics documentation on how to use [Patch-Based-Inference](https://github.com/Koldim2001/YOLO-Patch-Based-Inference). This guide is designed to provide you with all the necessary information to implement Patch-Based-Inference with various ultralytics deep learning models, including YOLOv8, YOLOv9, SAM, and RTDETR.

This library simplifies SAHI-like inference [link](https://docs.ultralytics.com/ru/guides/sahi-tiled-inference/) for instance segmentation tasks, enabling the detection of small objects in images. It caters to both object detection and instance segmentation tasks, supporting a wide range of Ultralytics models. 

Model Support: The library provides support for various ultralytics deep learning models, including YOLOv8, YOLOv9, FastSAM, and RTDETR. Users can choose from pre-trained options or use custom-trained models to best suit their task requirements.

The library also provides a sleek customization of the visualization of the inference results for all models, both in the standard approach (direct network run) and the unique patch-based variant.

## Installation
You can install the library via pip:

```bash
pip install patched_yolo_infer
```

[![PyPI Version](https://img.shields.io/pypi/v/patched-yolo-infer.svg)](https://pypi.org/project/patched-yolo-infer/) - Click here to visit the PyPI page for `patched-yolo-infer`, where you can find more information and documentation.

Note: If CUDA support is available, it's recommended to pre-install PyTorch with CUDA support before installing the library. Otherwise, the CPU version will be installed by default.

---

</details>

## Notebooks

Interactive notebooks are provided to showcase the functionality of the library. These notebooks cover batch-inference procedures for detection, instance segmentation, inference custom visualization, and more. Each notebook is paired with a tutorial on YouTube, making it easy to learn and implement features.


| **Topic** | **Notebook** | **YouTube** |
| ----- | -------- | ------- |
| [Patch-Based-Inference Example](https://nbviewer.org/github/Koldim2001/YOLO-Patch-Based-Inference/blob/main/examples/example_patch_based_inference.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FUao91GyB-ojGRN_okUxYyfagTT9tdsP?usp=sharing) | <p align="center"><a href="https://youtu.be/IfbNOLROym4"><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Youtube Video"></a></p> |
| [Example of utilizing a function to visualize basic Ultralytics model inference results and managing overlapping image crops](https://nbviewer.org/github/Koldim2001/YOLO-Patch-Based-Inference/blob/main/examples/example_extra_functions.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eM4o1e0AUQrS1mLDpcgK9HKInWEvnaMn?usp=sharing) | <p align="center"><a href="https://youtu.be/nBQuWa63188"><img width=30% src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube-rect.png" alt="Youtube Video"></a></p> |


---
## Examples:

#### Detection example:
![detection](https://github.com/Koldim2001/YOLO-Patch-Based-Inference/blob/main/readme_content/getection.gif)


#### Instance Segmentation example 1:
![segmentation](https://github.com/Koldim2001/YOLO-Patch-Based-Inference/raw/main/readme_content/segment_1.gif)


#### Instance Segmentation example 2:
![segmentation](https://github.com/Koldim2001/YOLO-Patch-Based-Inference/raw/main/readme_content/segment_2.gif)


---
## Usage

### 1. Patch-Based-Inference
To carry out patch-based inference of YOLO models using our library, you need to follow a sequential procedure. First, you create an instance of the MakeCropsDetectThem class, providing all desired parameters related to YOLO inference and the patch segmentation principle.<br/> Subsequently, you pass the obtained object of this class to CombineDetections, which facilitates the consolidation of all predictions from each overlapping crop, followed by intelligent suppression of duplicates. <br/>Upon completion, you receive the result, from which you can extract the desired outcome of frame processing.

The output obtained from the process includes several attributes that can be leveraged for further analysis or visualization:

1. img: This attribute contains the original image on which the inference was performed. It provides context for the detected objects.

2. confidences: This attribute holds the confidence scores associated with each detected object. These scores indicate the model's confidence level in the accuracy of its predictions.

3. boxes: These bounding boxes are represented as a list of lists, where each list contains four values: [x_min, y_min, x_max, y_max]. These values correspond to the coordinates of the top-left and bottom-right corners of each bounding box.

4. masks: If available, this attribute provides segmentation masks corresponding to the detected objects. These masks can be used to precisely delineate object boundaries.

5. classes_ids: This attribute contains the class IDs assigned to each detected object. These IDs correspond to specific object classes defined during the model training phase.

6. classes_names: These are the human-readable names corresponding to the class IDs. They provide semantic labels for the detected objects, making the results easier to interpret.

```python
import cv2
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections

# Load the image 
img_path = 'test_image.jpg'
img = cv2.imread(img_path)

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
result = CombineDetections(element_crops, nms_threshold=0.05, match_metric='IOS')  

# Final Results:
img=result.image
confidences=result.filtered_confidences
boxes=result.filtered_boxes
masks=result.filtered_masks
classes_ids=result.filtered_classes_id
classes_names=result.filtered_classes_names
```

#### Explanation of possible input arguments:

**MakeCropsDetectThem**
Class implementing cropping and passing crops through a neural network for detection/segmentation.\
**Args:**
- **image** (*np.ndarray*): Input image BGR.
- **model_path** (*str*): Path to the YOLO model.
- **model** (*ultralytics model*) Pre-initialized model object. If provided, the model will be used directly instead of loading from model_path.
- **imgsz** (*int*): Size of the input image for inference YOLO.
- **conf** (*float*): Confidence threshold for detections YOLO.
- **iou** (*float*): IoU threshold for non-maximum suppression YOLOv8 of single crop.
- **classes_list** (*List[int] or None*): List of classes to filter detections. If None, all classes are considered. Defaults to None.
- **segment** (*bool*): Whether to perform segmentation (YOLOv8-seg).
- **shape_x** (*int*): Size of the crop in the x-coordinate.
- **shape_y** (*int*): Size of the crop in the y-coordinate.
- **overlap_x** (*float*): Percentage of overlap along the x-axis.
- **overlap_y** (*float*): Percentage of overlap along the y-axis.
- **show_crops** (*bool*): Whether to visualize the cropping.
- **resize_initial_size** (*bool*): Whether to resize the results to the original image size (ps: slow operation).

**CombineDetections**
Class implementing combining masks/boxes from multiple crops + NMS (Non-Maximum Suppression).\
**Args:**
- **element_crops** (*MakeCropsDetectThem*): Object containing crop information.
- **nms_threshold** (*float*): IoU/IoS threshold for non-maximum suppression.
- **match_metric** (*str*): Matching metric, either 'IOU' or 'IOS'.
- **intelligent_sorter** (*bool*): Enable sorting by area and rounded confidence parameter. 
            If False, sorting will be done only by confidence (usual nms). (Dafault is False)



---
### 2. Custom inference visualization:
Visualizes custom results of object detection or segmentation on an image.

**Args:**
- **img** (*numpy.ndarray*): The input image in BGR format.
- **boxes** (*list*): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
- **classes_ids** (*list*): A list of class IDs for each detection.
- **confidences** (*list*): A list of confidence scores corresponding to each bounding box. Default is an empty list.
- **classes_names** (*list*): A list of class names corresponding to the class IDs. Default is an empty list.
- **masks** (*list*): A list of masks. Default is an empty list.
- **segment** (*bool*): Whether to perform instance segmentation. Default is False.
- **show_boxes** (*bool*): Whether to show bounding boxes. Default is True.
- **show_class** (*bool*): Whether to show class labels. Default is True.
- **fill_mask** (*bool*): Whether to fill the segmented regions with color. Default is False.
- **alpha** (*float*): The transparency of filled masks. Default is 0.3.
- **color_class_background** (*tuple*): The background BGR color for class labels. Default is (0, 0, 255) (red).
- **color_class_text** (*tuple*): The text color for class labels. Default is (255, 255, 255) (white).
- **thickness** (*int*): The thickness of bounding box and text. Default is 4.
- **font**: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
- **font_scale** (*float*): The scale factor for font size. Default is 1.5.
- **delta_colors** (*int*): The random seed offset for color variation. Default is seed=0.
- **dpi** (*int*): Final visualization size (plot is bigger when dpi is higher). Default is 150.
- **random_object_colors** (*bool*): If true, colors for each object are selected randomly. Default is False.
- **show_confidences** (*bool*): If true and show_class=True, confidences near class are visualized. Default is False.
- **axis_off** (*bool*): If true, axis is turned off in the final visualization. Default is True.
- **show_classes_list** (*list*): If empty, visualize all classes. Otherwise, visualize only classes in the list.
- **return_image_array** (*bool*): If True, the function returns the image (BGR np.array) instead of displaying it. 
                                   Default is False.


Example of using:
```python
from patched_yolo_infer import visualize_results

# Assuming result is an instance of the CombineDetections class
result = CombineDetections(...) 

# Visualizing the results using the visualize_results function
visualize_results(
    img=result.image,
    confidences=result.filtered_confidences,
    boxes=result.filtered_boxes,
    masks=result.filtered_masks,
    classes_ids=result.filtered_classes_id,
    classes_names=result.filtered_classes_names,
    segment=False,
)
```
