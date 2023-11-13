With Supervision, you can easily [annotate](https://supervision.roboflow.com/annotators/) predictions obtained from a variety of object detection and segmentation models. This document outlines how to run inference using the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 model, load these predictions into Supervision, and annotate the image.

## Run Inference

First, you'll need to obtain predictions from your object detection or segmentation model.
```python
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image = cv2.imread("image.jpg")
results = model(image)[0]
```

## Load Predictions into Supervision

Now that we have predictions from a model, we can load them into Supervision. We can do so using the [`sv.Detections.from_ultralytics`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_ultralytics) method, which accepts model results from both detection and segmentation models.

```python
import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8n.pt")
image = cv2.imread("image.jpg")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)
```

You can conveniently load predictions from other computer vision frameworks and libraries using:

- [`from_deepsparse`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_deepsparse) ([Deepsparse](https://github.com/neuralmagic/deepsparse))
- [`from_detectron2`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_detectron2) ([Detectron2](https://github.com/facebookresearch/detectron2))
- [`from_mmdetection`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_mmdetection) ([MMDetection](https://github.com/open-mmlab/mmdetection))
- [`from_roboflow`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_roboflow) ([Roboflow Inference](https://github.com/roboflow/inference))
- [`from_sam`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_sam) ([Segment Anything Model](https://github.com/facebookresearch/segment-anything))
- [`from_transformers`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_transformers) ([HuggingFace Transformers](https://github.com/huggingface/transformers))
- [`from_yolo_nas`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_yolo_nas) ([YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md))


## Annotate Image

Finally, we can annotate the image with the predictions. Since we are working with an object detection model, we will use the [`sv.BoundingBoxAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.BoundingBoxAnnotator) and [`sv.LabelAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.LabelAnnotator) classes. If you are running the segmentation model [`sv.MaskAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.MaskAnnotator) is a drop-in replacement for [`sv.BoundingBoxAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.BoundingBoxAnnotator) that will allow you to draw masks instead of boxes.

```python
import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8n.pt")
image = cv2.imread("image.jpg")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    results.names[class_id]
    for class_id
    in detections.class_id
]

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)
```

![Predictions plotted on an image](https://media.roboflow.com/supervision_annotate_example.png)

## Display Annotated Image

To display the annotated image in Jupyter Notebook or Google Colab, use the [`sv.plot_image`](https://supervision.roboflow.com/utils/notebook/#supervision.utils.notebook.plot_image) function.

```python
sv.plot_image(annotated_image)
```
