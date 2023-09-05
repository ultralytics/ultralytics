# SAHI: Slicing Aided Hyper Inference

![teaser](https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif)

## 0. Preparation

- Install latest version of SAHI and ultralytics:

```bash
pip install -U ultralytics sahi
```

- Import required modules:

```python
from sahi.utils.yolov8 import download_yolov8s_model

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
```

- Download a YOLOv8 model and two test images:

```python
# Download YOLOv5s6 model to 'models/yolov5s6.pt'
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

# Download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')
```

## 1. Standard Inference with a YOLOv8 Model

- Instantiate a detection model by defining model weight path and other parameters:

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)
```

- Perform prediction by feeding the get_prediction function with an image path and a DetectionModel instance:

```python
result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)
```

- Or perform prediction by feeding the get_prediction function with a numpy image and a DetectionModel instance:

```python
result = get_prediction(read_image("demo_data/small-vehicles1.jpeg"), detection_model)
```

- Visualize predicted bounding boxes and masks over the original image:

```python
result.export_visuals(export_dir="demo_data/")

Image("demo_data/prediction_visual.png")
```

## 2. Sliced Inference with a YOLOv8 Model

- To perform sliced prediction we need to specify slice parameters. In this example we will perform prediction over slices of 256x256 with an overlap ratio of 0.2:

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

Performing prediction on 15 number of slices.

- Visualize predicted bounding boxes and masks over the original image:

```python
result.export_visuals(export_dir="demo_data/")

Image("demo_data/prediction_visual.png")
```

## 3. Prediction Result

- Predictions are returned as [sahi.prediction.PredictionResult](https://github.com/obss/sahi/blob/b115f08f0c0aeeb151adf24e47c222d3483cc931/demo/sahi/prediction.py), you can access the object prediction list as:

```python
object_prediction_list = result.object_prediction_list
```

```python
object_prediction_list[0]
```

- ObjectPrediction's can be converted to [COCO annotation](https://cocodataset.org/#format-data) format:

```python
result.to_coco_annotations()[:3]
```

- ObjectPrediction's can be converted to [COCO prediction](https://github.com/i008/COCO-dataset-explorer) format:

```python
result.to_coco_predictions(image_id=1)[:3]
```

- ObjectPrediction's can be converted to [imantics](https://github.com/jsbroks/imantics) annotation format:

```python
result.to_imantics_annotations()[:3]
```

- ObjectPrediction's can be converted to [fiftyone](https://github.com/voxel51/fiftyone) detection format:

```python
result.to_fiftyone_detections()[:3]
```

## 4. Batch Prediction

- Set inference arguments and perform sliced inference on given folder:

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
