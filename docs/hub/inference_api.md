---
comments: true
---

# 🚧 Page Under Construction ⚒

This page is currently under construction!️ 👷Please check back later for updates. 😃🔜


# YOLO Inference API

The YOLO Inference API allows you to access the YOLOv8 object detection capabilities via a RESTful API. This enables you to run object detection on images without the need to install and set up the YOLOv8 environment locally.

## API URL

The API URL is the address used to access the YOLO Inference API. In this case, the base URL is:

```
https://api.ultralytics.com/v1/predict
```

## Example Usage in Python

To access the YOLO Inference API with the specified model and API key using Python, you can use the following code:

```python
import requests

# API URL, use actual MODEL_ID
url = f"https://api.ultralytics.com/v1/predict/MODEL_ID"

# Headers, use actual API_KEY
headers = {"x-api-key": "API_KEY"}

# Inference arguments (optional)
data = {"size": 640, "confidence": 0.25, "iou": 0.45}

# Load image and send request
with open("path/to/image.jpg", "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, headers=headers, files=files, data=data)

print(response.json())
```

In this example, replace `API_KEY` with your actual API key, `MODEL_ID` with the desired model ID, and `path/to/image.jpg` with the path to the image you want to analyze.


## Example Usage with CLI

You can use the YOLO Inference API with the command-line interface (CLI) by utilizing the `curl` command. Replace `API_KEY` with your actual API key, `MODEL_ID` with the desired model ID, and `image.jpg` with the path to the image you want to analyze:

```bash
curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
	-H "x-api-key: API_KEY" \
	-F "image=@/path/to/image.jpg" \
	-F "size=640" \
	-F "confidence=0.25" \
	-F "iou=0.45"
```

## Passing Arguments

This command sends a POST request to the YOLO Inference API with the specified `MODEL_ID` in the URL and the `API_KEY` in the request `headers`, along with the image file specified by `@path/to/image.jpg`.

Here's an example of passing the `size`, `confidence`, and `iou` arguments via the API URL using the `requests` library in Python:

```python
import requests

# API URL, use actual MODEL_ID
url = f"https://api.ultralytics.com/v1/predict/MODEL_ID"

# Headers, use actual API_KEY
headers = {"x-api-key": "API_KEY"}

# Inference arguments (optional)
data = {"size": 640, "confidence": 0.25, "iou": 0.45}

# Load image and send request
with open("path/to/image.jpg", "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, headers=headers, files=files, data=data)

print(response.json())
```

In this example, the `data` dictionary contains the query arguments `size`, `confidence`, and `iou`, which tells the API to run inference at image size 640 with confidence and IoU thresholds of 0.25 and 0.45.

This will send the query parameters along with the file in the POST request. See the table below for a full list of available inference arguments.

| Inference Argument | Default | Type    | Notes                                          |
|--------------------|---------|---------|------------------------------------------------|
| `size`             | `640`   | `int`   | valid range is `32` - `1280` pixels            |
| `confidence`       | `0.25`  | `float` | valid range is `0.01` - `1.0`                  |
| `iou`              | `0.45`  | `float` | valid range is `0.0` - `0.95`                  |
| `url`              | `''`    | `str`   | optional image URL if not image file is passed |
| `normalize`        | `False` | `bool`  |                                                |

## Return JSON format

The YOLO Inference API returns a JSON list with the detection results. The format of the JSON list will be the same as the one produced locally by the `results[0].tojson()` command.

The JSON list contains information about the detected objects, their coordinates, classes, and confidence scores.

### Detect Model Format

YOLO detection models, such as `yolov8n.pt`, can return JSON responses from local inference, CLI API inference, and Python API inference. All of these methods produce the same JSON response format.

!!! example "Detect Model JSON Response"

    === "Local"
        ```python
        from ultralytics import YOLO
        
        # Load model
        model = YOLO('yolov8n.pt')

        # Run inference
        results = model('image.jpg')

        # Print image.jpg results in JSON format
        print(results[0].tojson())  
        ```

    === "CLI API"
        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \ 
            -H "x-api-key: API_KEY" \
            -F "image=@/path/to/image.jpg" \
            -F "size=640" \
            -F "confidence=0.25" \
            -F "iou=0.45"
        ```

    === "Python API"
        ```python
        import requests
        
        # API URL, use actual MODEL_ID
        url = f"https://api.ultralytics.com/v1/predict/MODEL_ID"
        
        # Headers, use actual API_KEY
        headers = {"x-api-key": "API_KEY"}
        
        # Inference arguments (optional)
        data = {"size": 640, "confidence": 0.25, "iou": 0.45}
        
        # Load image and send request
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)
        
        print(response.json())
        ```

    === "JSON Response"
        ```json
        {
          "success": True,
          "message": "Inference complete.",
          "data": [
            {
              "name": "person",
              "class": 0,
              "confidence": 0.8359682559967041,
              "box": {
                "x1": 0.08974208831787109,
                "y1": 0.27418340047200523,
                "x2": 0.8706787109375,
                "y2": 0.9887352837456598
              }
            },
            {
              "name": "person",
              "class": 0,
              "confidence": 0.8189555406570435,
              "box": {
                "x1": 0.5847355842590332,
                "y1": 0.05813225640190972,
                "x2": 0.8930277824401855,
                "y2": 0.9903111775716146
              }
            },
            {
              "name": "tie",
              "class": 27,
              "confidence": 0.2909725308418274,
              "box": {
                "x1": 0.3433395862579346,
                "y1": 0.6070465511745877,
                "x2": 0.40964522361755373,
                "y2": 0.9849439832899306
              }
            }
          ]
        }
        ```

### Segment Model Format

YOLO segmentation models, such as `yolov8n-seg.pt`, can return JSON responses from local inference, CLI API inference, and Python API inference. All of these methods produce the same JSON response format.

!!! example "Segment Model JSON Response"

    === "Local"
        ```python
        from ultralytics import YOLO
        
        # Load model
        model = YOLO('yolov8n-seg.pt')

        # Run inference
        results = model('image.jpg')

        # Print image.jpg results in JSON format
        print(results[0].tojson())  
        ```

    === "CLI API"
        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \ 
            -H "x-api-key: API_KEY" \
            -F "image=@/path/to/image.jpg" \
            -F "size=640" \
            -F "confidence=0.25" \
            -F "iou=0.45"
        ```

    === "Python API"
        ```python
        import requests
        
        # API URL, use actual MODEL_ID
        url = f"https://api.ultralytics.com/v1/predict/MODEL_ID"
        
        # Headers, use actual API_KEY
        headers = {"x-api-key": "API_KEY"}
        
        # Inference arguments (optional)
        data = {"size": 640, "confidence": 0.25, "iou": 0.45}
        
        # Load image and send request
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)
        
        print(response.json())
        ```

    === "JSON Response"
        Note `segments` `x` and `y` lengths may vary from one object to another. Larger or more complex objects may have more segment points.
        ```json
        {
          "success": True,
          "message": "Inference complete.",
          "data": [
            {
              "name": "person",
              "class": 0,
              "confidence": 0.856913149356842,
              "box": {
                "x1": 0.1064866065979004,
                "y1": 0.2798851860894097,
                "x2": 0.8738358497619629,
                "y2": 0.9894873725043403
              },
              "segments": {
                "x": [
                  0.421875,
                  0.4203124940395355,
                  0.41718751192092896
                  ...
                ],
                "y": [
                  0.2888889014720917,
                  0.2916666567325592,
                  0.2916666567325592
                  ...
                ]
              }
            },
            {
              "name": "person",
              "class": 0,
              "confidence": 0.8512625694274902,
              "box": {
                "x1": 0.5757311820983887,
                "y1": 0.053943040635850696,
                "x2": 0.8960096359252929,
                "y2": 0.985154045952691
              },
              "segments": {
                "x": [
                  0.7515624761581421,
                  0.75,
                  0.7437499761581421
                  ...
                ],
                "y": [
                  0.0555555559694767,
                  0.05833333358168602,
                  0.05833333358168602
                  ...
                ]
              }
            },
            {
              "name": "tie",
              "class": 27,
              "confidence": 0.6485961675643921,
              "box": {
                "x1": 0.33911995887756347,
                "y1": 0.6057066175672743,
                "x2": 0.4081430912017822,
                "y2": 0.9916408962673611
              },
              "segments": {
                "x": [
                  0.37187498807907104,
                  0.37031251192092896,
                  0.3687500059604645
                  ...
                ],
                "y": [
                  0.6111111044883728,
                  0.6138888597488403,
                  0.6138888597488403
                  ...
                ]
              }
            }
          ]
        }
        ```


### Pose Model Format

YOLO pose models, such as `yolov8n-pose.pt`, can return JSON responses from local inference, CLI API inference, and Python API inference. All of these methods produce the same JSON response format.

!!! example "Pose Model JSON Response"

    === "Local"
        ```python
        from ultralytics import YOLO
        
        # Load model
        model = YOLO('yolov8n-seg.pt')

        # Run inference
        results = model('image.jpg')

        # Print image.jpg results in JSON format
        print(results[0].tojson())  
        ```

    === "CLI API"
        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \ 
            -H "x-api-key: API_KEY" \
            -F "image=@/path/to/image.jpg" \
            -F "size=640" \
            -F "confidence=0.25" \
            -F "iou=0.45"
        ```

    === "Python API"
        ```python
        import requests
        
        # API URL, use actual MODEL_ID
        url = f"https://api.ultralytics.com/v1/predict/MODEL_ID"
        
        # Headers, use actual API_KEY
        headers = {"x-api-key": "API_KEY"}
        
        # Inference arguments (optional)
        data = {"size": 640, "confidence": 0.25, "iou": 0.45}
        
        # Load image and send request
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)
        
        print(response.json())
        ```

    === "JSON Response"
        Note COCO-keypoints pretrained models will have 17 human keypoints. The `visible` part of the keypoints indicates whether a keypoint is visible or obscured. Obscured keypoints may be outside the image or may not be visible, i.e. a person's eyes facing away from the camera.
        ```json
        {
          "success": True,
          "message": "Inference complete.",
          "data": [
            {
              "name": "person",
              "class": 0,
              "confidence": 0.8439509868621826,
              "box": {
                "x1": 0.1125,
                "y1": 0.28194444444444444,
                "x2": 0.7953125,
                "y2": 0.9902777777777778
              },
              "keypoints": {
                "x": [
                  0.5058594942092896,
                  0.5103894472122192,
                  0.4920862317085266
                  ...
                ],
                "y": [
                  0.48964157700538635,
                  0.4643048942089081,
                  0.4465252459049225
                  ...
                ],
                "visible": [
                  0.8726999163627625,
                  0.653947651386261,
                  0.9130823612213135
                  ...
                ]
              }
            },
            {
              "name": "person",
              "class": 0,
              "confidence": 0.7474289536476135,
              "box": {
                "x1": 0.58125,
                "y1": 0.0625,
                "x2": 0.8859375,
                "y2": 0.9888888888888889
              },
              "keypoints": {
                "x": [
                  0.778544008731842,
                  0.7976160049438477,
                  0.7530890107154846
                  ...
                ],
                "y": [
                  0.27595141530036926,
                  0.2378823608160019,
                  0.23644638061523438
                  ...
                ],
                "visible": [
                  0.8900790810585022,
                  0.789978563785553,
                  0.8974530100822449
                  ...
                ]
              }
            }
          ]
        }
        ```