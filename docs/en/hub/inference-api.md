---
comments: true
description: Learn how to run inference using the Ultralytics HUB Inference API. Includes examples in Python and cURL for quick integration.
keywords: Ultralytics, HUB, Inference API, Python, cURL, REST API, YOLO, image processing, machine learning, AI integration
---

# Ultralytics HUB Inference API

The [Ultralytics HUB](https://bit.ly/ultralytics_hub) Inference API allows you to run inference through our REST API without the need to install and set up the Ultralytics YOLO environment locally.

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with an arrow pointing to the Ultralytics Inference API card](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/inference-api/hub_inference_api_1.jpg)

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/OpWpBI35A5Y"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics HUB Inference API Walkthrough
</p>

## Python

To access the [Ultralytics HUB](https://bit.ly/ultralytics_hub) Inference API using Python, use the following code:

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

!!! note "Note"

    Replace `MODEL_ID` with the desired model ID, `API_KEY` with your actual API key, and `path/to/image.jpg` with the path to the image you want to run inference on.

## cURL

To access the [Ultralytics HUB](https://bit.ly/ultralytics_hub) Inference API using cURL, use the following code:

```bash
curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
	-H "x-api-key: API_KEY" \
	-F "image=@/path/to/image.jpg" \
	-F "size=640" \
	-F "confidence=0.25" \
	-F "iou=0.45"
```

!!! note "Note"

    Replace `MODEL_ID` with the desired model ID, `API_KEY` with your actual API key, and `path/to/image.jpg` with the path to the image you want to run inference on.

## Arguments

See the table below for a full list of available inference arguments.

| Argument     | Default | Type    | Description                                                          |
|--------------|---------|---------|----------------------------------------------------------------------|
| `image`      |         | `image` | Image file to be used for inference.                                 |
| `url`        |         | `str`   | URL of the image if not passing a file.                              |
| `size`       | `640`   | `int`   | Size of the input image, valid range is `32` - `1280` pixels.        |
| `confidence` | `0.25`  | `float` | Confidence threshold for predictions, valid range `0.01` - `1.0`.    |
| `iou`        | `0.45`  | `float` | Intersection over Union (IoU) threshold, valid range `0.0` - `0.95`. |

## Response

The [Ultralytics HUB](https://bit.ly/ultralytics_hub) Inference API returns a JSON response.

### Classification

!!! Example "Classification Model"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # Load model
        model = YOLO("yolov8n-cls.pt")

        # Run inference
        results = model("image.jpg")

        # Print image.jpg results in JSON format
        print(results[0].tojson())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
            -H "x-api-key: API_KEY" \
            -F "image=@/path/to/image.jpg" \
            -F "size=640" \
            -F "confidence=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

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

    === "Response"

        ```json
        {
          success: true,
          message: "Inference complete.",
          data: [
            {
              class: 0,
              name: "person",
              confidence: 0.92
            }
          ]
        }
        ```

### Detection

!!! Example "Detection Model"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # Load model
        model = YOLO("yolov8n.pt")

        # Run inference
        results = model("image.jpg")

        # Print image.jpg results in JSON format
        print(results[0].tojson())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
            -H "x-api-key: API_KEY" \
            -F "image=@/path/to/image.jpg" \
            -F "size=640" \
            -F "confidence=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

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

    === "Response"

        ```json
        {
          success: true,
          message: "Inference complete.",
          data: [
            {
              class: 0,
              name: "person",
              confidence: 0.92,
              width: 0.4893378019332886,
              height: 0.7437513470649719,
              xcenter: 0.4434437155723572,
              ycenter: 0.5198975801467896
            }
          ]
        }
        ```

### OBB

!!! Example "OBB Model"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # Load model
        model = YOLO("yolov8n-obb.pt")

        # Run inference
        results = model("image.jpg")

        # Print image.jpg results in JSON format
        print(results[0].tojson())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
            -H "x-api-key: API_KEY" \
            -F "image=@/path/to/image.jpg" \
            -F "size=640" \
            -F "confidence=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

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

    === "Response"

        ```json
        {
          success: true,
          message: "Inference complete.",
          data: [
            {
              class: 0,
              name: "person",
              confidence: 0.92,
              obb: [
                0.669310450553894,
                0.6247171759605408,
                0.9847468137741089,
                ...
              ]
            }
          ]
        }
        ```

### Segmentation

!!! Example "Segmentation Model"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # Load model
        model = YOLO("yolov8n-seg.pt")

        # Run inference
        results = model("image.jpg")

        # Print image.jpg results in JSON format
        print(results[0].tojson())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
            -H "x-api-key: API_KEY" \
            -F "image=@/path/to/image.jpg" \
            -F "size=640" \
            -F "confidence=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

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

    === "Response"

        ```json
        {
          success: true,
          message: "Inference complete.",
          data: [
            {
              class: 0,
              name: "person",
              confidence: 0.92,
              segment: [0.44140625, 0.15625, 0.439453125, ...]
            }
          ]
        }
        ```

### Pose

!!! Example "Pose Model"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # Load model
        model = YOLO("yolov8n-pose.pt")

        # Run inference
        results = model("image.jpg")

        # Print image.jpg results in JSON format
        print(results[0].tojson())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
            -H "x-api-key: API_KEY" \
            -F "image=@/path/to/image.jpg" \
            -F "size=640" \
            -F "confidence=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

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

    === "Response"

        ```json
        {
          success: true,
          message: "Inference complete.",
          data: [
            {
              class: 0,
              name: "person",
              confidence: 0.92,
              keypoints: [
                0.5290805697441101,
                0.20698919892311096,
                1.0,
                0.5263055562973022,
                0.19584226608276367,
                1.0,
                0.5094948410987854,
                0.19120082259178162,
                1.0,
                ...
              ]
            }
          ]
        }
        ```
