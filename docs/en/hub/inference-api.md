---
comments: true
description: Learn how to run inference using the Ultralytics HUB Inference API. Includes examples in Python and cURL for quick integration.
keywords: Ultralytics, HUB, Inference API, Python, cURL, REST API, YOLO, image processing, machine learning, AI integration
---

# Ultralytics HUB Inference API

After you [train a model](./models.md#train-model), you can use the [Shared Inference API](#shared-inference-api) for free. If you are a [Pro](./pro.md) user, you can access the [Dedicated Inference API](#dedicated-inference-api). The [Ultralytics HUB](https://www.ultralytics.com/hub) Inference API allows you to run inference through our REST API without the need to install and set up the Ultralytics YOLO environment locally.

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with an arrow pointing to the Dedicated Inference API card and one to the Shared Inference API card](https://github.com/ultralytics/docs/releases/download/0/hub-inference-api-card.avif)

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/OpWpBI35A5Y"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics HUB Inference API Walkthrough
</p>

## Dedicated Inference API

In response to high demand and widespread interest, we are thrilled to unveil the [Ultralytics HUB](https://www.ultralytics.com/hub) Dedicated Inference API, offering single-click deployment in a dedicated environment for our [Pro](./pro.md) users!

!!! note

    We are excited to offer this feature FREE during our public beta as part of the [Pro Plan](./pro.md), with paid tiers possible in the future.

- **Global Coverage:** Deployed across 38 regions worldwide, ensuring low-latency access from any location. [See the full list of Google Cloud regions](https://cloud.google.com/about/locations).
- **Google Cloud Run-Backed:** Backed by Google Cloud Run, providing infinitely scalable and highly reliable infrastructure.
- **High Speed:** Sub-100ms latency is possible for YOLOv8n inference at 640 resolution from nearby regions based on Ultralytics testing.
- **Enhanced Security:** Provides robust security features to protect your data and ensure compliance with industry standards. [Learn more about Google Cloud security](https://cloud.google.com/security).

To use the [Ultralytics HUB](https://www.ultralytics.com/hub) Dedicated Inference API, click on the **Start Endpoint** button. Next, use the unique endpoint URL as described in the guides below.

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with an arrow pointing to the Start Endpoint button in Dedicated Inference API card](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dedicated-inference-api.avif)

!!! tip

    Choose the region with the lowest latency for the best performance as described in the [documentation](https://docs.ultralytics.com/reference/hub/google/__init__/).

To shut down the dedicated endpoint, click on the **Stop Endpoint** button.

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with an arrow pointing to the Stop Endpoint button in Dedicated Inference API card](https://github.com/ultralytics/docs/releases/download/0/deploy-tab-model-page-stop-endpoint.avif)

## Shared Inference API

To use the [Ultralytics HUB](https://www.ultralytics.com/hub) Shared Inference API, follow the guides below.

Free users have the following usage limits:

- 100 calls / hour
- 1000 calls / month

[Pro](./pro.md) users have the following usage limits:

- 1000 calls / hour
- 10000 calls / month

## Python

To access the [Ultralytics HUB](https://www.ultralytics.com/hub) Inference API using Python, use the following code:

```python
import requests

# API URL, use actual MODEL_ID
url = "https://api.ultralytics.com/v1/predict/MODEL_ID"

# Headers, use actual API_KEY
headers = {"x-api-key": "API_KEY"}

# Inference arguments (optional)
data = {"imgsz": 640, "conf": 0.25, "iou": 0.45}

# Load image and send request
with open("path/to/image.jpg", "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, headers=headers, files=files, data=data)

print(response.json())
```

!!! note

    Replace `MODEL_ID` with the desired model ID, `API_KEY` with your actual API key, and `path/to/image.jpg` with the path to the image you want to run inference on.

    If you are using our [Dedicated Inference API](#dedicated-inference-api), replace the `url` as well.

## cURL

To access the [Ultralytics HUB](https://www.ultralytics.com/hub) Inference API using cURL, use the following code:

```bash
curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
	-H "x-api-key: API_KEY" \
	-F "file=@/path/to/image.jpg" \
	-F "imgsz=640" \
	-F "conf=0.25" \
	-F "iou=0.45"
```

!!! note

    Replace `MODEL_ID` with the desired model ID, `API_KEY` with your actual API key, and `path/to/image.jpg` with the path to the image you want to run inference on.

    If you are using our [Dedicated Inference API](#dedicated-inference-api), replace the `url` as well.

## Arguments

See the table below for a full list of available inference arguments.

| Argument | Default | Type    | Description                                                          |
| -------- | ------- | ------- | -------------------------------------------------------------------- |
| `file`   |         | `file`  | Image or video file to be used for inference.                        |
| `imgsz`  | `640`   | `int`   | Size of the input image, valid range is `32` - `1280` pixels.        |
| `conf`   | `0.25`  | `float` | Confidence threshold for predictions, valid range `0.01` - `1.0`.    |
| `iou`    | `0.45`  | `float` | Intersection over Union (IoU) threshold, valid range `0.0` - `0.95`. |

## Response

The [Ultralytics HUB](https://www.ultralytics.com/hub) Inference API returns a JSON response.

### Classification

!!! example "Classification Model"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # Load model
        model = YOLO("yolov8n-cls.pt")

        # Run inference
        results = model("image.jpg")

        # Print image.jpg results in JSON format
        print(results[0].to_json())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
            -H "x-api-key: API_KEY" \
            -F "file=@/path/to/image.jpg" \
            -F "imgsz=640" \
            -F "conf=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL, use actual MODEL_ID
        url = "https://api.ultralytics.com/v1/predict/MODEL_ID"

        # Headers, use actual API_KEY
        headers = {"x-api-key": "API_KEY"}

        # Inference arguments (optional)
        data = {"imgsz": 640, "conf": 0.25, "iou": 0.45}

        # Load image and send request
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "Response"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```

### Detection

!!! example "Detection Model"

    === "`ultralytics`"

        ```python
        from ultralytics import YOLO

        # Load model
        model = YOLO("yolov8n.pt")

        # Run inference
        results = model("image.jpg")

        # Print image.jpg results in JSON format
        print(results[0].to_json())
        ```

    === "cURL"

        ```bash
        curl -X POST "https://api.ultralytics.com/v1/predict/MODEL_ID" \
            -H "x-api-key: API_KEY" \
            -F "file=@/path/to/image.jpg" \
            -F "imgsz=640" \
            -F "conf=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL, use actual MODEL_ID
        url = "https://api.ultralytics.com/v1/predict/MODEL_ID"

        # Headers, use actual API_KEY
        headers = {"x-api-key": "API_KEY"}

        # Inference arguments (optional)
        data = {"imgsz": 640, "conf": 0.25, "iou": 0.45}

        # Load image and send request
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "Response"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92,
                  "box": {
                    "x1": 118,
                    "x2": 416,
                    "y1": 112,
                    "y2": 660
                  }
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```

### OBB

!!! example "OBB Model"

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
            -F "file=@/path/to/image.jpg" \
            -F "imgsz=640" \
            -F "conf=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL, use actual MODEL_ID
        url = "https://api.ultralytics.com/v1/predict/MODEL_ID"

        # Headers, use actual API_KEY
        headers = {"x-api-key": "API_KEY"}

        # Inference arguments (optional)
        data = {"imgsz": 640, "conf": 0.25, "iou": 0.45}

        # Load image and send request
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "Response"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92,
                  "box": {
                    "x1": 374.85565,
                    "x2": 392.31824,
                    "x3": 412.81805,
                    "x4": 395.35547,
                    "y1": 264.40704,
                    "y2": 267.45728,
                    "y3": 150.0966,
                    "y4": 147.04634
                  }
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```

### Segmentation

!!! example "Segmentation Model"

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
            -F "file=@/path/to/image.jpg" \
            -F "imgsz=640" \
            -F "conf=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL, use actual MODEL_ID
        url = "https://api.ultralytics.com/v1/predict/MODEL_ID"

        # Headers, use actual API_KEY
        headers = {"x-api-key": "API_KEY"}

        # Inference arguments (optional)
        data = {"imgsz": 640, "conf": 0.25, "iou": 0.45}

        # Load image and send request
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "Response"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92,
                  "box": {
                    "x1": 118,
                    "x2": 416,
                    "y1": 112,
                    "y2": 660
                  },
                  "segments": {
                    "x": [
                      266.015625,
                      266.015625,
                      258.984375,
                      ...
                    ],
                    "y": [
                      110.15625,
                      113.67188262939453,
                      120.70311737060547,
                      ...
                    ]
                  }
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```

### Pose

!!! example "Pose Model"

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
            -F "file=@/path/to/image.jpg" \
            -F "imgsz=640" \
            -F "conf=0.25" \
            -F "iou=0.45"
        ```

    === "Python"

        ```python
        import requests

        # API URL, use actual MODEL_ID
        url = "https://api.ultralytics.com/v1/predict/MODEL_ID"

        # Headers, use actual API_KEY
        headers = {"x-api-key": "API_KEY"}

        # Inference arguments (optional)
        data = {"imgsz": 640, "conf": 0.25, "iou": 0.45}

        # Load image and send request
        with open("path/to/image.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, headers=headers, files=files, data=data)

        print(response.json())
        ```

    === "Response"

        ```json
        {
          "images": [
            {
              "results": [
                {
                  "class": 0,
                  "name": "person",
                  "confidence": 0.92,
                  "box": {
                    "x1": 118,
                    "x2": 416,
                    "y1": 112,
                    "y2": 660
                  },
                  "keypoints": {
                    "visible": [
                      0.9909399747848511,
                      0.8162999749183655,
                      0.9872099757194519,
                      ...
                    ],
                    "x": [
                      316.3871765136719,
                      315.9374694824219,
                      304.878173828125,
                      ...
                    ],
                    "y": [
                      156.4207763671875,
                      148.05775451660156,
                      144.93240356445312,
                      ...
                    ]
                  }
                }
              ],
              "shape": [
                750,
                600
              ],
              "speed": {
                "inference": 200.8,
                "postprocess": 0.8,
                "preprocess": 2.8
              }
            }
          ],
          "metadata": ...
        }
        ```
