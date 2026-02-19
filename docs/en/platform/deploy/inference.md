---
comments: true
description: Learn how to test YOLO models with the Ultralytics Platform inference API including browser testing and programmatic access.
keywords: Ultralytics Platform, inference, API, YOLO, object detection, prediction, testing
---

# Inference

[Ultralytics Platform](https://platform.ultralytics.com) provides an inference API for testing trained models. Use the browser-based `Predict` tab for quick validation or the REST API for programmatic access.

<!-- Screenshot: model-predict-tab-with-detections-overlay.avif -->

## Predict Tab

Every model includes a `Predict` tab for browser-based inference:

1. Navigate to your model
2. Click the **Predict** tab
3. Upload an image, use an example, or open your webcam
4. View predictions instantly with bounding box overlays

<!-- Screenshot: predict-tab-image-upload-dropzone.avif -->

### Input Methods

The predict panel supports multiple input methods:

| Method              | Description                                              |
| ------------------- | -------------------------------------------------------- |
| **Image upload**    | Drag and drop or click to upload an image                |
| **Example images**  | Click built-in examples (dataset images or defaults)     |
| **Webcam capture**  | Live camera feed with single-frame capture               |

### Upload Image

Drag and drop or click to upload:

- **Supported formats**: JPEG, PNG, WebP, AVIF, HEIC, JP2, TIFF, BMP, and more
- **Max size**: 10MB
- **Auto-inference**: Results appear automatically after upload

### Example Images

The predict panel shows example images from your model's linked dataset. If no dataset is linked, default examples are used:

| Image        | Content                    |
| ------------ | -------------------------- |
| `bus.jpg`    | Street scene with vehicles |
| `zidane.jpg` | Sports scene with people   |

For OBB models, aerial images of boats and airports are shown instead.

Example images are preloaded on page load for instant response.

### Webcam

Click the webcam card to start a live camera feed:

1. Grant camera permission when prompted
2. Click the video preview to capture a frame
3. Inference runs automatically on the captured frame
4. Click again to restart the webcam

### View Results

Inference results display:

- **Bounding boxes** with class labels as SVG overlays
- **Confidence scores** for each detection
- **Class colors** from your dataset's color palette (or the Ultralytics default palette)
- **Speed breakdown**: Preprocess, inference, postprocess, and network time

<!-- Screenshot: predict-tab-results-with-detections-and-speed-stats.avif -->

The results panel shows:

| Field              | Description                                     |
| ------------------ | ----------------------------------------------- |
| **Detections list** | Each detection with class name and confidence   |
| **Speed stats**    | Preprocess, inference, postprocess, network (ms)|
| **JSON response**  | Raw API response in a code block                |

## Inference Parameters

Adjust detection behavior with parameters in the collapsible **Parameters** section:

<!-- Screenshot: predict-tab-parameters-sliders.avif -->

| Parameter      | Range   | Default | Description                  |
| -------------- | ------- | ------- | ---------------------------- |
| **Confidence** | 0.01-1.0 | 0.25    | Minimum confidence threshold |
| **IoU**        | 0.0-0.95 | 0.70    | NMS IoU threshold            |
| **Image Size** | 320, 640, 1280 | 640     | Input resize dimension (button toggle) |

Changing any parameter automatically re-runs inference on the current image (debounced at 500ms).

### Confidence Threshold

Filter predictions by confidence:

- **Higher (0.5+)**: Fewer, more certain predictions
- **Lower (0.1-0.25)**: More predictions, some noise
- **Default (0.25)**: Balanced for most use cases

### IoU Threshold

Control Non-Maximum Suppression:

- **Higher (0.7+)**: Allow more overlapping boxes
- **Lower (0.3-0.5)**: Merge nearby detections more aggressively
- **Default (0.70)**: Balanced NMS behavior for most use cases

## Deployment Predict

Each running [dedicated endpoint](endpoints.md) includes a `Predict` tab directly on its deployment card. This uses the deployment's own inference service rather than the shared predict service, letting you test your deployed endpoint from the browser.

## REST API

Access inference programmatically:

### Authentication

Include your API key in requests:

```bash
Authorization: Bearer YOUR_API_KEY
```

### Endpoint

```
POST https://platform.ultralytics.com/api/models/{model_slug}/predict
```

### Request

=== "cURL"

    ```bash
    curl -X POST \
      "https://platform.ultralytics.com/api/models/username/project/model/predict" \
      -H "Authorization: Bearer YOUR_API_KEY" \
      -F "file=@image.jpg" \
      -F "conf=0.25" \
      -F "iou=0.7"
    ```

=== "Python"

    ```python
    import requests

    url = "https://platform.ultralytics.com/api/models/username/project/model/predict"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    files = {"file": open("image.jpg", "rb")}
    data = {"conf": 0.25, "iou": 0.7}

    response = requests.post(url, headers=headers, files=files, data=data)
    print(response.json())
    ```

<!-- Screenshot: predict-tab-code-examples-python-tab.avif -->

### Response

```json
{
    "success": true,
    "predictions": [
        {
            "class": "person",
            "confidence": 0.92,
            "box": {
                "x1": 100,
                "y1": 50,
                "x2": 300,
                "y2": 400
            }
        },
        {
            "class": "car",
            "confidence": 0.87,
            "box": {
                "x1": 400,
                "y1": 200,
                "x2": 600,
                "y2": 350
            }
        }
    ],
    "image": {
        "width": 1920,
        "height": 1080
    }
}
```

<!-- Screenshot: predict-tab-json-response-view.avif -->

### Response Fields

| Field                      | Type    | Description                |
| -------------------------- | ------- | -------------------------- |
| `success`                  | boolean | Request status             |
| `predictions`              | array   | List of detections         |
| `predictions[].class`      | string  | Class name                 |
| `predictions[].confidence` | float   | Detection confidence (0-1) |
| `predictions[].box`        | object  | Bounding box coordinates   |
| `image`                    | object  | Original image dimensions  |

### Task-Specific Responses

Response format varies by task:

=== "Detection"

    ```json
    {
      "class": "person",
      "confidence": 0.92,
      "box": {"x1": 100, "y1": 50, "x2": 300, "y2": 400}
    }
    ```

=== "Segmentation"

    ```json
    {
      "class": "person",
      "confidence": 0.92,
      "box": {"x1": 100, "y1": 50, "x2": 300, "y2": 400},
      "segments": [[100, 50], [150, 60], ...]
    }
    ```

=== "Pose"

    ```json
    {
      "class": "person",
      "confidence": 0.92,
      "box": {"x1": 100, "y1": 50, "x2": 300, "y2": 400},
      "keypoints": [
        {"x": 200, "y": 75, "conf": 0.95},
        ...
      ]
    }
    ```

=== "Classification"

    ```json
    {
      "predictions": [
        {"class": "cat", "confidence": 0.95},
        {"class": "dog", "confidence": 0.03}
      ]
    }
    ```

## Rate Limits

Shared inference has rate limits:

| Plan | Requests/Minute | Requests/Day |
| ---- | --------------- | ------------ |
| Free | 10              | 100          |
| Pro  | 60              | 10,000       |

For higher limits, deploy a [dedicated endpoint](endpoints.md).

## Error Handling

Common error responses:

| Code | Message         | Solution             |
| ---- | --------------- | -------------------- |
| 400  | Invalid image   | Check file format    |
| 401  | Unauthorized    | Verify API key       |
| 404  | Model not found | Check model slug     |
| 429  | Rate limited    | Wait or upgrade plan |
| 500  | Server error    | Retry request        |

## FAQ

### Can I run inference on video?

The API accepts individual frames. For video:

1. Extract frames locally
2. Send each frame to the API
3. Aggregate results

For real-time video, consider deploying a [dedicated endpoint](endpoints.md).

### How do I get the annotated image?

The API returns JSON predictions. To visualize:

1. Use predictions to draw boxes locally
2. Use Ultralytics `plot()` method:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("image.jpg")
results[0].save("annotated.jpg")
```

### What's the maximum image size?

- **Upload limit**: 10MB
- **Recommended**: <5MB for fast inference
- **Auto-resize**: Images are resized to the selected `Image Size` parameter

Large images are automatically resized while preserving aspect ratio.

### Can I run batch inference?

The current API processes one image per request. For batch:

1. Send concurrent requests
2. Use a dedicated endpoint for higher throughput
3. Consider local inference for large batches
