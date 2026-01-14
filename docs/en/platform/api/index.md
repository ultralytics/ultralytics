---
comments: true
description: Complete REST API reference for Ultralytics Platform including authentication, endpoints, and examples for datasets, models, and deployments.
keywords: Ultralytics Platform, REST API, API reference, authentication, endpoints, YOLO, programmatic access
---

# REST API Reference

[Ultralytics Platform](https://platform.ultralytics.com) provides a comprehensive REST API for programmatic access to datasets, models, training, and deployments.

<!-- Screenshot: platform-api-overview.avif -->

## Authentication

All API requests require authentication via API key.

### Get API Key

1. Go to **Settings > API Keys**
2. Click **Create Key**
3. Copy the generated key

See [API Keys](../account/api-keys.md) for detailed instructions.

### Authorization Header

Include your API key in all requests:

```bash
Authorization: Bearer ul_your_api_key_here
```

### Example

```bash
curl -H "Authorization: Bearer ul_abc123..." \
  https://platform.ultralytics.com/api/datasets
```

## Base URL

All API endpoints use:

```
https://platform.ultralytics.com/api
```

## Rate Limits

| Plan       | Requests/Minute | Requests/Day |
| ---------- | --------------- | ------------ |
| Free       | 60              | 1,000        |
| Pro        | 300             | 50,000       |
| Enterprise | Custom          | Custom       |

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 55
X-RateLimit-Reset: 1640000000
```

## Response Format

All responses are JSON:

```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 100
  }
}
```

### Error Responses

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid dataset ID",
    "details": { ... }
  }
}
```

## Datasets API

### List Datasets

```
GET /api/datasets
```

**Query Parameters:**

| Parameter | Type   | Description                  |
| --------- | ------ | ---------------------------- |
| `page`    | int    | Page number (default: 1)     |
| `limit`   | int    | Items per page (default: 20) |
| `task`    | string | Filter by task type          |

**Response:**

```json
{
    "success": true,
    "data": [
        {
            "id": "dataset_abc123",
            "name": "my-dataset",
            "slug": "my-dataset",
            "task": "detect",
            "imageCount": 1000,
            "classCount": 10,
            "visibility": "private",
            "createdAt": "2024-01-15T10:00:00Z"
        }
    ]
}
```

### Get Dataset

```
GET /api/datasets/{datasetId}
```

### Create Dataset

```
POST /api/datasets
```

**Body:**

```json
{
    "name": "my-dataset",
    "task": "detect",
    "description": "A custom detection dataset"
}
```

### Delete Dataset

```
DELETE /api/datasets/{datasetId}
```

### Export Dataset

```
POST /api/datasets/{datasetId}/export
```

Returns NDJSON format download URL.

## Projects API

### List Projects

```
GET /api/projects
```

### Get Project

```
GET /api/projects/{projectId}
```

### Create Project

```
POST /api/projects
```

**Body:**

```json
{
    "name": "my-project",
    "description": "Detection experiments"
}
```

### Delete Project

```
DELETE /api/projects/{projectId}
```

## Models API

### List Models

```
GET /api/models
```

**Query Parameters:**

| Parameter   | Type   | Description         |
| ----------- | ------ | ------------------- |
| `projectId` | string | Filter by project   |
| `task`      | string | Filter by task type |

### Get Model

```
GET /api/models/{modelId}
```

### Upload Model

```
POST /api/models
```

**Multipart Form:**

| Field       | Type   | Description    |
| ----------- | ------ | -------------- |
| `file`      | file   | Model .pt file |
| `projectId` | string | Target project |
| `name`      | string | Model name     |

### Delete Model

```
DELETE /api/models/{modelId}
```

### Download Model

```
GET /api/models/{modelId}/files
```

Returns signed download URLs for model files.

### Run Inference

```
POST /api/models/{modelId}/predict
```

**Multipart Form:**

| Field  | Type  | Description          |
| ------ | ----- | -------------------- |
| `file` | file  | Image file           |
| `conf` | float | Confidence threshold |
| `iou`  | float | IoU threshold        |

**Response:**

```json
{
    "success": true,
    "predictions": [
        {
            "class": "person",
            "confidence": 0.92,
            "box": { "x1": 100, "y1": 50, "x2": 300, "y2": 400 }
        }
    ]
}
```

## Training API

### Start Training

```
POST /api/training/start
```

**Body:**

```json
{
    "modelId": "model_abc123",
    "datasetId": "dataset_xyz789",
    "epochs": 100,
    "imageSize": 640,
    "gpuType": "rtx-4090"
}
```

### Get Training Status

```
GET /api/models/{modelId}/training
```

### Cancel Training

```
DELETE /api/models/{modelId}/training
```

## Deployments API

### List Deployments

```
GET /api/deployments
```

**Query Parameters:**

| Parameter | Type   | Description     |
| --------- | ------ | --------------- |
| `modelId` | string | Filter by model |

### Create Deployment

```
POST /api/deployments
```

**Body:**

```json
{
    "modelId": "model_abc123",
    "region": "us-central1",
    "minInstances": 0,
    "maxInstances": 10
}
```

### Get Deployment

```
GET /api/deployments/{deploymentId}
```

### Start Deployment

```
POST /api/deployments/{deploymentId}/start
```

### Stop Deployment

```
POST /api/deployments/{deploymentId}/stop
```

### Delete Deployment

```
DELETE /api/deployments/{deploymentId}
```

### Get Metrics

```
GET /api/deployments/{deploymentId}/metrics
```

### Get Logs

```
GET /api/deployments/{deploymentId}/logs
```

**Query Parameters:**

| Parameter  | Type   | Description          |
| ---------- | ------ | -------------------- |
| `severity` | string | INFO, WARNING, ERROR |
| `limit`    | int    | Number of entries    |

## Export API

### List Exports

```
GET /api/exports
```

### Create Export

```
POST /api/exports
```

**Body:**

```json
{
    "modelId": "model_abc123",
    "format": "onnx"
}
```

**Supported Formats:**

`onnx`, `torchscript`, `openvino`, `tensorrt`, `coreml`, `tflite`, `saved_model`, `graphdef`, `paddle`, `ncnn`, `edgetpu`, `tfjs`, `mnn`, `rknn`, `imx`, `axelera`, `executorch`

### Get Export Status

```
GET /api/exports/{exportId}
```

## Error Codes

| Code               | Description                |
| ------------------ | -------------------------- |
| `UNAUTHORIZED`     | Invalid or missing API key |
| `FORBIDDEN`        | Insufficient permissions   |
| `NOT_FOUND`        | Resource not found         |
| `VALIDATION_ERROR` | Invalid request data       |
| `RATE_LIMITED`     | Too many requests          |
| `INTERNAL_ERROR`   | Server error               |

## SDK Support

For easier integration, use the Ultralytics Python package:

```python
import os

from ultralytics import YOLO

# Set API key
os.environ["ULTRALYTICS_API_KEY"] = "ul_your_key"

# Train with Platform integration
model = YOLO("yolo11n.pt")
model.train(data="ul://username/datasets/my-dataset", project="username/my-project", name="experiment-1", epochs=100)
```

## Webhooks

Webhooks notify your server of Platform events:

| Event                | Description          |
| -------------------- | -------------------- |
| `training.started`   | Training job started |
| `training.epoch`     | Epoch completed      |
| `training.completed` | Training finished    |
| `training.failed`    | Training failed      |
| `export.completed`   | Export ready         |

Webhook setup is available in Enterprise plans.

## FAQ

### How do I paginate large results?

Use `page` and `limit` parameters:

```bash
GET /api/datasets?page=2 &
limit=50
```

### Can I use the API without an SDK?

Yes, all functionality is available via REST. The SDK is a convenience wrapper.

### Are there API client libraries?

Currently, use the Ultralytics Python package or make direct HTTP requests. Official client libraries for other languages are planned.

### How do I handle rate limits?

Implement exponential backoff:

```python
import time


def api_request_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code != 429:
            return response
        wait = 2**attempt
        time.sleep(wait)
    raise Exception("Rate limit exceeded")
```
