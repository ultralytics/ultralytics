---
comments: true
description: Complete REST API reference for Ultralytics Platform including authentication, endpoints, and examples for datasets, models, and deployments.
keywords: Ultralytics Platform, REST API, API reference, authentication, endpoints, YOLO, programmatic access
---

# REST API Reference

[Ultralytics Platform](https://platform.ultralytics.com) provides a comprehensive REST API for programmatic access to datasets, models, training, and deployments.

<!-- Screenshot: platform-api-overview.avif -->

!!! tip "Quick Start"

    ```bash
    # List your datasets
    curl -H "Authorization: Bearer YOUR_API_KEY" \
      https://platform.ultralytics.com/api/datasets

    # Run inference on a model
    curl -X POST \
      -H "Authorization: Bearer YOUR_API_KEY" \
      -F "file=@image.jpg" \
      https://platform.ultralytics.com/api/models/MODEL_ID/predict
    ```

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

### Get Models Trained on Dataset

```
GET /api/datasets/{datasetId}/models
```

Returns list of models that were trained using this dataset, showing the relationship between datasets and the models they produced.

**Response:**

```json
{
    "success": true,
    "data": [
        {
            "id": "model_abc123",
            "name": "experiment-1",
            "projectId": "project_xyz",
            "trainedAt": "2024-01-15T10:00:00Z",
            "metrics": {
                "mAP50": 0.85,
                "mAP50-95": 0.72
            }
        }
    ]
}
```

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

## Activity API

Track and manage activity events for your account.

### List Activity

```
GET /api/activity
```

**Query Parameters:**

| Parameter   | Type   | Description              |
| ----------- | ------ | ------------------------ |
| `startDate` | string | Filter from date (ISO)   |
| `endDate`   | string | Filter to date (ISO)     |
| `search`    | string | Search in event messages |

### Mark Events Seen

```
POST /api/activity/mark-seen
```

### Archive Events

```
POST /api/activity/archive
```

## Trash API

Manage soft-deleted resources (30-day retention).

### List Trash

```
GET /api/trash
```

### Restore Item

```
POST /api/trash
```

**Body:**

```json
{
    "itemId": "item_abc123",
    "type": "dataset"
}
```

### Empty Trash

```
POST /api/trash/empty
```

Permanently deletes all items in trash.

## Billing API

Manage credits and subscriptions.

### Get Balance

```
GET /api/billing/balance
```

**Response:**

```json
{
    "success": true,
    "data": {
        "cashBalance": 5000000,
        "creditBalance": 20000000,
        "reservedAmount": 0,
        "totalBalance": 25000000
    }
}
```

!!! note "Micro-USD"

    All amounts are in micro-USD (1,000,000 = $1.00) for precise accounting.

### Get Usage Summary

```
GET /api/billing/usage-summary
```

Returns plan details, limits, and usage metrics.

### Create Checkout Session

```
POST /api/billing/checkout-session
```

**Body:**

```json
{
    "amount": 25
}
```

Creates a Stripe checkout session for credit purchase ($5-$1000).

### Create Subscription Checkout

```
POST /api/billing/subscription-checkout
```

Creates a Stripe checkout session for Pro subscription.

### Create Portal Session

```
POST /api/billing/portal-session
```

Returns URL to Stripe billing portal for subscription management.

### Get Payment History

```
GET /api/billing/payments
```

Returns list of payment transactions from Stripe.

## Storage API

### Get Storage Info

```
GET /api/storage
```

**Response:**

```json
{
    "success": true,
    "data": {
        "used": 1073741824,
        "limit": 107374182400,
        "percentage": 1.0
    }
}
```

## GDPR API

GDPR compliance endpoints for data export and deletion.

### Export/Delete Account Data

```
POST /api/gdpr
```

**Body:**

```json
{
    "action": "export"
}
```

| Action   | Description                 |
| -------- | --------------------------- |
| `export` | Download all account data   |
| `delete` | Delete account and all data |

!!! warning "Irreversible Action"

    Account deletion is permanent and cannot be undone. All data, models, and deployments will be deleted.

## API Keys API

### List API Keys

```
GET /api/api-keys
```

### Create API Key

```
POST /api/api-keys
```

**Body:**

```json
{
    "name": "training-server",
    "scopes": ["training", "models"]
}
```

### Delete API Key

```
DELETE /api/api-keys/{keyId}
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

## Python Integration

For easier integration, use the Ultralytics Python package.

### Installation & Setup

```bash
pip install ultralytics
```

Verify installation:

```bash
yolo check
```

!!! warning "Package Version Requirement"

    Platform integration requires **ultralytics>=8.4.0**. Lower versions will NOT work with Platform.

### Authentication

**Method 1: CLI Configuration (Recommended)**

```bash
yolo settings api_key=YOUR_API_KEY
```

**Method 2: Environment Variable**

```bash
export ULTRALYTICS_API_KEY=YOUR_API_KEY
```

**Method 3: In Code**

```python
from ultralytics import settings

settings.api_key = "YOUR_API_KEY"
```

### Using Platform Datasets

Reference datasets with `ul://` URIs:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Train on your Platform dataset
model.train(
    data="ul://your-username/your-dataset",
    epochs=100,
    imgsz=640,
)
```

**URI Format:**

```
ul://{username}/{resource-type}/{name}

Examples:
ul://john/datasets/coco-custom     # Dataset
ul://john/my-project               # Project
ul://john/my-project/exp-1         # Specific model
ul://ultralytics/yolo26/yolo26n    # Official model
```

### Pushing to Platform

Send results to a Platform project:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Results automatically sync to Platform
model.train(
    data="coco8.yaml",
    epochs=100,
    project="ul://your-username/my-project",
    name="experiment-1",
)
```

**What syncs:**

- Training metrics (real-time)
- Final model weights
- Validation plots
- Console output
- System metrics

### API Examples

**Load a model from Platform:**

```python
# Your own model
model = YOLO("ul://username/project/model-name")

# Official model
model = YOLO("ul://ultralytics/yolo26/yolo26n")
```

**Run inference:**

```python
results = model("image.jpg")

# Access results
for r in results:
    boxes = r.boxes  # Detection boxes
    masks = r.masks  # Segmentation masks
    keypoints = r.keypoints  # Pose keypoints
    probs = r.probs  # Classification probabilities
```

**Export model:**

```python
# Export to ONNX
model.export(format="onnx", imgsz=640, half=True)

# Export to TensorRT
model.export(format="engine", imgsz=640, half=True)

# Export to CoreML
model.export(format="coreml", imgsz=640)
```

**Validation:**

```python
metrics = model.val(data="ul://username/my-dataset")

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
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
