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

=== "cURL"

    ```bash
    curl -H "Authorization: Bearer ul_abc123..." \
      https://platform.ultralytics.com/api/datasets
    ```

=== "Python"

    ```python
    import requests

    headers = {"Authorization": "Bearer ul_abc123..."}
    response = requests.get(
        "https://platform.ultralytics.com/api/datasets",
        headers=headers,
    )
    data = response.json()
    ```

=== "JavaScript"

    ```javascript
    const response = await fetch("https://platform.ultralytics.com/api/datasets", {
      headers: { Authorization: "Bearer ul_abc123..." },
    });
    const data = await response.json();
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

| Parameter  | Type   | Description                      |
| ---------- | ------ | -------------------------------- |
| `username` | string | Filter by username               |
| `slug`     | string | Fetch single dataset by slug     |
| `limit`    | int    | Items per page (default: 20, max: 500) |
| `owner`    | string | Workspace owner username         |

**Response:**

```json
{
    "datasets": [
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
    ],
    "total": 1
}
```

### Get Dataset

```
GET /api/datasets/{datasetId}
```

Returns full dataset details including metadata, class names, and split counts.

### Create Dataset

```
POST /api/datasets
```

**Body:**

```json
{
    "slug": "my-dataset",
    "name": "My Dataset",
    "task": "detect",
    "description": "A custom detection dataset",
    "visibility": "private",
    "classNames": ["person", "car"]
}
```

### Update Dataset

```
PATCH /api/datasets/{datasetId}
```

**Body (partial update):**

```json
{
    "name": "Updated Name",
    "description": "New description",
    "visibility": "public"
}
```

### Delete Dataset

```
DELETE /api/datasets/{datasetId}
```

Soft-deletes the dataset (moved to trash, recoverable for 30 days).

### Clone Dataset

```
POST /api/datasets/{datasetId}/clone
```

Creates a copy of the dataset with all images and labels.

### Export Dataset

```
GET /api/datasets/{datasetId}/export
```

Returns NDJSON format export of the dataset (images and annotations).

### Get Class Statistics

```
GET /api/datasets/{datasetId}/class-stats
```

Returns cached class distribution, location heatmap, and dimension statistics.

**Response:**

```json
{
    "classes": [{ "classId": 0, "count": 1500, "imageCount": 450 }],
    "imageStats": {
        "widthHistogram": [{ "bin": 640, "count": 120 }],
        "heightHistogram": [{ "bin": 480, "count": 95 }]
    },
    "locationHeatmap": { "bins": [[]], "maxCount": 50 },
    "classNames": ["person", "car", "dog"],
    "cached": true
}
```

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

### Auto-Annotate Dataset

```
POST /api/datasets/{datasetId}/predict
```

Run YOLO inference on dataset images to auto-generate annotations.

### Dataset Ingest

```
POST /api/datasets/ingest
```

Create a dataset ingest job to process uploaded ZIP files containing images and labels.

### Dataset Images

#### List Images

```
GET /api/datasets/{datasetId}/images
```

**Query Parameters:**

| Parameter | Type   | Description                          |
| --------- | ------ | ------------------------------------ |
| `split`   | string | Filter by split (train/val/test)     |
| `cursor`  | string | Pagination cursor                    |
| `limit`   | int    | Items per page                       |

#### Get Signed Image URLs

```
POST /api/datasets/{datasetId}/images/urls
```

Get signed URLs for a batch of image hashes (for display in the browser).

#### Delete Image

```
DELETE /api/datasets/{datasetId}/images/{hash}
```

#### Get Image Labels

```
GET /api/datasets/{datasetId}/images/{hash}/labels
```

Returns annotations and class names for a specific image.

#### Update Image Labels

```
PUT /api/datasets/{datasetId}/images/{hash}/labels
```

**Body:**

```json
{
    "labels": [
        { "classId": 0, "bbox": [0.5, 0.5, 0.2, 0.3] }
    ]
}
```

#### Bulk Image Operations

```
PATCH /api/datasets/{datasetId}/images/bulk
```

Move or copy images between datasets.

```
DELETE /api/datasets/{datasetId}/images/bulk
```

Bulk delete images from the dataset.

## Projects API

### List Projects

```
GET /api/projects
```

**Query Parameters:**

| Parameter  | Type   | Description              |
| ---------- | ------ | ------------------------ |
| `username` | string | Filter by username       |
| `limit`    | int    | Items per page           |
| `owner`    | string | Workspace owner username |

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

### Update Project

```
PATCH /api/projects/{projectId}
```

### Delete Project

```
DELETE /api/projects/{projectId}
```

Soft-deletes the project (moved to trash).

### Clone Project

```
POST /api/projects/{projectId}/clone
```

### Project Icon

```
POST /api/projects/{projectId}/icon
```

Upload a project icon (multipart form with image file).

```
DELETE /api/projects/{projectId}/icon
```

Remove the project icon.

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

### List Completed Models

```
GET /api/models/completed
```

Returns models that have finished training (for use in model selectors).

### Get Model

```
GET /api/models/{modelId}
```

### Create Model

```
POST /api/models
```

**Multipart Form:**

| Field       | Type   | Description    |
| ----------- | ------ | -------------- |
| `file`      | file   | Model .pt file |
| `projectId` | string | Target project |
| `name`      | string | Model name     |

### Update Model

```
PATCH /api/models/{modelId}
```

### Delete Model

```
DELETE /api/models/{modelId}
```

### Download Model Files

```
GET /api/models/{modelId}/files
```

Returns signed download URLs for model files.

### Clone Model

```
POST /api/models/{modelId}/clone
```

Clone a model to a new project.

### Track Download

```
POST /api/models/{modelId}/track-download
```

Track model download analytics.

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

### Get Predict Token

```
POST /api/models/{modelId}/predict/token
```

Get a short-lived token for client-side prediction requests.

### Warmup Predict Service

```
POST /api/models/{modelId}/predict/warmup
```

Warm up the predict service for faster first inference.

## Training API

### Start Training

```
POST /api/training/start
```

**Body:**

```json
{
    "modelId": "model_abc123",
    "projectId": "project_xyz789",
    "gpuType": "rtx-4090",
    "trainArgs": {
        "model": "yolo11n.pt",
        "data": "ul://username/datasets/my-dataset",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16
    }
}
```

### Get Training Status

```
GET /api/models/{modelId}/training
```

Returns the current training job status, metrics, and progress for a model.

### Cancel Training

```
DELETE /api/models/{modelId}/training
```

Terminates the running compute instance and marks the job as cancelled.

## Deployments API

### List Deployments

```
GET /api/deployments
```

**Query Parameters:**

| Parameter | Type   | Description          |
| --------- | ------ | -------------------- |
| `modelId` | string | Filter by model      |
| `status`  | string | Filter by status     |
| `limit`   | int    | Max results (default: 20, max: 100) |

### Create Deployment

```
POST /api/deployments
```

**Body:**

```json
{
    "modelId": "model_abc123",
    "region": "us-central1"
}
```

Creates a dedicated inference endpoint in the specified region.

### Get Deployment

```
GET /api/deployments/{deploymentId}
```

### Delete Deployment

```
DELETE /api/deployments/{deploymentId}
```

### Start Deployment

```
POST /api/deployments/{deploymentId}/start
```

Resume a stopped deployment.

### Stop Deployment

```
POST /api/deployments/{deploymentId}/stop
```

Pause a running deployment (stops billing).

### Health Check

```
GET /api/deployments/{deploymentId}/health
```

Returns the health status of the deployment endpoint.

### Run Inference on Deployment

```
POST /api/deployments/{deploymentId}/predict
```

Send an image directly to a deployment endpoint for inference.

### Get Metrics

```
GET /api/deployments/{deploymentId}/metrics
```

Returns request counts, latency, and error rate metrics with sparkline data.

### Get Logs

```
GET /api/deployments/{deploymentId}/logs
```

**Query Parameters:**

| Parameter  | Type   | Description          |
| ---------- | ------ | -------------------- |
| `severity` | string | INFO, WARNING, ERROR |
| `limit`    | int    | Number of entries    |

## Monitoring API

### Aggregated Metrics

```
GET /api/monitoring
```

Returns aggregated metrics across all user deployments (total requests, active deployments, error rate, average latency).

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

### Cancel Export

```
DELETE /api/exports/{exportId}
```

### Track Export Download

```
POST /api/exports/{exportId}/track-download
```

Track export download analytics.

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

**Query Parameters:**

| Parameter | Type   | Description                              |
| --------- | ------ | ---------------------------------------- |
| `type`    | string | Filter: `all`, `project`, `dataset`, `model` |
| `page`    | int    | Page number (default: 1)                 |
| `limit`   | int    | Items per page (default: 50, max: 200)   |
| `owner`   | string | Workspace owner username                 |

### Restore Item

```
POST /api/trash
```

**Body:**

```json
{
    "id": "item_abc123",
    "type": "dataset"
}
```

### Permanently Delete Item

```
DELETE /api/trash
```

**Body:**

```json
{
    "id": "item_abc123",
    "type": "dataset"
}
```

!!! warning "Irreversible"

    Permanent deletion cannot be undone.

### Empty Trash

```
DELETE /api/trash/empty
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

### Get Transactions

```
GET /api/billing/transactions
```

Returns transaction history with pagination.

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

Creates a checkout session for credit purchase ($5-$1000).

### Create Subscription Checkout

```
POST /api/billing/subscription-checkout
```

Creates a checkout session for Pro subscription.

### Create Portal Session

```
POST /api/billing/portal-session
```

Returns URL to billing portal for subscription management.

### Auto Top-Up

#### Get Auto Top-Up Config

```
GET /api/billing/auto-topup
```

#### Update Auto Top-Up Config

```
PATCH /api/billing/auto-topup
```

**Body:**

```json
{
    "enabled": true,
    "thresholdCents": 500,
    "amountCents": 2500
}
```

### Payment Methods

#### List Payment Methods

```
GET /api/billing/payment-methods
```

#### Create Setup Intent

```
POST /api/billing/payment-methods/setup
```

Returns a client secret for adding a new payment method.

#### Set Default Payment Method

```
POST /api/billing/payment-methods/default
```

#### Delete Payment Method

```
DELETE /api/billing/payment-methods/{id}
```

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

### Recalculate Storage

```
POST /api/storage
```

Triggers a recalculation of storage usage.

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
    "name": "training-server"
}
```

### Delete API Key

```
DELETE /api/api-keys
```

**Body:**

```json
{
    "keyId": "key_abc123"
}
```

## Teams & Members API

### List Teams

```
GET /api/teams
```

### Create Team

```
POST /api/teams/create
```

**Body:**

```json
{
    "name": "My Team",
    "slug": "my-team"
}
```

### List Members

```
GET /api/members
```

Returns members of the current workspace.

### Invite Member

```
POST /api/members
```

**Body:**

```json
{
    "email": "user@example.com",
    "role": "editor"
}
```

### Update Member Role

```
PATCH /api/members/{userId}
```

### Remove Member

```
DELETE /api/members/{userId}
```

### Transfer Ownership

```
POST /api/members/transfer-ownership
```

### Invites

#### Accept Invite

```
POST /api/invites/accept
```

#### Get Invite Info

```
GET /api/invites/info
```

**Query Parameters:**

| Parameter | Type   | Description     |
| --------- | ------ | --------------- |
| `token`   | string | Invite token    |

#### Revoke Invite

```
DELETE /api/invites/{inviteId}
```

#### Resend Invite

```
POST /api/invites/{inviteId}/resend
```

## Explore API

### Search Public Content

```
GET /api/explore/search
```

**Query Parameters:**

| Parameter | Type   | Description                           |
| --------- | ------ | ------------------------------------- |
| `q`       | string | Search query                          |
| `type`    | string | Resource type (project, dataset, model) |
| `sort`    | string | Sort order                            |

### Sidebar Data

```
GET /api/explore/sidebar
```

Returns curated content for the Explore sidebar.

## User & Settings APIs

### Get User by Username

```
GET /api/users
```

**Query Parameters:**

| Parameter  | Type   | Description          |
| ---------- | ------ | -------------------- |
| `username` | string | Username to look up  |

### Check Username Availability

```
GET /api/username/check
```

**Query Parameters:**

| Parameter  | Type   | Description          |
| ---------- | ------ | -------------------- |
| `username` | string | Username to check    |

### Settings

```
GET /api/settings
POST /api/settings
```

Get or update user profile settings (display name, bio, social links, etc.).

### Profile Icon

```
POST /api/settings/icon
DELETE /api/settings/icon
```

Upload or remove profile avatar.

### Onboarding

```
POST /api/onboarding
```

Complete onboarding flow (set region, username).

## GDPR API

GDPR compliance endpoints for data export and deletion.

### Export Account Data

```
GET /api/gdpr
```

Download all account data as a JSON export.

### Delete Account

```
POST /api/gdpr
```

**Body:**

```json
{
    "action": "delete"
}
```

!!! warning "Irreversible Action"

    Account deletion is permanent and cannot be undone. All data, models, and deployments will be deleted.

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
GET /api/datasets?page=2&limit=50
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
