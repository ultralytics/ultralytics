---
plans: [free, pro, enterprise]
comments: true
description: Complete REST API reference for Ultralytics Platform including authentication, rate limits, and endpoints for datasets, images, models, training, exports, and deployments.
keywords: Ultralytics Platform, REST API, API reference, OpenAPI, authentication, endpoints, YOLO, programmatic access
integrations_path: ../../integrations
---

# REST API Reference

[Ultralytics Platform](https://platform.ultralytics.com) provides a REST API for programmatic access to datasets, images, projects, models, training, exports, and deployments.

![Ultralytics Platform Interactive API Documentation](https://cdn.ul.run/i/902c8979434234a150d1d2df438ec1fb.avif)<!-- screenshot -->

!!! tip "Quick Start"

    === "cURL"

        ```bash
        # List the datasets owned by a workspace
        curl -H "Authorization: Bearer YOUR_API_KEY" \
          https://platform.ultralytics.com/api/datasets/YOUR_USERNAME
        ```

    === "Python SDK"

        ```bash
        pip install "ultralytics-platform>=0.1.5" # Python 3.11+
        ```

        ```python
        from ultralytics_platform import Platform

        client = Platform(api_key="YOUR_API_KEY")
        datasets = client.datasets.list("YOUR_USERNAME")
        ```

    Every endpoint below lists its `client.<resource>.<method>(...)` call from the
    [`ultralytics-platform`](https://pypi.org/project/ultralytics-platform/) SDK, which is generated from the same
    contract as this reference.

!!! tip "Interactive API Reference"

    This page is a guided tour of the API. The generated, always-current reference lives at
    [platform.ultralytics.com/api/docs](https://platform.ultralytics.com/api/docs), and the machine-readable OpenAPI 3.2
    document that powers it is published at
    [platform.ultralytics.com/openapi.json](https://platform.ultralytics.com/openapi.json). Both are generated directly
    from the server-side contract, so they are the authority whenever this page and the schema disagree.

## API Overview

The API is organized around the core Platform resources:

```mermaid
graph LR
    A[API Key]:::start --> B[Datasets]:::proc
    A --> C[Projects]:::proc
    B -->|images| G[Images]:::proc
    C -->|contains| D[Models]:::proc
    B -->|train on| D
    D -->|deploy| E[Deployments]:::proc
    D -->|export| F[Exports]:::proc

    classDef start fill:#4CAF50,color:#fff
    classDef proc fill:#2196F3,color:#fff
```

| Resource                                   | Description                     | Key Operations                                        |
| ------------------------------------------ | ------------------------------- | ----------------------------------------------------- |
| [Datasets](../data/datasets.md)            | Labeled image collections       | CRUD, ingest, versions, classes, splits, clone        |
| [Images](../data/annotation.md)            | Individual images and labels    | Read, annotate, move split, delete, auto-annotate     |
| [Projects](../train/projects.md)           | Model workspaces                | CRUD, clone                                           |
| [Models](../train/models.md)               | Trained checkpoints             | CRUD, predict, download, clone, training status       |
| [Training](../train/cloud-training.md)     | Cloud GPU training jobs         | GPU availability, start, progress, cancel             |
| [Exports](../train/models.md#export-model) | Format conversion jobs          | Create, list, status, cancel                          |
| [Deployments](../deploy/endpoints.md)      | Dedicated inference endpoints   | Create, start/stop/replace, predict, metrics, logs    |
| [Trash](../account/trash.md)               | Soft-deleted resources          | List, restore, permanently delete                     |
| [Storage](../integrations/index.md)        | Cloud storage integrations      | Connect, discover, browse, disconnect                 |
| [Account](../account/settings.md)          | Plan, credits, storage, profile | Account summary, API keys, storage usage, user lookup |
| [Billing](../account/billing.md)           | Plan usage and ledger           | Usage summary, transactions                           |
| [Explore](../explore.md)                   | Public content search           | Search projects and datasets                          |

## Authentication

Most endpoints require an API key. Endpoints that expose public content — reading a public dataset, project, or model,
listing public dataset images, running inference on a public model, or searching Explore — also accept anonymous
requests and simply return more when a key is supplied.

### Get an API Key

1. Go to `Settings` > `API Keys`
2. Click `Create Key`
3. Copy the generated key

See [API Keys](../account/api-keys.md) for detailed instructions.

### Authorization Header

Include your API key as a bearer token:

```http
Authorization: Bearer YOUR_API_KEY
```

!!! info "API Key Format"

    API keys are the literal prefix `ul_` followed by 40 hexadecimal characters, 43 characters in total (for example
    `ul_a1b2c3d4e5f6789012345678901234567890abcd`). Requests with a missing header, a malformed key, or a revoked key
    return `401`. Keep your key secret -- never commit it to version control or share it publicly.

### Example

=== "cURL"

    ```bash
    curl -H "Authorization: Bearer YOUR_API_KEY" \
      https://platform.ultralytics.com/api/account/summary
    ```

=== "Python"

    ```python
    from ultralytics_platform import Platform

    client = Platform(api_key="YOUR_API_KEY")  # or set ULTRALYTICS_API_KEY
    account = client.account.summary()
    print(account["username"], account["plan"], account["creditsCents"])
    ```

=== "JavaScript"

    ```javascript
    const response = await fetch("https://platform.ultralytics.com/api/account/summary", {
      headers: { Authorization: "Bearer YOUR_API_KEY" },
    });
    const account = await response.json();
    ```

## Base URL

All API endpoints use:

```text
https://platform.ultralytics.com/api
```

## Resource Paths

Resources are addressed by the same human-readable names that appear in Platform URLs, not by database IDs:

| Resource   | Path                                    | Example                                 |
| ---------- | --------------------------------------- | --------------------------------------- |
| Dataset    | `/api/datasets/{owner}/{dataset}`       | `/api/datasets/acme-vision/warehouse`   |
| Project    | `/api/projects/{owner}/{project}`       | `/api/projects/acme-vision/inspection`  |
| Model      | `/api/models/{owner}/{project}/{model}` | `/api/models/acme-vision/inspection/v3` |
| Deployment | `/api/deployments/{owner}/{deployment}` | `/api/deployments/acme-vision/edge-1`   |
| Image      | `/api/images/{imageId}`                 | `/api/images/65f1c0a2b3d4e5f601234567`  |

- `{owner}` is a personal username or a team workspace handle: 4-32 characters, lowercase alphanumeric with single
  hyphens between segments.
- `{dataset}`, `{project}`, `{model}`, and `{deployment}` follow the same lowercase-hyphenated pattern, up to 128
  characters.
- `{imageId}` and `{exportId}` are 24-character hexadecimal IDs returned by the API.
- Renaming a resource through `PATCH` changes the display `name` and the URL name together, and the response returns
  the current URL name so you can keep following it.

!!! note "Workspace Selection"

    There is no `owner` query parameter. Workspace-scoped paths carry the owner in the path, and account-scoped
    endpoints (`/api/account/summary`, `/api/api-keys`, `/api/storage`, `/api/billing/*`, `/api/trash`,
    `/api/integrations/buckets`) operate on the workspace that issued the API key. To act on a team workspace, use an
    API key created in that workspace.

## Rate Limits

The API enforces sliding-window limits per API key. Each route falls into one category, and each category
has an independent counter, so 20 predict requests do not consume your default allowance.

| Category       | Limit            | Applies To                                                                                |
| -------------- | ---------------- | ----------------------------------------------------------------------------------------- |
| **Default**    | 100 requests/min | Every route not listed below                                                              |
| **Training**   | 10 requests/min  | `POST /api/training/start`                                                                |
| **Upload**     | 10 requests/min  | Signed upload URLs, upload completion, and dataset ingest                                 |
| **Predict**    | 20 requests/min  | Model and deployment inference through Platform API routes                                |
| **Export**     | 20 requests/min  | Model export routes and dataset export/version routes                                     |
| **Download**   | 30 requests/min  | Model file downloads                                                                      |
| **Mutation**   | 10 requests/min  | Listing API keys, connecting or discovering cloud storage, and deployment `PATCH` actions |
| **Hydrate**    | 20 requests/min  | `POST /api/datasets/{owner}/{dataset}/images` (fetching a selected set of images)         |
| **Clustering** | 10 requests/min  | `GET /api/datasets/{owner}/{dataset}/images/clustering`                                   |

Browser-only Platform routes, such as billing checkout and team management, have their own limits that do not apply to
API-key traffic.

When throttled, the API returns `429` with both headers and a JSON body:

```http
Retry-After: 12
X-RateLimit-Reset: 2026-02-21T12:34:56.000Z
```

```json
{
    "error": "Rate limit exceeded",
    "retryAfter": 12,
    "resetAt": "2026-02-21T12:34:56.000Z"
}
```

### Dedicated Endpoints (Unlimited)

[Dedicated endpoints](../deploy/endpoints.md) are **not subject to Platform API-key rate limits** when you call the
deployment's own `serviceUrl` directly (for example, `https://predict-abc123.run.app/predict`). Throughput then depends
on the deployed service configuration.

!!! tip "Handling Rate Limits"

    When you receive a `429`, wait for `Retry-After` seconds (or until `X-RateLimit-Reset`) before retrying. See the
    [rate limit FAQ](#how-do-i-handle-rate-limits) for an exponential backoff implementation.

## Response Format

### Success Responses

Responses are JSON objects with resource-specific fields. There is no generic envelope: list endpoints return a named
collection alongside counts, and mutations return the changed identifiers.

```json
{
    "datasets": [{ "id": "65f1c0a2b3d4e5f601234567", "owner": "acme-vision", "dataset": "warehouse" }],
    "total": 1,
    "region": "us"
}
```

Data-bearing responses also include `region` (`us`, `eu`, or `ap`), the storage region for that workspace.

### Error Responses

Every error response is a JSON object with an `error` message:

```json
{
    "error": "Dataset not found"
}
```

| HTTP Status | Meaning                                                     |
| ----------- | ----------------------------------------------------------- |
| `200`       | Success                                                     |
| `201`       | Created                                                     |
| `202`       | Accepted, work continues asynchronously                     |
| `400`       | Invalid path, query, or request body                        |
| `401`       | Missing or invalid authentication                           |
| `402`       | Insufficient credits (training)                             |
| `403`       | Insufficient permissions, plan, or quota                    |
| `404`       | Resource not found                                          |
| `409`       | Conflict with current state (duplicate name, job in flight) |
| `413`       | Prediction input too large                                  |
| `422`       | Model classes do not match the dataset (auto-annotation)    |
| `429`       | Rate limit exceeded                                         |
| `500`       | Server error                                                |
| `502`       | Upstream provider or service call failed                    |
| `503`       | Dependent service temporarily unavailable                   |

## Pagination

Pagination style depends on the collection:

| Style             | Endpoints                                              | Parameters                                        |
| ----------------- | ------------------------------------------------------ | ------------------------------------------------- |
| Limit only        | Datasets, projects, models, exports, deployments lists | `limit`                                           |
| Offset and limit  | Dataset images, image clustering, Explore search       | `offset`, `limit`, plus `hasMore` in the response |
| Cursor            | Dataset images (large datasets)                        | `cursor`, `includeTotal`, plus `nextCursor`       |
| Page number       | Trash                                                  | `page`, `limit`, plus `totalPages`                |
| Opaque page token | Deployment logs                                        | `pageToken`, plus `nextPageToken`                 |

---

## Datasets API

Create, browse, and manage labeled image datasets for training YOLO models. See
[Datasets documentation](../data/datasets.md).

### List Datasets

```http
GET /api/datasets/{owner}
```

**Python SDK:** `client.datasets.list(owner)`

Returns the owner's public datasets, plus private datasets when your key can view that workspace.

**Query Parameters:**

| Parameter          | Type    | Description                                                     |
| ------------------ | ------- | --------------------------------------------------------------- |
| `limit`            | int     | Maximum datasets to return (default: 1000, max: 1000)           |
| `includeSamples`   | boolean | Include sample image previews (default: `true`)                 |
| `includeImageUrls` | boolean | Include full-size sample image fallback URLs (default: `false`) |

=== "cURL"

    ```bash
    curl -H "Authorization: Bearer YOUR_API_KEY" \
      "https://platform.ultralytics.com/api/datasets/acme-vision?limit=10&includeSamples=false"
    ```

=== "Python"

    ```python
    from ultralytics_platform import Platform

    client = Platform()  # reads ULTRALYTICS_API_KEY
    for ds in client.datasets.list("acme-vision", limit=10, include_samples="false")["datasets"]:
        print(f"{ds['dataset']}: {ds['imageCount']} images, task={ds['task']}")
    ```

**Response:**

```json
{
    "datasets": [
        {
            "id": "65f1c0a2b3d4e5f601234567",
            "owner": "acme-vision",
            "dataset": "warehouse",
            "name": "Warehouse",
            "task": "detect",
            "visibility": "private",
            "imageCount": 1000,
            "classCount": 2,
            "classNames": ["person", "forklift"],
            "splits": { "train": 800, "val": 200, "test": 0, "labeled": 1000 },
            "annotationCount": 5400,
            "starCount": 3,
            "isStarred": false,
            "status": "ready",
            "createdAt": "2026-01-15T10:00:00Z",
            "updatedAt": "2026-01-16T08:30:00Z"
        }
    ],
    "total": 1,
    "region": "us"
}
```

### Get Dataset

```http
GET /api/datasets/{owner}/{dataset}
```

**Python SDK:** `client.datasets.retrieve(owner, dataset)`

Returns the full dataset object under a `dataset` key, including `classNames`, `splits`, `versions`, `source`, and the
user-defined `metadata` object.

### Create Dataset

```http
POST /api/datasets
```

**Python SDK:** `client.datasets.create(dataset=..., name=...)`

**Body:**

```json
{
    "dataset": "warehouse",
    "name": "Warehouse",
    "task": "detect",
    "description": "Forklift and pedestrian safety dataset",
    "classNames": ["person", "forklift"],
    "visibility": "private",
    "metadata": { "location": "factory-1", "reviewed": true },
    "owner": "acme-vision"
}
```

| Field         | Type   | Required | Description                                                               |
| ------------- | ------ | -------- | ------------------------------------------------------------------------- |
| `dataset`     | string | Yes      | Dataset name used in Platform URLs (lowercase, hyphenated, max 128 chars) |
| `name`        | string | Yes      | Display name (max 100 chars)                                              |
| `description` | string | No       | Description (max 1000 chars)                                              |
| `task`        | string | No       | Task type (default: `detect`)                                             |
| `classNames`  | array  | No       | Class names in index order (max 25,000)                                   |
| `format`      | string | No       | Annotation format: `yolo` (default), `coco`, `raw`, `ndjson`              |
| `visibility`  | string | No       | `public` or `private`                                                     |
| `tags`        | array  | No       | Up to 50 tags of 50 characters each                                       |
| `license`     | string | No       | Dataset license identifier                                                |
| `metadata`    | object | No       | Custom JSON metadata                                                      |
| `owner`       | string | No       | Team workspace handle; defaults to your personal workspace                |

!!! note "Supported Tasks"

    Valid `task` values when creating or updating a dataset: `detect`, `segment`, `semantic`, `depth`, `classify`,
    `pose`, and `obb`. Depth datasets have no classes.

**Response (`201`):**

```json
{
    "id": "65f1c0a2b3d4e5f601234567",
    "owner": "acme-vision",
    "dataset": "warehouse",
    "region": "us"
}
```

### Update Dataset

```http
PATCH /api/datasets/{owner}/{dataset}
```

**Python SDK:** `client.datasets.update(owner, dataset)`

**Body (partial update):**

```json
{
    "name": "Warehouse Safety",
    "description": "New description",
    "visibility": "public",
    "metadata": { "location": "factory-2", "reviewed": true }
}
```

Accepted fields: `name`, `description`, `visibility`, `metadata`, `tags`, `classNames`, `classColors`, `format`, `task`,
`license`, `iconColor`, `iconLetter`, and `starred`. Send an empty `metadata` object (`{}`) to clear custom metadata.
Metadata keys are limited to 128 characters and the serialized object to 500,000 characters.

**Response:**

```json
{
    "success": true,
    "dataset": "warehouse-safety"
}
```

Renaming changes the URL name, so use the returned `dataset` value for subsequent requests.

### Delete Dataset

```http
DELETE /api/datasets/{owner}/{dataset}
```

**Python SDK:** `client.datasets.delete(owner, dataset)`

Moves the dataset to [trash](../account/trash.md), where it is recoverable for 30 days.

### Clone Dataset

```http
POST /api/datasets/{owner}/{dataset}/clone
```

**Python SDK:** `client.datasets.clone(owner, dataset)`

Copies an accessible dataset, with its images and labels, into your personal workspace or a team workspace.

**Optional body (all fields optional):**

```json
{
    "dataset": "warehouse-copy",
    "name": "Warehouse Copy",
    "description": "Cloned for experimentation",
    "visibility": "private",
    "license": "CC-BY-4.0",
    "owner": "acme-vision"
}
```

**Response (`201`):** `id`, `owner`, `dataset`, `name`, `imageCount`, `classCount`, and `region`. Datasets backed by a
connected storage source return `409` because their files are not copied.

### Download a Dataset Export

```http
GET /api/datasets/{owner}/{dataset}/export
```

**Python SDK:** `client.datasets.export(owner, dataset)`

Returns a signed NDJSON download URL. Omit `v` to export the dataset's current state, reusing the cached export when
nothing changed since it was generated.

**Query Parameters:**

| Parameter | Type    | Description                                                     |
| --------- | ------- | --------------------------------------------------------------- |
| `v`       | integer | Saved version number (1-indexed). Omit for the current dataset. |

**Response:**

```json
{
    "downloadUrl": "https://storage.googleapis.com/...&signature=...",
    "cached": true
}
```

Requesting a specific version returns `downloadUrl` and `version` instead of `cached`.

### Create Dataset Version

```http
POST /api/datasets/{owner}/{dataset}/export
```

**Python SDK:** `client.datasets.create_export(owner, dataset)`

Creates an immutable numbered snapshot of the dataset and stores its NDJSON export. Requires editor access.

**Body (optional):**

```json
{
    "description": "Added 500 training images"
}
```

**Response:**

```json
{
    "version": 3,
    "downloadUrl": "https://storage.googleapis.com/...&signature=...",
    "reused": false
}
```

`reused` is `true` when the dataset is unchanged since the previous version and that snapshot was returned instead.

### Update Version Description

```http
PATCH /api/datasets/{owner}/{dataset}/export
```

**Python SDK:** `client.datasets.update_export(owner, dataset, version=..., description=...)`

**Body:**

```json
{
    "version": 2,
    "description": "Fixed mislabeled classes"
}
```

**Response:** `{"ok": true}`

### Restore Dataset Version

```http
POST /api/datasets/{owner}/{dataset}/restore
```

**Python SDK:** `client.datasets.restore(owner, dataset, version=...)`

Rebuilds images, annotations, and classes from a saved version without copying image bytes.

**Body:**

```json
{
    "version": 2
}
```

**Response:** `{"version": 2, "imageCount": 1000}`

### Get Dataset Statistics

```http
GET /api/datasets/{owner}/{dataset}/class-stats
```

**Python SDK:** `client.datasets.class_stats(owner, dataset)`

Returns per-class annotation counts, image and annotation histograms, and heatmaps. Large datasets are sampled, in
which case `sampleSize` reports how many images contributed.

**Response (abbreviated):**

```json
{
    "classes": [{ "classId": 0, "count": 1500, "imageCount": 450 }],
    "imageStats": {
        "widthHistogram": [{ "bin": 640, "count": 120, "size": 1 }],
        "heightHistogram": [{ "bin": 480, "count": 95, "size": 1 }],
        "pointsHistogram": [{ "bin": 4, "count": 200, "size": 1 }],
        "formatDistribution": { "jpg": 900, "png": 100 },
        "fileSizeHistogram": [{ "bin": 250000, "count": 300, "size": 50000 }],
        "objectsPerImageHistogram": [{ "bin": 5, "count": 210, "size": 1 }],
        "bboxWidthHistogram": [{ "bin": 120, "count": 340, "size": 20 }],
        "bboxHeightHistogram": [{ "bin": 90, "count": 300, "size": 20 }]
    },
    "locationHeatmap": {
        "bins": [
            [5, 10],
            [8, 3]
        ],
        "maxCount": 50
    },
    "dimensionHeatmap": {
        "bins": [
            [2, 5],
            [3, 1]
        ],
        "maxCount": 12,
        "minWidth": 10,
        "maxWidth": 1920,
        "minHeight": 10,
        "maxHeight": 1080
    },
    "classNames": ["person", "forklift"],
    "cached": true,
    "sampleSize": null
}
```

### Manage Classes

Merge classes (reassign annotations to a target class, then remove the sources):

```http
POST /api/datasets/{owner}/{dataset}/classes/merge
```

**Python SDK:** `client.datasets.merge_classes(owner, dataset, source_class_ids=..., target_class_id=...)`

```json
{
    "sourceClassIds": [2, 4],
    "targetClassId": 1
}
```

Delete classes (their annotations are deleted and the remaining class IDs shift down):

```http
POST /api/datasets/{owner}/{dataset}/classes/delete
```

**Python SDK:** `client.datasets.delete_classes(owner, dataset, class_ids=...)`

```json
{
    "classIds": [2, 4]
}
```

Both operations return `success`, the updated `classNames` and `classColors`, and a summary of what changed
(`mergedClassIds` and `targetClassId`, or `deletedClassIds` and `deletedAnnotations`).

!!! warning "Class IDs Are Positional"

    Because remaining IDs shift after a merge or delete, these operations are not idempotent. Re-fetch the dataset to
    get current class indices before issuing another class operation.

### Redistribute Splits

```http
POST /api/datasets/{owner}/{dataset}/splits/redistribute
```

**Python SDK:** `client.datasets.redistribute_splits(owner, dataset, train=..., val=..., test=...)`

Randomly reassigns images across splits. The three percentages must total 100.

```json
{
    "train": 80,
    "val": 20,
    "test": 0
}
```

**Response:** `success`, the resulting `splits` counts, and `modified` (number of images moved).

### Dataset Embeddings

```http
GET /api/datasets/{owner}/{dataset}/embeddings
POST /api/datasets/{owner}/{dataset}/embeddings
DELETE /api/datasets/{owner}/{dataset}/embeddings
```

**Python SDK:** `client.datasets.embeddings(owner, dataset)`, `client.datasets.create_embeddings(owner, dataset)`,
`client.datasets.delete_embeddings(owner, dataset)`

`GET` returns the analysis summary (`analyzedAt`, `embeddingsCount`, `latestImageAt`, `activeJob`). `POST` queues an
embedding analysis and returns `202` with a `jobId`. `DELETE` cancels the active job and returns the cancelled job ID
or `null`.

### Image Clustering

```http
GET /api/datasets/{owner}/{dataset}/images/clustering
```

**Python SDK:** `client.datasets.clustering(owner, dataset)`

Returns the UMAP 2D layout from a completed analysis, paginated with `offset` and `limit` (default and max 50,000).
Each entry has `id`, `umapX`, `umapY`, `split`, `classIds`, `width`, `height`, `bytes`, `labelCount`, and `missing`.

### List Models Trained on a Dataset

```http
GET /api/datasets/{owner}/{dataset}/models
```

**Python SDK:** `client.datasets.models(owner, dataset)`

**Response:**

```json
{
    "models": [
        {
            "id": "65f1c0a2b3d4e5f601234599",
            "owner": "acme-vision",
            "project": "inspection",
            "model": "v3",
            "name": "v3",
            "status": "completed",
            "task": "detect",
            "epochs": 100,
            "bestEpoch": 87,
            "metrics": { "mAP50": 0.85, "mAP50-95": 0.72, "precision": 0.88, "recall": 0.81 },
            "startedAt": "2026-01-14T22:00:00Z",
            "completedAt": "2026-01-15T10:00:00Z",
            "createdAt": "2026-01-14T21:55:00Z"
        }
    ],
    "count": 1
}
```

### List Dataset Images

```http
GET /api/datasets/{owner}/{dataset}/images
```

**Python SDK:** `client.datasets.images(owner, dataset)`

**Query Parameters:**

| Parameter           | Type    | Description                                                                                                                                                         |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `limit`             | int     | Maximum images to return (default: 50, max: 5000)                                                                                                                   |
| `offset`            | int     | Images to skip (default: 0)                                                                                                                                         |
| `cursor`            | string  | Last image ID from the previous page, for cursor pagination                                                                                                         |
| `includeTotal`      | boolean | Include the total matching count (default: `true`)                                                                                                                  |
| `split`             | string  | Filter by split: `train`, `val`, `test`                                                                                                                             |
| `hasLabel`          | boolean | Filter by annotation state                                                                                                                                          |
| `hasError`          | boolean | Filter by processing error state                                                                                                                                    |
| `classIds`          | string  | Comma-separated class IDs; returns images containing any of them                                                                                                    |
| `search`            | string  | Substring match on filename and custom metadata (max 200 chars)                                                                                                     |
| `sort`              | string  | `newest` (default), `oldest`, `name-asc`, `name-desc`, `height-asc`, `height-desc`, `width-asc`, `width-desc`, `size-asc`, `size-desc`, `labels-asc`, `labels-desc` |
| `includeThumbnails` | boolean | Include signed thumbnail URLs (default: `true`)                                                                                                                     |
| `includeImageUrls`  | boolean | Include signed full-size image URLs (default: `false`)                                                                                                              |
| `includeLabels`     | boolean | Include capped preview annotations (default: `false`)                                                                                                               |

**Response:**

```json
{
    "images": [
        {
            "id": "65f1c0a2b3d4e5f601234567",
            "hash": "9f2c1d4b6a8e0f3c5d7b9a1e2f4c6d8b",
            "ext": "jpg",
            "name": "aisle-04.jpg",
            "thumbnailUrl": "https://storage.googleapis.com/...&signature=...",
            "width": 1920,
            "height": 1080,
            "split": "train",
            "labelCount": 6,
            "bytes": 284213,
            "error": null
        }
    ],
    "total": 1000,
    "hasMore": true,
    "classes": ["person", "forklift"],
    "errorCount": 0,
    "nextCursor": "65f1c0a2b3d4e5f601234567"
}
```

### Get Selected Images

```http
POST /api/datasets/{owner}/{dataset}/images
```

**Python SDK:** `client.datasets.selected_images(owner, dataset, image_ids=...)`

Returns the same image shape for up to 1,000 supplied image IDs, and accepts the same filter and URL query parameters
as the list operation.

```json
{
    "imageIds": ["65f1c0a2b3d4e5f601234567", "65f1c0a2b3d4e5f601234568"]
}
```

### Ingest Dataset Data

```http
POST /api/datasets/{owner}/{dataset}/ingest
```

**Python SDK:** `client.datasets.ingest(owner, dataset, body=...)`

Processes a completed upload, a remote archive, or a connected storage source into an existing dataset. Supply exactly
one source:

| Field            | Type   | Description                                                                                                                                                 |
| ---------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sessionId`      | string | Upload session from `POST /api/upload/signed-url`, already completed                                                                                        |
| `sourceUrl`      | string | Public HTTP or HTTPS URL of a ZIP, TAR, TAR.GZ, TGZ, or NDJSON file (max 4096 chars)                                                                        |
| `reference`      | object | A connected source: cloud storage (`provider: "cloud"`, `integrationId`, `target`, `prefix`) or On Premise (`provider: "local"`, `keyId`, `root`, `prefix`) |
| `targetSplit`    | string | `train`, `val`, or `test`; overrides the archive's split structure                                                                                          |
| `conflictPolicy` | string | `skip`, `keep_both`, or `replace` for filename or content conflicts                                                                                         |
| `classMapping`   | object | Maps incoming class names to a class index, an existing or new class name, or `null` to skip                                                                |
| `imageMetadata`  | object | Custom metadata keyed by each image's archive-relative path or NDJSON `file` value                                                                          |

Upload sessions are bound to a dataset by the `assetId` passed to `POST /api/upload/signed-url`, and ingest rejects a
session that belongs to a different dataset.

**Body (uploaded archive):**

```json
{
    "sessionId": "session_abc123",
    "targetSplit": "train"
}
```

**Body (remote archive or NDJSON):**

```json
{
    "sourceUrl": "https://example.com/my-dataset.zip"
}
```

**Body (importing labels on a later ingest):**

```json
{
    "sessionId": "session_abc123",
    "classMapping": { "person": 0, "automobile": "forklift", "background": null }
}
```

**Body (attaching per-image metadata):**

```json
{
    "sessionId": "session_abc123",
    "imageMetadata": {
        "airbus-wing.jpg": { "aircraft": { "family": "A350" }, "inspectionStatus": "reviewed" },
        "images/tail.jpg": { "aircraft": { "family": "A320" }, "inspectionSeverity": 2 }
    }
}
```

Metadata keys must match the normalized path inside the archive, including folders. For NDJSON imports, each record can
carry its own `metadata` object, which takes precedence over a matching `imageMetadata` entry. Archive paths are limited
to 1,024 characters, top-level metadata keys to 128 characters, and each metadata object — as well as the whole
`imageMetadata` map — to 500,000 serialized characters.

!!! note "Class Mapping"

    The first ingest creates classes from the archive automatically. On later ingests, archive classes omitted from
    `classMapping` fall back to a case-insensitive match against existing dataset classes. Labels are skipped only for
    classes explicitly mapped to `null` or without a matching existing class.

**Response (`201`):**

```json
{
    "jobId": "65f1c0a2b3d4e5f6012345aa",
    "status": "queued"
}
```

```mermaid
graph LR
    A[POST /api/datasets]:::start --> B[POST /api/upload/signed-url]:::proc
    B --> C[PUT archive to signed URL]:::proc
    C --> D[POST /api/upload/complete]:::proc
    D --> E["POST /api/datasets/{owner}/{dataset}/ingest"]:::proc
    E --> F[Process archive]:::proc
    F --> G[Dataset ready]:::out

    classDef start fill:#4CAF50,color:#fff
    classDef proc fill:#2196F3,color:#fff
    classDef out fill:#9C27B0,color:#fff
```

??? example "Upload one image with metadata using Python"

    The same code handles a group of images: add more files to the ZIP and matching entries to `imageMetadata`.

    ```python
    import io
    import zipfile
    from pathlib import Path

    import requests

    api = "https://platform.ultralytics.com/api"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    owner, dataset = "acme-vision", "warehouse"
    dataset_id = "65f1c0a2b3d4e5f601234567"  # id returned by POST /api/datasets
    image_path = Path("airbus-wing.jpg")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(image_path, image_path.name)
    data = archive.getvalue()

    signed = requests.post(
        f"{api}/upload/signed-url",
        headers=headers,
        json={
            "assetType": "datasets",
            "assetId": dataset_id,
            "filename": "images.zip",
            "contentType": "application/zip",
            "totalBytes": len(data),
        },
    )
    signed.raise_for_status()
    upload = signed.json()

    requests.put(upload["uploadUrl"], headers={"Content-Type": "application/zip"}, data=data).raise_for_status()
    requests.post(
        f"{api}/upload/complete",
        headers=headers,
        json={"sessionId": upload["sessionId"]},
    ).raise_for_status()

    ingest = requests.post(
        f"{api}/datasets/{owner}/{dataset}/ingest",
        headers=headers,
        json={
            "sessionId": upload["sessionId"],
            "imageMetadata": {
                "airbus-wing.jpg": {
                    "aircraft": {"family": "A350", "section": "wing"},
                    "inspectionStatus": "reviewed",
                }
            },
        },
    )
    ingest.raise_for_status()
    print(ingest.json())
    ```

---

## Images API

Inspect, annotate, move, and delete dataset images by their 24-character image ID. See
[Annotation documentation](../data/annotation.md).

### Get Image

```http
GET /api/images/{imageId}
```

**Python SDK:** `client.images.retrieve(image_id)`

Returns `metadata` (custom, user-defined), `properties` (filename, hash, dimensions, split, counts, timestamps),
`labels`, and the dataset's `classNames`.

### Update Image

```http
PATCH /api/images/{imageId}
```

**Python SDK:** `client.images.update(image_id, body=...)`

Replaces **either** the annotations **or** the custom metadata — send one of the two shapes, not both.

**Body (annotations):**

```json
{
    "labels": [
        { "classId": 0, "bbox": [0.5, 0.5, 0.2, 0.3] },
        { "classId": 1, "segments": [0.1, 0.2, 0.3, 0.2, 0.2, 0.4] }
    ]
}
```

**Body (metadata):**

```json
{
    "metadata": { "location": "strasbourg", "reviewed": true }
}
```

!!! info "Coordinate Format"

    Label coordinates use YOLO normalized values between 0 and 1. Bounding boxes use
    `[x_center, y_center, width, height]`. Segmentation labels use `segments`, a flattened list of polygon vertices
    `[x1, y1, x2, y2, ...]`. Pose labels use `keypoints` in one consistent flat shape: pairs `[x1, y1, x2, y2, ...]` or
    triples `[x1, y1, v1, x2, y2, v2, ...]`, where visibility conventionally uses 0, 1, or 2. Oriented boxes use `obb`
    corners. Saved coordinates are rounded to 5 decimal places, and an image accepts at most 10,000 annotations.

### Delete Image

```http
DELETE /api/images/{imageId}
```

**Python SDK:** `client.images.delete(image_id)`

Permanently deletes one image and its annotations.

### Auto-Annotate Image

```http
POST /api/images/{imageId}/predict
```

**Python SDK:** `client.images.predict(image_id, model_id=...)`

Runs YOLO inference on the image and returns predicted annotations. It does not save them — write the results back with
`PATCH /api/images/{imageId}` when you are happy with them.

| Field        | Type   | Required | Description                                                          |
| ------------ | ------ | -------- | -------------------------------------------------------------------- |
| `modelId`    | string | Yes      | Fully qualified model URI, `ul://{owner}/{project}/{model}`          |
| `confidence` | float  | No       | Confidence threshold, 0.01 – 1.0 (default: 0.25)                     |
| `iou`        | float  | No       | IoU threshold for non-maximum suppression, 0.0 – 0.95 (default: 0.7) |

**Response:** `success`, `predictions` (annotation objects), `modelUsed`, and `inferenceTime`. A model whose classes do
not match the dataset returns `422`.

### Bulk Move Images

```http
PATCH /api/images/bulk
```

**Python SDK:** `client.images.update_bulk(image_ids=..., split=...)`

Moves up to 1,000 images from one dataset into a different split.

```json
{
    "imageIds": ["65f1c0a2b3d4e5f601234567"],
    "split": "val",
    "conflictPolicy": "skip"
}
```

Filename or content conflicts return `409` until you choose a basket-wide `conflictPolicy` of `skip`, `keep_both`, or
`replace`. The response reports `modifiedCount`, `skippedCount`, and `targetSplit`.

### Bulk Delete Images

```http
DELETE /api/images/bulk
```

**Python SDK:** `client.images.delete_bulk(image_ids=...)`

```json
{
    "imageIds": ["65f1c0a2b3d4e5f601234567", "65f1c0a2b3d4e5f601234568"]
}
```

Deletes up to 1,000 images from a single dataset and returns `deletedCount` and `deletedImageIds`.

### Get Signed Image URLs

```http
POST /api/images/urls
```

**Python SDK:** `client.images.urls(image_ids=...)`

Returns temporary signed URLs for up to 100 image IDs from one dataset.

```json
{
    "imageIds": ["65f1c0a2b3d4e5f601234567"]
}
```

**Response:** `urls` and `thumbnails`, both keyed by image ID.

---

## Projects API

Organize your models into projects. Each model belongs to one project. See
[Projects documentation](../train/projects.md).

### List Projects

```http
GET /api/projects/{owner}
```

**Python SDK:** `client.projects.list(owner)`

**Query Parameters:**

| Parameter | Type | Description                                        |
| --------- | ---- | -------------------------------------------------- |
| `limit`   | int  | Maximum projects to return (default: 20, max: 500) |

### Get Project

```http
GET /api/projects/{owner}/{project}
```

**Python SDK:** `client.projects.retrieve(owner, project)`

Returns the `project` object, a `models` array of per-model summaries (status, metrics, epochs, weights, train args),
and `isOwner`.

### Create Project

```http
POST /api/projects
```

**Python SDK:** `client.projects.create(project=..., name=...)`

| Field         | Type   | Required | Description                                                |
| ------------- | ------ | -------- | ---------------------------------------------------------- |
| `project`     | string | Yes      | Project name used in Platform URLs                         |
| `name`        | string | Yes      | Display name (max 100 chars)                               |
| `description` | string | No       | Description (max 1000 chars)                               |
| `visibility`  | string | No       | `public` or `private`                                      |
| `tags`        | array  | No       | Up to 50 tags                                              |
| `license`     | string | No       | Project license identifier                                 |
| `metadata`    | object | No       | Custom JSON metadata                                       |
| `owner`       | string | No       | Team workspace handle; defaults to your personal workspace |

=== "cURL"

    ```bash
    curl -X POST \
      -H "Authorization: Bearer YOUR_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "project": "inspection",
        "name": "Inspection",
        "description": "Detection experiments",
        "metadata": {"department": "manufacturing", "cost_center": "cv-01"}
      }' \
      https://platform.ultralytics.com/api/projects
    ```

=== "Python"

    ```python
    created = client.projects.create(
        project="inspection",
        name="Inspection",
        description="Detection experiments",
        metadata={"department": "manufacturing", "cost_center": "cv-01"},
    )
    owner, project = created["owner"], created["project"]
    ```

**Response (`201`):** `id`, `owner`, `project`, `region`.

### Update Project

```http
PATCH /api/projects/{owner}/{project}
```

**Python SDK:** `client.projects.update(owner, project)`

Accepted fields: `name`, `description`, `visibility`, `metadata`, `tags`, `license`, `archived`, `iconColor`,
`iconLetter`, `viewPreferences`, and `starred`.

```json
{
    "metadata": { "department": "research", "program": "inspection" }
}
```

Send an empty `metadata` object (`{}`) to clear it. Project metadata uses the same 128-character key and
500,000-character serialized-object limits as dataset metadata.

### Delete Project

```http
DELETE /api/projects/{owner}/{project}
```

**Python SDK:** `client.projects.delete(owner, project)`

Moves the project and its models to [trash](../account/trash.md), returning `cascadedModels`.

### Clone Project

```http
POST /api/projects/{owner}/{project}/clone
```

**Python SDK:** `client.projects.clone(owner, project)`

Clones an accessible project and its completed models. The optional body accepts `project`, `name`, `description`,
`visibility`, `license`, and a destination `owner`.

---

## Models API

Manage trained YOLO models — view metrics, download weights, run inference, and monitor training. See
[Models documentation](../train/models.md).

### List Models in a Project

```http
GET /api/models/{owner}/{project}
```

**Python SDK:** `client.models.list(owner, project)`

**Query Parameters:**

| Parameter | Type | Description                                      |
| --------- | ---- | ------------------------------------------------ |
| `limit`   | int  | Maximum models to return (default: 20, max: 100) |

### Get Model

```http
GET /api/models/{owner}/{project}/{model}
```

**Python SDK:** `client.models.retrieve(owner, project, model)`

**Query Parameters:**

| Parameter  | Type | Description                                                             |
| ---------- | ---- | ----------------------------------------------------------------------- |
| `analysis` | int  | Set to `1` to return per-image validation analysis instead of the model |

The default response contains the `model` object — status, task, metrics, `trainArgs`, `trainResults`, `classNames`,
`computeCost`, `metadata`, and more — plus `isOwner`.

### Create Model

```http
POST /api/models
```

**Python SDK:** `client.models.create(body=...)`

Creates an untrained model record you can attach weights to or train.

| Field         | Type   | Required | Description                                                            |
| ------------- | ------ | -------- | ---------------------------------------------------------------------- |
| `project`     | string | Yes      | Destination project name                                               |
| `owner`       | string | No       | Workspace handle; defaults to your personal workspace                  |
| `model`       | string | No       | Model name used in Platform URLs; generated when omitted               |
| `name`        | string | No       | Display name (only accepted alongside `model`)                         |
| `description` | string | No       | Description (max 1000 chars)                                           |
| `task`        | string | No       | `detect`, `segment`, `semantic`, `depth`, `classify`, `pose`, or `obb` |
| `metadata`    | object | No       | Custom JSON metadata                                                   |
| `trainArgs`   | object | No       | Training arguments to record                                           |
| `metrics`     | object | No       | Metrics such as `mAP50`, `mAP50-95`, `precision`, `recall`             |
| `epochs`      | number | No       | Epoch count for an already-trained model                               |
| `version`     | string | No       | Version label (max 50 chars)                                           |

**Response (`201`):** `id`, `owner`, `project`, `model`, `region`.

!!! note "Model File Upload"

    To attach `.pt` weights, request a signed upload URL with `assetType: "models"` and this model's `id` as `assetId`,
    `PUT` the file to the returned URL, then call `POST /api/upload/complete` with the returned `sessionId`.

### Update Model

```http
PATCH /api/models/{owner}/{project}/{model}
```

**Python SDK:** `client.models.update(owner, project, model)`

Accepted fields include `name`, `description`, `color`, `metadata`, `status`, `license`, `datasetSlug`, `trainArgs`,
`trainResults`, `epochs`, `bestEpoch`, `bestFitness`, `version`, `trainingError`, and `starred`.

```json
{
    "metadata": { "release": "candidate-3", "reviewed": true }
}
```

Custom `metadata` is separate from training-owned fields such as `trainArgs`, `environment`, and `trainResults`, and
uses the same size limits as dataset metadata.

### Delete Model

```http
DELETE /api/models/{owner}/{project}/{model}
```

**Python SDK:** `client.models.delete(owner, project, model)`

Moves the model to [trash](../account/trash.md) for 30 days.

### Download Model Files

```http
GET /api/models/{owner}/{project}/{model}/files
```

**Python SDK:** `client.models.files(owner, project, model)`

Returns short-lived signed URLs for the model's weights.

```json
{
    "files": [
        {
            "name": "best.pt",
            "size": 6534127,
            "downloadUrl": "https://storage.googleapis.com/...&signature=..."
        }
    ]
}
```

### Clone Model

```http
POST /api/models/{owner}/{project}/{model}/clone
```

**Python SDK:** `client.models.clone(owner, project, model, project_body=...)`

Copies an accessible model into an existing project.

```json
{
    "owner": "acme-vision",
    "project": "inspection",
    "model": "v3-copy",
    "name": "V3 Copy",
    "description": "Cloned from a public model"
}
```

| Field         | Type   | Required | Description                                          |
| ------------- | ------ | -------- | ---------------------------------------------------- |
| `project`     | string | Yes      | Destination project name                             |
| `owner`       | string | No       | Destination workspace; defaults to your personal one |
| `model`       | string | No       | Destination model name                               |
| `name`        | string | No       | Destination display name                             |
| `description` | string | No       | Description for the clone                            |

### Run Inference

```http
POST /api/models/{owner}/{project}/{model}/predict
```

**Python SDK:** `client.models.predict(owner, project, model, body=...)`

Public models can be predicted without authentication. Private and shared models require an API key with access to the
parent project.

**Multipart Form:**

{% include "macros/platform-inference-parameters.md" %}

Provide either `file` or `source`. Depth models also accept `bits` (`8`, `12`, or `16`) to select the depth map's PNG
quantization. Requests that exceed the service's input limits return `413`.

=== "cURL"

    ```bash
    curl -X POST \
      -H "Authorization: Bearer YOUR_API_KEY" \
      -F "file=@image.jpg" \
      -F "conf=0.5" \
      https://platform.ultralytics.com/api/models/acme-vision/inspection/v3/predict
    ```

=== "Python"

    ```python
    with open("image.jpg", "rb") as f:
        resp = client.models.predict("acme-vision", "inspection", "v3", body={"file": f, "conf": 0.5})
    results = resp["images"][0]["results"]
    ```

**Response:**

Each entry in `images` carries `shape`, `speed`, `results`, and, for dense-prediction tasks, a `semantic_mask` or
`depth` PNG payload (depth values are `pixel × max / divisor`, with divisor 255 for the default 8-bit map and 65535 when
`bits` is 12 or 16). The `metadata` object reports image count, function timings, task, and service versions. Internal
model paths are never returned.

```json
{
    "images": [
        {
            "shape": [1080, 1920],
            "speed": { "preprocess": 2.1, "inference": 12.4, "postprocess": 1.3 },
            "results": [
                {
                    "class": 0,
                    "name": "person",
                    "confidence": 0.92,
                    "box": { "x1": 100, "y1": 50, "x2": 300, "y2": 400 }
                }
            ]
        }
    ],
    "metadata": {
        "imageCount": 1,
        "functionTimeAlive": 184.2,
        "functionTimeCall": 0.31,
        "task": "detect",
        "version": { "ultralytics": "8.4.120" }
    }
}
```

### Check Training Progress

```http
GET /api/models/{owner}/{project}/{model}/training
```

**Python SDK:** `client.models.training(owner, project, model)`

Returns `job`, containing status, epoch progress, timing, compute details, train args, epoch metrics, and safe error
details, or `null` when the model has never been trained. Models in public projects are readable without
authentication.

### Cancel Training

```http
DELETE /api/models/{owner}/{project}/{model}/training
```

**Python SDK:** `client.models.delete_training(owner, project, model)`

Terminates the running compute instance and marks the job cancelled. Returns `409` when training is no longer active.

---

## Training API

Launch YOLO training on cloud GPUs and monitor progress in real time. See
[Cloud Training documentation](../train/cloud-training.md).

```mermaid
graph LR
    A[POST /api/training/start]:::start --> B[Job Created]:::proc
    B --> C{Training}:::decide
    C -->|progress| D[GET .../training]:::proc
    C -->|cancel| E[DELETE .../training]:::error
    C -->|complete| F[Model Ready]:::out
    F --> G[Deploy or Export]:::proc

    classDef start fill:#4CAF50,color:#fff
    classDef proc fill:#2196F3,color:#fff
    classDef decide fill:#FF9800,color:#fff
    classDef out fill:#9C27B0,color:#fff
    classDef error fill:#F44336,color:#fff
```

### Get GPU Availability

```http
GET /api/training/gpu-availability
```

**Python SDK:** `client.training.gpu_availability()`

Returns current stock status keyed by GPU ID. Public and unauthenticated; pass `managed=true` to include managed
training capacity, which does require an API key.

### Start Training

```http
POST /api/training/start
```

**Python SDK:** `client.training.start(model_id=..., train_args=...)`

| Field                   | Type    | Required | Description                                                         |
| ----------------------- | ------- | -------- | ------------------------------------------------------------------- |
| `modelId`               | string  | Yes      | ID of the model to train                                            |
| `trainArgs`             | object  | Yes      | YOLO training arguments; `model`, `data`, and `epochs` are required |
| `gpuType`               | string  | No       | Cloud GPU to use (default: `rtx-4090`)                              |
| `captureDatasetVersion` | boolean | No       | Save an immutable dataset version for this run (default: `false`)   |

=== "cURL"

    ```bash
    curl -X POST \
      -H "Authorization: Bearer YOUR_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "modelId": "65f1c0a2b3d4e5f601234599",
        "gpuType": "rtx-4090",
        "trainArgs": {
          "model": "yolo26n.pt",
          "data": "ul://acme-vision/datasets/warehouse",
          "epochs": 100,
          "imgsz": 640,
          "batch": 16
        }
      }' \
      https://platform.ultralytics.com/api/training/start
    ```

=== "Python"

    ```python
    resp = client.training.start(
        model_id="65f1c0a2b3d4e5f601234599",
        gpu_type="rtx-4090",
        train_args={
            "model": "yolo26n.pt",
            "data": "ul://acme-vision/datasets/warehouse",
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
        },
    )
    print(resp["billing"]["estimatedCostDisplay"])
    ```

**Response:**

```json
{
    "modelId": "65f1c0a2b3d4e5f601234599",
    "status": "starting",
    "gpuType": "rtx-4090",
    "estimatedCost": { "pricePerHour": 0.69, "gpuMemoryGb": 24 },
    "billing": {
        "estimatedCostCents": 138,
        "estimatedCostDisplay": "$1.38",
        "balanceCents": 2500
    }
}
```

Training returns `402` when your credit balance is too low and `503` when no capacity is available for the requested
GPU.

!!! note "GPU Types"

    26 GPU types are available, from `rtx-2000-ada` through `b300`, including `rtx-4090`, `l40s`, `a100-80gb-pcie`,
    `a100-80gb-sxm`, `rtx-pro-6000`, `h100-sxm`, `h200-sxm`, and `b200`. See
    [Cloud Training](../train/cloud-training.md) for the full list with pricing.

---

## Exports API

Convert models to optimized formats like ONNX, TensorRT, CoreML, and LiteRT for edge deployment. See
[Deploy documentation](../deploy/index.md).

### List Exports

```http
GET /api/models/{owner}/{project}/{model}/exports
```

**Python SDK:** `client.exports.list(owner, project, model)`

**Query Parameters:**

| Parameter | Type   | Description                                                                      |
| --------- | ------ | -------------------------------------------------------------------------------- |
| `status`  | string | Filter by `queued`, `starting`, `running`, `completed`, `failed`, or `cancelled` |
| `limit`   | int    | Maximum exports to return (default: 20, max: 100)                                |

### Create Export

```http
POST /api/models/{owner}/{project}/{model}/exports
```

**Python SDK:** `client.exports.create(owner, project, model, format=...)`

| Field     | Type   | Required    | Description                                                                                                                                                                                               |
| --------- | ------ | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`  | string | Yes         | Target export format (see table below)                                                                                                                                                                    |
| `gpuType` | string | Conditional | Required when `format` is `engine`; use a supported [GPU or Jetson target](../train/models.md#nvidia-jetson-tensorrt-targets)                                                                             |
| `args`    | object | No          | Export options: `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `conf`, `iou`, `batch`, `workspace`, `nms`, `optimize`, `keras`, and `name` (device target for RKNN, QNN, Hailo, and Ascend formats) |

=== "cURL"

    ```bash
    curl -X POST \
      -H "Authorization: Bearer YOUR_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"format": "onnx", "args": {"imgsz": 640, "quantize": 16}}' \
      https://platform.ultralytics.com/api/models/acme-vision/inspection/v3/exports
    ```

=== "Python"

    ```python
    export = client.exports.create("acme-vision", "inspection", "v3", format="onnx", args={"imgsz": 640, "quantize": 16})
    status = client.exports.retrieve("acme-vision", "inspection", "v3", export["id"])
    print(status["export"]["status"])
    ```

**Response (`201`):** `id`, `format`, `status` (`queued` or `running`), `gpuType`, `region`. An equivalent export that
is already in flight returns `409`.

**Supported Formats:**

Use the `format` argument from the shared export table below. PyTorch is the source format and is not an API export
target.

{% include "macros/export-table.md" %}

### Get Export Status

```http
GET /api/models/{owner}/{project}/{model}/exports/{exportId}
```

**Python SDK:** `client.exports.retrieve(owner, project, model, export_id)`

Returns the `export` object with `status`, `format`, `args`, `gpuType`, timestamps, and — once complete — a `file`
object containing `size`, `downloadUrl`, and `downloadFilename`.

### Cancel or Delete Export

```http
DELETE /api/models/{owner}/{project}/{model}/exports/{exportId}
```

**Python SDK:** `client.exports.delete(owner, project, model, export_id)`

Cancels an active export or deletes a finished one and its file. The response reports which happened:

```json
{
    "success": true,
    "action": "cancelled"
}
```

---

## Deployments API

Deploy models to dedicated inference endpoints with health checks and monitoring. See
[Endpoints documentation](../deploy/endpoints.md).

```mermaid
graph LR
    A[Create]:::start --> B[Deploying]:::proc
    B --> C[Ready]:::out
    C -->|action stop| D[Stopped]:::extern
    C -->|action replace| B
    D -->|action start| C
    C -->|delete| E[Deleted]:::error
    D -->|delete| E
    C -->|predict| F[Inference Results]:::out

    classDef start fill:#4CAF50,color:#fff
    classDef proc fill:#2196F3,color:#fff
    classDef out fill:#9C27B0,color:#fff
    classDef error fill:#F44336,color:#fff
    classDef extern fill:#607D8B,color:#fff
```

### List Deployments

```http
GET /api/deployments/{owner}
```

**Python SDK:** `client.deployments.list(owner)`

**Query Parameters:**

| Parameter | Type   | Description                                                          |
| --------- | ------ | -------------------------------------------------------------------- |
| `status`  | string | `creating`, `deploying`, `ready`, `stopping`, `stopped`, or `failed` |
| `model`   | string | Filter by `{project}/{model}`, for example `inspection/v3`           |
| `limit`   | int    | Maximum deployments to return (default: 20, max: 100)                |

Anonymous callers must filter by one public model; listing a whole workspace requires authentication.

### Create Deployment

```http
POST /api/deployments/{owner}
```

**Python SDK:** `client.deployments.create(owner, project=..., model=..., deployment=..., name=..., region=...)`

**Body:**

```json
{
    "project": "inspection",
    "model": "v3",
    "deployment": "edge-1",
    "name": "Edge 1",
    "region": "us-central1"
}
```

| Field        | Type   | Required | Description                            |
| ------------ | ------ | -------- | -------------------------------------- |
| `project`    | string | Yes      | Project containing the model           |
| `model`      | string | Yes      | Model to deploy                        |
| `deployment` | string | Yes      | Deployment name used in Platform URLs  |
| `name`       | string | Yes      | Display name                           |
| `region`     | string | Yes      | One of 42 supported deployment regions |

**Response (`201`):** `id`, `deployment`, `status` (`creating`), `message`, and `region`.

!!! note "Resource Sizing"

    CPU, memory, and instance scaling are managed by the Platform from your plan limits, and the create request does not
    accept a resource configuration. The current values are returned in the `resources` object on every deployment read.

!!! tip "Region Selection"

    Choose a region close to your users for lowest latency. The Platform UI shows latency estimates for all 42
    available regions.

### Get Deployment

```http
GET /api/deployments/{owner}/{deployment}
```

**Python SDK:** `client.deployments.retrieve(owner, deployment)`

Returns the `deployment` object with `status`, `statusMessage`, `region`, `serviceUrl`, and `resources`.

### Start, Stop, or Replace a Deployment

```http
PATCH /api/deployments/{owner}/{deployment}
```

**Python SDK:** `client.deployments.update(owner, deployment, body=...)`

A single `action` field selects the operation:

=== "Start"

    ```json
    { "action": "start" }
    ```

=== "Stop"

    ```json
    { "action": "stop" }
    ```

=== "Replace model"

    ```json
    {
        "action": "replace",
        "project": "inspection",
        "model": "v4",
        "name": "Edge 1 (v4)"
    }
    ```

Replacing rolls out a new revision while preserving the deployment ID, region, and endpoint URL; the existing revision
stays live if the rollout fails. The replacement model must be a completed model with weights that your key can access.
Completed operations return `200` with `status` `ready` or `stopped`; operations still rolling out return `202` with
`deploying` or `stopping`.

### Delete Deployment

```http
DELETE /api/deployments/{owner}/{deployment}
```

**Python SDK:** `client.deployments.delete(owner, deployment)`

Permanently removes the inference endpoint.

### Health Check

```http
GET /api/deployments/{owner}/{deployment}/health
```

**Python SDK:** `client.deployments.health(owner, deployment)`

Pings and warms the endpoint, returning `healthy`, `latencyMs`, and the upstream `status` code.

### Run Inference on a Deployment

```http
POST /api/deployments/{owner}/{deployment}/predict
```

**Python SDK:** `client.deployments.predict(owner, deployment, body=...)`

Routes an image or video through the dedicated endpoint. The request and response contracts match
[model inference](#run-inference).

**Multipart Form:**

{% include "macros/platform-inference-parameters.md" %}

### Get Metrics

```http
GET /api/deployments/{owner}/{deployment}/metrics
```

**Python SDK:** `client.deployments.metrics(owner, deployment)`

**Query Parameters:**

| Parameter   | Type    | Description                                                                    |
| ----------- | ------- | ------------------------------------------------------------------------------ |
| `range`     | string  | `1h`, `6h`, `24h` (default), `7d`, or `30d`                                    |
| `sparkline` | boolean | Return the compact dashboard summary instead of full series (default: `false`) |

The full response contains `summary` (request totals, error rate, average and p50/p95/p99 latency) and `timeSeries`
(requests, errors, latency, CPU, memory, instance count). The sparkline response returns `requests24h`,
`totalRequests`, `errorRate`, and `avgLatencyMs`.

### Get Logs

```http
GET /api/deployments/{owner}/{deployment}/logs
```

**Python SDK:** `client.deployments.logs(owner, deployment)`

**Query Parameters:**

| Parameter   | Type   | Description                                                                                      |
| ----------- | ------ | ------------------------------------------------------------------------------------------------ |
| `severity`  | string | Comma-separated: `DEBUG`, `INFO`, `NOTICE`, `WARNING`, `ERROR`, `CRITICAL`, `ALERT`, `EMERGENCY` |
| `limit`     | int    | Entries to return (default: 50, max: 200)                                                        |
| `pageToken` | string | Pagination token from a previous response                                                        |

---

## Trash API

View, restore, and permanently delete soft-deleted projects, datasets, and models. Items are purged automatically after
30 days. See [Trash documentation](../account/trash.md).

### List Trash

```http
GET /api/trash
```

**Python SDK:** `client.lifecycle.trash()`

**Query Parameters:**

| Parameter | Type   | Description                                       |
| --------- | ------ | ------------------------------------------------- |
| `type`    | string | `all` (default), `project`, `dataset`, or `model` |
| `page`    | int    | Page number (default: 1)                          |
| `limit`   | int    | Items per page (default: 50, max: 200)            |

The response includes `items` (each with `daysRemaining`), `total`, `page`, `limit`, `totalPages`, and a `summary`
with totals by type.

### Restore Item

```http
POST /api/trash
```

**Python SDK:** `client.lifecycle.restore(id=..., type=...)`

```json
{
    "id": "65f1c0a2b3d4e5f601234567",
    "type": "dataset"
}
```

Restoring a project also restores the models that were trashed with it, reported as `restoredModels`.

### Permanently Delete

```http
DELETE /api/trash
```

**Python SDK:** `client.lifecycle.delete_trash(body=...)`

Delete one item:

```json
{
    "id": "65f1c0a2b3d4e5f601234567",
    "type": "dataset"
}
```

Or empty the whole trash:

```json
{
    "all": true
}
```

The response reports `deletedCount`, plus `cascadedModels` and `survivingDeployments` where relevant.

!!! warning "Irreversible"

    Permanent deletion cannot be undone. The resource and all associated data are removed.

---

## Upload API

Upload files directly to cloud storage using signed URLs. Completing a model upload attaches its weights; completing a
dataset archive upload records the session, which you then pass to
[dataset ingest](#ingest-dataset-data). See [Data documentation](../data/index.md).

### Get Signed Upload URL

```http
POST /api/upload/signed-url
```

**Python SDK:** `client.upload.signed_url(body=...)`

**Body:**

```json
{
    "assetType": "datasets",
    "assetId": "65f1c0a2b3d4e5f601234567",
    "filename": "warehouse.zip",
    "contentType": "application/zip",
    "totalBytes": 52428800
}
```

| Field         | Type   | Required | Description                                 |
| ------------- | ------ | -------- | ------------------------------------------- |
| `assetType`   | string | Yes      | `datasets`, `models`, `images`, or `videos` |
| `assetId`     | string | Yes      | ID of the target dataset or model           |
| `filename`    | string | Yes      | Original filename (max 256 chars)           |
| `contentType` | string | Yes      | MIME type                                   |
| `totalBytes`  | number | Yes      | File size in bytes                          |

!!! note "Dataset Archive Filenames"

    When `assetType` is `datasets`, `filename` must end in `.zip`, `.tar`, `.tar.gz`, `.tgz`, or `.ndjson`. Package
    loose images into an archive before uploading.

**Response:**

```json
{
    "sessionId": "session_abc123",
    "uploadUrl": "https://storage.googleapis.com/...&signature=...",
    "expiresAt": "2026-02-22T12:00:00Z"
}
```

Upload the file with a `PUT` request to `uploadUrl`, using the same `Content-Type` you declared.

### Complete Upload

```http
POST /api/upload/complete
```

**Python SDK:** `client.upload.complete(session_id=...)`

```json
{
    "sessionId": "session_abc123",
    "checksum": "<optional sha-256 hex>"
}
```

**Response:** `success` and a `file` object with `size` and `contentType`. For models this attaches the weights; for
dataset archives, call [ingest](#ingest-dataset-data) next to start processing.

---

## Storage Integrations API

Connect read-only Google Cloud Storage, Amazon S3, or Azure Blob Storage accounts and browse them as dataset sources.
See [Integrations documentation](../integrations/index.md).

### List Integrations

```http
GET /api/integrations/buckets
```

**Python SDK:** `client.storage_integrations.list()`

Returns `integrations`, each with `id`, `provider`, `credentialIdentity`, `targets`, and `createdAt`. Credentials are
never returned.

### Discover Locations

```http
POST /api/integrations/buckets/discover
```

**Python SDK:** `client.storage_integrations.discover(body=...)`

Lists the buckets or containers readable with the supplied credentials, without saving them.

=== "Google Cloud Storage"

    ```json
    {
        "provider": "gcs",
        "credentials": {
            "client_email": "svc@project.iam.gserviceaccount.com",
            "private_key": "-----BEGIN PRIVATE KEY-----\n...",
            "project_id": "my-project"
        }
    }
    ```

=== "Amazon S3"

    ```json
    {
        "provider": "s3",
        "credentials": {
            "access_key_id": "AKIA...",
            "secret_access_key": "...",
            "region": "us-east-1"
        }
    }
    ```

=== "Azure Blob Storage"

    ```json
    {
        "provider": "azure",
        "credentials": {
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=..."
        }
    }
    ```

**Response:** `{"targets": ["my-bucket", "another-bucket"]}`

### Connect Storage

```http
POST /api/integrations/buckets
```

**Python SDK:** `client.storage_integrations.create(body=...)`

Same credential shapes as discovery, plus a required `targets` array of 1-50 bucket or container names. Returns `201`
with the stored integration. Temporary S3 credentials (`ASIA` access keys) are rejected.

### Browse Objects

```http
GET /api/integrations/buckets/{id}/objects
```

**Python SDK:** `client.storage_integrations.objects(id, target=...)`

**Query Parameters:**

| Parameter | Type   | Required | Description                                  |
| --------- | ------ | -------- | -------------------------------------------- |
| `target`  | string | Yes      | Bucket or container name                     |
| `prefix`  | string | No       | Folder prefix (max 1024 chars)               |
| `cursor`  | string | No       | Provider pagination cursor from a prior page |

Returns `entries` (each `kind` is `folder` or `file`) and an optional `cursor` for the next page.

### Disconnect Storage

```http
DELETE /api/integrations/buckets/{id}
```

**Python SDK:** `client.storage_integrations.delete(id)`

Removes the saved credentials without deleting provider data. Connected datasets remain visible, but their files stay
unavailable until the same storage account is reconnected. Requires workspace admin access.

---

## Dataset Import API

Import datasets from third-party services. See [Roboflow integration](../integrations/roboflow.md).

### Preview a Roboflow Import

```http
POST /api/integrations/roboflow/preview
```

**Python SDK:** `client.datasets.preview_roboflow(api_key=...)`

Resolves a Roboflow API key into an import plan: workspace details, `newDatasets` that would be imported, counts of
skipped, unsupported, and unresolved projects, `bytesTotal`, and your `storage` headroom. The Roboflow API key is read
from the body and is not persisted.

```json
{
    "apiKey": "ROBOFLOW_API_KEY"
}
```

### Import from Roboflow

```http
POST /api/integrations/roboflow/import
```

**Python SDK:** `client.datasets.import_roboflow(api_key=..., items=...)`

Queues ingest jobs for up to 500 selected Roboflow project versions, using the items returned by the preview.

```json
{
    "apiKey": "ROBOFLOW_API_KEY",
    "items": [
        {
            "workspace": "my-workspace",
            "projectId": "warehouse-safety",
            "projectName": "Warehouse Safety",
            "projectType": "object-detection",
            "latestVersion": 4
        }
    ]
}
```

**Response (`201`):** `imported`, `failed`, and `skipped` arrays. Imports require storage headroom, and each dataset
must fit your plan's per-import size limit.

---

## Account API

Inspect your Platform account, keys, storage, and public profiles. See [Settings documentation](../account/settings.md).

### Account Summary

```http
GET /api/account/summary
```

**Python SDK:** `client.account.summary()`

Returns the plan, credit balance, and resource counts for the workspace that issued the key.

```json
{
    "username": "acme-vision",
    "name": "Acme Vision",
    "accountType": "team",
    "plan": "pro",
    "creditsCents": 2500,
    "counts": { "projects": 4, "datasets": 7, "models": 21 },
    "teams": []
}
```

!!! note "Team List"

    `teams` is populated for browser sessions. API-key responses return an empty list, because a key is already scoped
    to a single workspace.

### List API Keys

```http
GET /api/api-keys
```

**Python SDK:** `client.account.api_keys()`

Returns `keys` with `keyId`, `name`, `keyPrefix`, and `createdAt` for the key's workspace. API-key-authenticated
requests receive metadata only; full key values are shown to the workspace owner in
[Settings > API Keys](../account/api-keys.md) in the Platform UI, which is also where keys are created and revoked.

### Check Storage Usage

```http
GET /api/storage
```

**Python SDK:** `client.account.storage()`

**Query Parameters:**

| Parameter | Type    | Description                                                  |
| --------- | ------- | ------------------------------------------------------------ |
| `details` | boolean | Include the ten largest storage consumers (default: `false`) |

**Response:**

```json
{
    "tier": "pro",
    "usage": {
        "storage": { "current": 1073741824, "limit": 107374182400, "percent": 1.0 },
        "datasets": { "current": 536870912, "limit": 107374182400, "percent": 0.5 }
    },
    "breakdown": {
        "byCategory": {
            "datasets": { "bytes": 536870912, "count": 2 },
            "models": { "bytes": 268435456, "count": 4 },
            "exports": { "bytes": 268435456, "count": 3 }
        },
        "topItems": [
            {
                "_id": "65f1c0a2b3d4e5f601234567",
                "name": "Warehouse",
                "slug": "warehouse",
                "sizeBytes": 536870912,
                "type": "dataset"
            }
        ]
    },
    "region": "us",
    "username": "acme-vision",
    "updatedAt": "2026-01-15T10:00:00Z"
}
```

### Get a Public User Profile

```http
GET /api/users
```

**Python SDK:** `client.account.profile(username=...)`

**Query Parameters:**

| Parameter  | Type   | Required | Description         |
| ---------- | ------ | -------- | ------------------- |
| `username` | string | Yes      | Username to look up |

Returns the public `user` profile with `followerCount` and, for authenticated callers, `isFollowed`.

### Follow or Unfollow a User

```http
PATCH /api/users
```

**Python SDK:** `client.account.follow(username=..., followed=...)`

```json
{
    "username": "target-user",
    "followed": true
}
```

**Response:** `followed` and the updated `followerCount`.

---

## Billing API

Check plan usage and your credit ledger. See [Billing documentation](../account/billing.md).

!!! note "Currency Units"

    Billing amounts are integers in US cents, where `100 = $1.00`.

### View Plan and Usage

```http
GET /api/billing/usage-summary
```

**Python SDK:** `client.billing.usage_summary()`

Returns `plan` (ID, status, billing cycle, period end), `metrics` (storage limit and usage), `trainingCredit`,
`features`, `creditsCents`, and seat counts.

### View Transactions

```http
GET /api/billing/transactions
```

**Python SDK:** `client.billing.transactions()`

**Query Parameters:**

| Parameter | Type   | Description                               |
| --------- | ------ | ----------------------------------------- |
| `from`    | string | Earliest transaction timestamp (ISO 8601) |
| `to`      | string | Latest transaction timestamp (ISO 8601)   |

Each transaction includes `id`, `type` (such as `purchase`, `training`, `monthly_grant`, or `refund`), `amountCents`,
`balanceAfter`, `createdAt`, an optional `receiptUrl`, and model context for training charges. Internal billing details are never returned.

---

## Explore API

Search public projects and datasets shared by the community. See [Explore documentation](../explore.md).

### Search Public Content

```http
GET /api/explore/search
```

**Python SDK:** `client.explore.search()`

**Query Parameters:**

| Parameter | Type    | Description                                                                                       |
| --------- | ------- | ------------------------------------------------------------------------------------------------- |
| `q`       | string  | Search term (max 200 chars)                                                                       |
| `type`    | string  | `all` (default), `projects`, or `datasets`                                                        |
| `sort`    | string  | `newest` (default), `oldest`, `stars`, `name-asc`, `name-desc`, `count-desc`, `count-asc`         |
| `offset`  | int     | Results to skip (default: 0)                                                                      |
| `limit`   | int     | Maximum results per resource type (default: 20, max: 100)                                         |
| `task`    | string  | Comma-separated task filters: `detect`, `segment`, `semantic`, `depth`, `classify`, `pose`, `obb` |
| `author`  | string  | Owner username filter                                                                             |
| `starred` | boolean | Return only content starred by the authenticated caller; requires an API key                      |

**Response:** `projects`, `datasets`, and `hasMore`.

```bash
curl "https://platform.ultralytics.com/api/explore/search?type=datasets&task=detect&sort=stars&limit=20"
```

---

## Python SDK

[`ultralytics-platform`](https://pypi.org/project/ultralytics-platform/) is a typed Python client generated from the
OpenAPI contract, with one method per endpoint (`client.datasets.list`, `client.models.predict`,
`client.exports.create`, ...). Every method accepts the path parameters positionally, other inputs as keyword arguments,
and optional per-request `timeout` and `extra_headers`.

```bash
pip install "ultralytics-platform>=0.1.5" # Python 3.11+
```

```python
from ultralytics_platform import Platform

with Platform() as client:  # reads ULTRALYTICS_API_KEY
    dataset = client.datasets.retrieve("acme-vision", "warehouse")
    images = client.datasets.images("acme-vision", "warehouse", limit=10)
    export = client.exports.create("acme-vision", "inspection", "v3", format="onnx")
```

`AsyncPlatform` exposes the same resource tree for `async`/`await` code, unsuccessful responses raise `APIError` with
`status_code`, `body`, and parsed `json`, and connection failures raise `APIConnectionError`. See the
[SDK repository](https://github.com/ultralytics/sdk) for the full README.

## Python Integration

For training and inference workflows, use the Ultralytics Python package, which handles authentication, uploads, and
real-time metric streaming automatically.

### Installation & Setup

```bash
pip install "ultralytics>=8.4.120"
```

Verify installation:

```bash
yolo check
```

### Authentication

=== "CLI (Recommended)"

    ```bash
    yolo login YOUR_API_KEY
    ```

=== "Environment Variable"

    ```bash
    export ULTRALYTICS_API_KEY=YOUR_API_KEY
    ```

=== "In Code"

    ```python
    from ultralytics import settings

    settings.update({"api_key": "YOUR_API_KEY"})
    ```

### Using Platform Datasets

Reference datasets with `ul://` URIs:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Train on your Platform dataset
model.train(
    data="ul://your-username/datasets/your-dataset",
    epochs=100,
    imgsz=640,
)
```

**URI Format:**

| Pattern                            | Description    |
| ---------------------------------- | -------------- |
| `ul://username/datasets/slug`      | Dataset        |
| `ul://username/project-name`       | Project        |
| `ul://username/project/model-name` | Specific model |
| `ul://ultralytics/yolo26/yolo26n`  | Official model |

### Pushing to Platform

Send results to a Platform project:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Results automatically sync to Platform
model.train(
    data="coco8.yaml",
    epochs=100,
    project="your-username/my-project",
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
model.export(format="onnx", imgsz=640, quantize=16)

# Export to TensorRT
model.export(format="engine", imgsz=640, quantize=16)

# Export to CoreML
model.export(format="coreml", imgsz=640)  # use imgsz=224 for classification
```

**Validation:**

```python
metrics = model.val(data="ul://username/datasets/my-dataset")

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

---

## FAQ

### How do I address a dataset or model in the API?

Use the same owner and name segments that appear in the Platform URL. A model at
`https://platform.ultralytics.com/acme-vision/inspection/v3` is `GET /api/models/acme-vision/inspection/v3`. Database
IDs are still returned in responses (as `id`), and a few routes take them directly — image routes take an `imageId`,
uploads take an `assetId`, and `POST /api/training/start` takes a `modelId`.

### How do I paginate large results?

It depends on the collection. Most list endpoints accept `limit`:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "https://platform.ultralytics.com/api/datasets/acme-vision?limit=50"
```

Dataset images, clustering, and Explore search use `offset` with `limit` and report `hasMore`:

```bash
curl "https://platform.ultralytics.com/api/explore/search?type=datasets&offset=20&limit=20&sort=stars"
```

Very large image sets are best walked with the cursor returned as `nextCursor`:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "https://platform.ultralytics.com/api/datasets/acme-vision/warehouse/images?limit=1000&includeTotal=false&cursor=LAST_IMAGE_ID"
```

Trash uses `page`, and deployment logs use the opaque `pageToken` returned as `nextPageToken`.

### Can I use the API without an SDK?

Yes. Every operation on this page is a plain HTTPS request, and the complete contract is published as OpenAPI 3.2 at
[platform.ultralytics.com/openapi.json](https://platform.ultralytics.com/openapi.json), which you can feed to a client
generator in any language. The [`ultralytics-platform`](#python-sdk) package is exactly that: a typed client generated
from the contract, while the `ultralytics` package adds real-time metric streaming and automatic model uploads on top of
training and inference. Browser-session-only account flows, such as billing checkout and team management, remain in the
Platform UI.

### How do I handle rate limits?

Use the `Retry-After` header from the `429` response to wait the right amount of time:

```python
import time

import requests


def api_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        if response.status_code != 429:
            return response
        wait = int(response.headers.get("Retry-After", 2**attempt))
        time.sleep(wait)
    raise RuntimeError("Rate limit exceeded")
```

### Why did my request return 403 instead of 404?

`404` means the resource does not exist or is not visible to your key at all. `403` means the resource was found but the
action needs more access than your key has — editor access to modify a dataset, owner access to delete a deployment,
admin access to disconnect storage, or a higher plan or quota for exports and deployments.

### Which endpoints work without an API key?

Reading public datasets, projects, and models, including their images, signed image URLs, class statistics, embedding
status, clustering layout, and export list; checking training progress on a public model; downloading a public model's
files; running inference on a public model; looking up a public user profile; listing deployments filtered to one public
model; and searching Explore. `GET /api/training/gpu-availability` is fully public unless you request managed capacity.
Everything else requires a key, and supplying one on a public endpoint also reveals your private resources.
