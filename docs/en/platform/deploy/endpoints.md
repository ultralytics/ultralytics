---
plans: [free, pro, enterprise]
comments: true
description: Deploy YOLO models to dedicated endpoints in 42 global regions with scale-to-zero behavior and monitoring on Ultralytics Platform.
keywords: Ultralytics Platform, deployment, endpoints, YOLO, production, scaling, global regions
---

# Dedicated Endpoints

[Ultralytics Platform](https://platform.ultralytics.com) enables deployment of YOLO models to dedicated endpoints in 42 global regions. Each endpoint is a single-tenant service with scale-to-zero behavior, a unique endpoint URL, and independent monitoring.

![Ultralytics Platform Model Deploy Tab With Region Map And Table](https://cdn.ul.run/i/176e99f44ab36318aec89d8a5309376f.avif)<!-- screenshot -->

## Create Endpoint

### From the Deploy Tab

Deploy a model from its `Deploy` tab:

1. Navigate to your model
2. Click the **Deploy** tab
3. Review the world map and the region table, which is sorted by measured latency from your location
4. Click **Deploy** in the region row you want to use

The deployment name is auto-generated from the model name and region city (e.g., `yolo26n-iowa`).

### From the Deployments Page

Create a deployment from the global `Deploy` page in the sidebar:

1. Click **New Deployment**
2. Select a model from the model selector
3. Select a region from the map or table
4. Review the editable, auto-generated deployment name and the fixed resource defaults
5. Click **Deploy Model**

![Ultralytics Platform New Deployment Dialog With Model Selector And Region Map](https://cdn.ul.run/i/d0447123225bbac5c67ae7aee0f15da2.avif)<!-- screenshot -->

### Deployment Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Creating: Deploy
    Creating --> Deploying: Container starting
    Deploying --> Ready: Health check passed
    Ready --> Stopping: Stop
    Stopping --> Stopped: Stopped
    Stopped --> Ready: Start
    Ready --> [*]: Delete
    Stopped --> [*]: Delete
    Creating --> Failed: Error
    Deploying --> Failed: Error
    Failed --> [*]: Delete

    classDef proc fill:#2196F3,color:#fff
    classDef out fill:#9C27B0,color:#fff
    classDef error fill:#F44336,color:#fff
    classDef extern fill:#607D8B,color:#fff
    class Creating,Deploying,Stopping proc
    class Ready out
    class Failed error
    class Stopped extern
```

Connect [Slack alerts](../integrations/slack.md) to receive a message when a deployment becomes ready or fails to start.

### Region Selection

Choose from 42 regions worldwide. The interactive region map and table show:

- **Region pins**: Color-coded by latency on a green-to-red gradient (faster regions are greener, slower regions are redder)
- **Deployed regions**: Highlighted with a "Deployed" badge
- **Deploying regions**: Animated pulse indicator
- **Bidirectional highlighting**: Hover on the map highlights the table row, and vice versa

![Ultralytics Platform Deploy Tab Region Latency Table Sorted By Latency](https://cdn.ul.run/i/b763bfb3b965aac1e274bfed782a82e8.avif)<!-- screenshot -->
The region table on the model `Deploy` tab includes:

| Column       | Description                              |
| ------------ | ---------------------------------------- |
| **Location** | City and country with flag icon          |
| **Zone**     | Region identifier                        |
| **Latency**  | Measured ping time (median of 3 pings)   |
| **Distance** | Distance from your location in km        |
| **Actions**  | Deploy button or "Deployed" status badge |

!!! note "New Deployment Dialog"

    The `New Deployment` dialog (from the global `Deploy` page) shows a simpler region table with only Location, Latency, and Select columns.

!!! tip "Choose Wisely"

    Select the region closest to your users for lowest latency. Use the **Rescan** button to re-measure latency from your current location.

## Available Regions

=== "Americas (14)"

    | Zone                    | Location               |
    | ----------------------- | ---------------------- |
    | us-central1             | Iowa, USA              |
    | us-east1                | South Carolina, USA    |
    | us-east4                | Northern Virginia, USA |
    | us-east5                | Columbus, USA          |
    | us-south1               | Dallas, USA            |
    | us-west1                | Oregon, USA            |
    | us-west2                | Los Angeles, USA       |
    | us-west3                | Salt Lake City, USA    |
    | us-west4                | Las Vegas, USA         |
    | northamerica-northeast1 | Montreal, Canada       |
    | northamerica-northeast2 | Toronto, Canada        |
    | northamerica-south1     | Queretaro, Mexico      |
    | southamerica-east1      | Sao Paulo, Brazil      |
    | southamerica-west1      | Santiago, Chile        |

=== "Europe (13)"

    | Zone              | Location               |
    | ----------------- | ---------------------- |
    | europe-west1      | St. Ghislain, Belgium  |
    | europe-west2      | London, UK             |
    | europe-west3      | Frankfurt, Germany     |
    | europe-west4      | Eemshaven, Netherlands |
    | europe-west6      | Zurich, Switzerland    |
    | europe-west8      | Milan, Italy           |
    | europe-west9      | Paris, France          |
    | europe-west10     | Berlin, Germany        |
    | europe-west12     | Turin, Italy           |
    | europe-north1     | Hamina, Finland        |
    | europe-north2     | Stockholm, Sweden      |
    | europe-central2   | Warsaw, Poland         |
    | europe-southwest1 | Madrid, Spain          |

=== "Asia-Pacific (12)"

    | Zone                 | Location               |
    | -------------------- | ---------------------- |
    | asia-east1           | Changhua, Taiwan       |
    | asia-east2           | Kowloon, Hong Kong     |
    | asia-northeast1      | Tokyo, Japan           |
    | asia-northeast2      | Osaka, Japan           |
    | asia-northeast3      | Seoul, South Korea     |
    | asia-south1          | Mumbai, India          |
    | asia-south2          | Delhi, India           |
    | asia-southeast1      | Jurong West, Singapore |
    | asia-southeast2      | Jakarta, Indonesia     |
    | asia-southeast3      | Bangkok, Thailand      |
    | australia-southeast1 | Sydney, Australia      |
    | australia-southeast2 | Melbourne, Australia   |

=== "Middle East & Africa (3)"

    | Zone          | Location                   |
    | ------------- | -------------------------- |
    | africa-south1 | Johannesburg, South Africa |
    | me-central1   | Doha, Qatar                |
    | me-west1      | Tel Aviv, Israel           |

## Endpoint Configuration

### New Deployment Dialog

The `New Deployment` dialog provides:

| Setting             | Description                  | Default |
| ------------------- | ---------------------------- | ------- |
| **Model**           | Select from completed models | -       |
| **Region**          | Deployment region            | -       |
| **Deployment Name** | Auto-generated, editable     | -       |
| **CPU Cores**       | Fixed default                | 1       |
| **Memory (GB)**     | Fixed default                | 2       |

![Ultralytics Platform New Deployment Dialog Fixed Resource Defaults](https://cdn.ul.run/i/574300ec688c813a92304f252b57476b.avif)<!-- screenshot -->
The disabled **Resources** panel is marked **Coming Soon** and cannot currently be expanded or customized. Deployments use `1 CPU`, `2 GiB` memory, `minInstances = 0`, and `maxInstances = 1`.

!!! note "Auto-Generated Names"

    The deployment name is automatically generated from the model name and region city (e.g., `yolo26n-iowa`). If you deploy the same model to the same region again, a numeric suffix is added (e.g., `yolo26n-iowa-2`).

### Deploy Tab (Quick Deploy)

When deploying from the model's `Deploy` tab, endpoints are created with default resources (1 CPU, 2 GB memory) with scale-to-zero enabled. The deployment name is auto-generated.

## Manage Endpoints

### View Modes

The deployments list supports three view modes:

| Mode        | Description                                               |
| ----------- | --------------------------------------------------------- |
| **Cards**   | Full detail cards with logs, code examples, predict panel |
| **Compact** | Grid of smaller cards with key metrics                    |
| **Table**   | DataTable with sortable columns and search                |

![Ultralytics Platform Deploy Tab Active Deployments Cards View](https://cdn.ul.run/i/9e21cbf292ff0ff31f787bec8ce9f678.avif)<!-- screenshot -->

### Deployment Card (Cards View)

Each deployment card in the cards view shows:

- **Header**: Name, region flag, status badge, start/stop/delete buttons
- **Endpoint URL**: Copyable URL with link to API docs
- **Metrics**: Request count (24h), P95 latency, error rate
- **Health check**: Live health indicator with latency and manual refresh
- **Tabs**: `Logs`, `Code`, and `Predict`

The `Logs` tab shows recent log entries with severity filtering (All / Errors). The `Code` tab shows ready-to-use code examples in Python, JavaScript, and cURL with your actual endpoint URL and API key. The `Predict` tab provides an inline predict panel for testing directly on the deployment.

### Deployment Statuses

| Status        | Description                             |
| ------------- | --------------------------------------- |
| **Creating**  | Deployment is being set up              |
| **Deploying** | Container is starting                   |
| **Ready**     | Endpoint is live and accepting requests |
| **Stopping**  | Endpoint is shutting down               |
| **Stopped**   | Endpoint is paused and unavailable      |
| **Failed**    | Deployment failed (see error message)   |

### Endpoint URL

Each endpoint has a unique URL, for example:

```text
https://predict-abc123.run.app
```

![Ultralytics Platform Deployment Card Endpoint Url With Copy Button](https://cdn.ul.run/i/4f02beb3dd4915d65c72051e0235b1ea.avif)<!-- screenshot -->
Click the copy button to copy the URL. Click the docs icon to view the auto-generated API documentation for the endpoint.

## Lifecycle Management

Control your endpoint state:

```mermaid
graph LR
    R[Ready]:::out -->|Stop| S[Stopped]:::extern
    S -->|Start| R
    R -->|Delete| D[Deleted]:::error
    S -->|Delete| D

    classDef out fill:#9C27B0,color:#fff
    classDef error fill:#F44336,color:#fff
    classDef extern fill:#607D8B,color:#fff
```

| Action     | Description                 |
| ---------- | --------------------------- |
| **Start**  | Resume a stopped endpoint   |
| **Stop**   | Pause the endpoint          |
| **Delete** | Permanently remove endpoint |

### Stop Endpoint

Stop an endpoint when you do not want it to accept requests:

1. Click the pause icon on the deployment card
2. Endpoint status changes to "Stopping" then "Stopped"

Stopped endpoints:

- Don't accept requests
- Can be restarted anytime

### Delete Endpoint

Permanently remove an endpoint:

1. Click the delete (trash) icon on the deployment card
2. Confirm deletion in the dialog

!!! warning "Permanent Action"

    Deletion is immediate and permanent. You can always create a new endpoint.

## Using Endpoints

### Authentication

Each deployment is created with an API key from your account. Include it in requests:

```bash
Authorization: Bearer YOUR_API_KEY
```

The API key prefix is displayed on the deployment card footer for identification. Generate keys from [API Keys](../account/api-keys.md).

### Direct Endpoint Requests

Send production requests directly to the URL shown on the deployment card. These requests do not pass through the
Platform API rate limiter. The endpoint currently runs one instance with the fixed resources described above.

### Request Example

=== "Python"

    ```python
    import requests

    # Deployment endpoint
    url = "https://predict-abc123.run.app/predict"

    # Headers with your deployment API key
    headers = {"Authorization": "Bearer YOUR_API_KEY"}

    # Inference parameters
    data = {"conf": 0.25, "iou": 0.7, "imgsz": 640}

    # Send image for inference
    with open("image.jpg", "rb") as f:
        response = requests.post(url, headers=headers, data=data, files={"file": f})

    print(response.json())
    ```

=== "JavaScript"

    ```javascript
    // Build form data with image and parameters
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("conf", "0.25");
    formData.append("iou", "0.7");
    formData.append("imgsz", "640");

    // Send image for inference
    const response = await fetch(
      "https://predict-abc123.run.app/predict",
      {
        method: "POST",
        headers: { Authorization: "Bearer YOUR_API_KEY" },
        body: formData,
      }
    );

    const result = await response.json();
    console.log(result);
    ```

=== "cURL"

    ```bash
    curl -X POST \
      "https://predict-abc123.run.app/predict" \
      -H "Authorization: Bearer YOUR_API_KEY" \
      -F "file=@image.jpg" \
      -F "conf=0.25" \
      -F "iou=0.7" \
      -F "imgsz=640"
    ```

### Request Parameters

{% include "macros/platform-inference-parameters.md" %}

!!! tip "Video Inference"

    Dedicated endpoints accept both images and videos via the `file` parameter.

    - **Image formats** (up to 100 MB): AVIF, BMP, DNG, HEIC, JP2, JPEG, JPG, MPO, PNG, TIF, TIFF, WEBP
    - **Video formats** (up to 100 MB): ASF, AVI, GIF, M4V, MKV, MOV, MP4, MPEG, MPG, TS, WEBM, WMV

    Each video frame is processed individually and results are returned per frame. You can also pass a public image URL or a base64-encoded image via the `source` parameter instead of `file`.

### Response Format

Same as [shared inference](inference.md#response) with task-specific fields.

## FAQ

### How many endpoints can I create?

Endpoint limits depend on plan:

- **Free**: Up to 3 deployments
- **Pro**: Up to 10 deployments
- **Enterprise**: Unlimited deployments

Each model can still be deployed to multiple regions within your plan quota.

### Can I change the region after deployment?

No, regions are fixed. To change regions:

1. Delete the existing endpoint
2. Create a new endpoint in the desired region

### How do I handle multi-region deployment?

For global coverage:

1. Deploy to multiple regions
2. Use a load balancer or DNS routing
3. Route users to the nearest endpoint

### What's the cold start time?

Cold start time depends on the model and whether the endpoint has scaled to zero. Platform's health-check request allows
up to 55 seconds so an idle endpoint has time to start.

Each deployment currently uses the generated endpoint URL shown on its deployment card.
