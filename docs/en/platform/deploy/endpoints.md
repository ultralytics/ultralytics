---
comments: true
description: Deploy YOLO models to dedicated endpoints in 43 global regions with auto-scaling and monitoring on Ultralytics Platform.
keywords: Ultralytics Platform, deployment, endpoints, YOLO, production, scaling, global regions
---

# Dedicated Endpoints

[Ultralytics Platform](https://platform.ultralytics.com) enables deployment of YOLO models to dedicated endpoints in 43 global regions. Each endpoint is a single-tenant service with auto-scaling, custom URLs, and independent monitoring.

<!-- Screenshot: platform-deploy-tab.avif -->

## Create Endpoint

Deploy a model to a dedicated endpoint:

1. Navigate to your model
2. Click the **Deploy** tab
3. Select a region from the map
4. Click **Deploy**

### Region Selection

Choose from 43 regions worldwide:

<!-- Screenshot: platform-deploy-map.avif -->

The interactive map shows:

- **Region pins**: Click to select
- **Latency indicators**: Color-coded by distance
    - Green: <100ms
    - Yellow: 100-200ms
    - Red: >200ms

### Region Table

View all regions with details:

<!-- Screenshot: platform-deploy-regions.avif -->

| Column       | Description        |
| ------------ | ------------------ |
| **Region**   | Region identifier  |
| **Location** | City/country       |
| **Latency**  | Measured ping time |
| **Status**   | Available/deployed |

!!! tip "Choose Wisely"

    Select the region closest to your users for lowest latency. Consider deploying to multiple regions for global coverage.

## Available Regions

### Americas (14 regions)

| Zone                    | Location            |
| ----------------------- | ------------------- |
| us-central1             | Iowa, USA           |
| us-east1                | South Carolina, USA |
| us-east4                | Virginia, USA       |
| us-east5                | Ohio, USA           |
| us-west1                | Oregon, USA         |
| us-west2                | Los Angeles, USA    |
| us-west3                | Salt Lake City, USA |
| us-west4                | Las Vegas, USA      |
| us-south1               | Dallas, USA         |
| northamerica-northeast1 | Montreal, Canada    |
| northamerica-northeast2 | Toronto, Canada     |
| southamerica-east1      | São Paulo, Brazil   |
| southamerica-west1      | Santiago, Chile     |

### Europe (12 regions)

| Zone              | Location            |
| ----------------- | ------------------- |
| europe-west1      | Belgium             |
| europe-west2      | London, UK          |
| europe-west3      | Frankfurt, Germany  |
| europe-west4      | Netherlands         |
| europe-west6      | Zurich, Switzerland |
| europe-west8      | Milan, Italy        |
| europe-west9      | Paris, France       |
| europe-west10     | Berlin, Germany     |
| europe-west12     | Turin, Italy        |
| europe-north1     | Finland             |
| europe-central2   | Warsaw, Poland      |
| europe-southwest1 | Madrid, Spain       |

### Asia-Pacific (14 regions)

| Zone                 | Location             |
| -------------------- | -------------------- |
| asia-east1           | Taiwan               |
| asia-east2           | Hong Kong            |
| asia-northeast1      | Tokyo, Japan         |
| asia-northeast2      | Osaka, Japan         |
| asia-northeast3      | Seoul, South Korea   |
| asia-south1          | Mumbai, India        |
| asia-south2          | Delhi, India         |
| asia-southeast1      | Singapore            |
| asia-southeast2      | Jakarta, Indonesia   |
| australia-southeast1 | Sydney, Australia    |
| australia-southeast2 | Melbourne, Australia |

### Middle East & Africa (3 regions)

| Zone        | Location             |
| ----------- | -------------------- |
| me-central1 | Doha, Qatar          |
| me-central2 | Dammam, Saudi Arabia |
| me-west1    | Tel Aviv, Israel     |

## Endpoint Configuration

When creating an endpoint:

<!-- Screenshot: platform-deploy-create.avif -->

| Setting           | Description               | Default |
| ----------------- | ------------------------- | ------- |
| **Region**        | Deployment region         | -       |
| **Min Instances** | Minimum running instances | 0       |
| **Max Instances** | Maximum scaling limit     | 10      |

### Scaling Options

| Setting     | Behavior                                 |
| ----------- | ---------------------------------------- |
| **Min = 0** | Scale to zero when idle (cost-effective) |
| **Min > 0** | Always-on for no cold starts             |
| **Max**     | Upper limit for traffic spikes           |

!!! warning "Cold Starts"

    With min instances = 0, the first request after idle triggers a cold start (2-5 seconds). Set min > 0 for latency-sensitive applications.

## Manage Endpoints

View and manage your endpoints:

<!-- Screenshot: platform-deploy-list.avif -->

### Endpoint Details

| Field         | Description                 |
| ------------- | --------------------------- |
| **URL**       | HTTPS endpoint for requests |
| **Region**    | Deployed region             |
| **Status**    | Running, Stopped, Deploying |
| **Instances** | Current/max instance count  |

### Endpoint URL

Each endpoint has a unique URL:

```
https://model-abc123-us-central1.a.run.app
```

<!-- Screenshot: platform-deploy-endpoint.avif -->

Click the copy button to copy the URL.

## Lifecycle Management

Control your endpoint state:

<!-- Screenshot: platform-deploy-lifecycle.avif -->

| Action     | Description                     |
| ---------- | ------------------------------- |
| **Start**  | Resume a stopped endpoint       |
| **Stop**   | Pause the endpoint (no billing) |
| **Delete** | Permanently remove endpoint     |

### Stop Endpoint

Stop an endpoint to pause billing:

1. Open endpoint actions menu
2. Click **Stop**
3. Confirm action

Stopped endpoints:

- Don't accept requests
- Don't incur charges
- Can be restarted anytime

### Delete Endpoint

Permanently remove an endpoint:

1. Open endpoint actions menu
2. Click **Delete**
3. Confirm deletion

!!! warning "Permanent Action"

    Deletion is immediate and permanent. You can always create a new endpoint.

## Using Endpoints

### Authentication

Include your API key in requests:

```bash
Authorization: Bearer YOUR_API_KEY
```

### Request Example

=== "cURL"

    ```bash
    curl -X POST \
      "https://model-abc123-us-central1.a.run.app/predict" \
      -H "Authorization: Bearer YOUR_API_KEY" \
      -F "file=@image.jpg"
    ```

=== "Python"

    ```python
    import requests

    url = "https://model-abc123-us-central1.a.run.app/predict"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    files = {"file": open("image.jpg", "rb")}

    response = requests.post(url, headers=headers, files=files)
    print(response.json())
    ```

### Response Format

Same as [shared inference](inference.md#response) with task-specific fields.

## Pricing

Dedicated endpoints bill based on:

| Component    | Rate                 |
| ------------ | -------------------- |
| **CPU**      | Per vCPU-second      |
| **Memory**   | Per GB-second        |
| **Requests** | Per million requests |

!!! tip "Cost Optimization"

    - Use scale-to-zero for development endpoints
    - Set appropriate max instances
    - Monitor usage in the [Monitoring](monitoring.md) dashboard

## FAQ

### How many endpoints can I create?

There's no hard limit. Each model can have endpoints in multiple regions. Total endpoints depend on your plan.

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

Cold start varies by model size:

| Model   | Cold Start |
| ------- | ---------- |
| YOLO26n | ~2 seconds |
| YOLO26m | ~3 seconds |
| YOLO26x | ~5 seconds |

Set min instances > 0 to eliminate cold starts.

### Can I use custom domains?

Custom domains are coming soon. Currently, endpoints use platform-generated URLs.
