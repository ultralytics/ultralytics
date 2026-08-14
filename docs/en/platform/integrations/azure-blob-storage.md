---
plans: [pro, enterprise]
comments: true
description: Connect Azure Blob Storage to Ultralytics Platform and train YOLO models on images in your Azure containers without uploading a copy.
keywords: Ultralytics Platform, Azure Blob Storage, Azure storage account, blob container, connection string, dataset import, YOLO, computer vision, cloud storage
title: Azure Blob Storage Datasets - Ultralytics Platform
---

# Azure Blob Storage Integration

The [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs) integration connects your storage account containers to [Ultralytics Platform](https://platform.ultralytics.com). Your images stay in your containers — Platform indexes them in place, so you can browse, annotate, and train YOLO models without uploading a copy.

!!! note "Pro feature"

    Azure Blob Storage datasets require a [Pro or Enterprise plan](../account/billing.md#plans). Free workspaces see the integration and are prompted to upgrade when connecting. Existing Azure Blob Storage datasets stay fully accessible if a subscription ends — only new connections and imports require Pro.

## Get a Connection String

Platform only ever reads from your storage — it never writes, modifies, or deletes your blobs. The current integration requires an account access-key connection string, which grants account-wide privileges even though Platform uses only list and read operations:

1. In the Azure portal, open your storage account.
2. Go to **Security + networking > Access keys**.
3. Copy a **connection string**.

The string must carry `AccountName`, `AccountKey`, and HTTPS as its protocol — the form Azure gives you under **Access
keys**. SAS-token connection strings, which contain `SharedAccessSignature` instead of `AccountKey`, are rejected.

!!! note "Public Azure cloud only"

    Connections use the standard `blob.core.windows.net` endpoint. Sovereign clouds (Azure China, Azure Government) and custom blob endpoints are not supported.

## Connect to Platform

1. Go to **Settings > [Integrations](index.md)** and select **Azure Blob Storage** from the integration list.
2. Paste the connection string.
3. Click **Find available containers**, then select the containers to connect. You can also enter a container name
   manually.
4. Click **Connect**. Platform verifies it can list and read each selected container before saving anything.

![Ultralytics Platform Azure Blob Storage Integration Settings](https://cdn.ul.run/i/85be53f25bc7085ce898b2a7b3839974.avif)<!-- screenshot -->

You need the workspace admin or owner [role](../account/teams.md#roles-and-permissions) to connect cloud storage. One
connection carries up to 50 containers, and discovery lists up to 300 containers in the storage account.

Reconnecting the same storage account later adds new containers to the existing integration. A saved credential is only replaced once its replacement can still read every container you've already connected.

!!! note "Credential security"

    An account key can authorize write and delete operations or create SAS tokens if it is exposed outside Platform. Credentials are encrypted at rest with AES-256-GCM, are never returned to the browser, and are never exposed to training workloads. Use a dedicated storage account where practical. To revoke access, rotate the storage account access keys in Azure.

## Create a Dataset from a Blob Container

1. Click **New Dataset** and open the **Cloud** tab.
2. Pick a connected container and browse to the folder containing your data.
3. Confirm the folder, adjust the dataset name, and create the dataset.

Platform lists the folder once and indexes what it finds:

- **Images** — `.jpg`, `.jpeg`, `.png`, `.webp`, and `.avif` blobs are indexed with dimensions read through bounded
  requests. Platform does not persist a second copy of the source image.
- **Labels** — YOLO `.txt` sidecars are parsed into Platform annotations, matched by the standard `images/` → `labels/` layout or as same-folder siblings.
- **Metadata** — a YAML file provides class names and pose keypoint shape, exactly like an [archive upload](../data/datasets.md#supported-formats). `data.yaml` and `data.yml` are preferred when the folder holds several.
- **Task** — a sample of the label files decides the task, so segment, pose, and OBB folders are recognized from their label shape rather than the task you picked in the dialog.
- **Splits** — `train`, `val`, and `test` folder names in the blob path assign splits automatically.

The dataset then behaves like any other: browse and [annotate](../data/annotation.md) it, set it public or private, share it with your [team](../account/teams.md), and [train](../train/index.md) on it through managed training. Originals are streamed on demand, and indexed images do not consume your Platform [storage quota](../account/billing.md).

!!! note "Limits"

    A single import indexes up to 50,000 blobs, and label or YAML files up to 1 MB each. Larger containers should be split across multiple datasets.

!!! warning "Keep indexed blobs immutable"

    Every indexed image is pinned to its blob ETag, and Platform fails closed if a blob changes underneath it. Add new blobs instead of overwriting existing ones.

## Failed Imports

If an import fails — an empty folder, a typo in the path, or revoked permissions — the dataset shows the error on its page. Editors can click **Retry import** to restart it with the stored container and folder, or create a new dataset pointing at the corrected path.

A retry re-lists the folder rather than resuming: blobs added since the first attempt are picked up, and blobs that are no longer there are dropped from the dataset.

## Training

Managed training works through the normal training flow. Training uses Platform's own copies of the pinned images for the duration of the run, and your Azure credentials are never exposed to training workloads.

## Disconnect a Connection

Disconnecting deletes the stored connection string without touching anything in Azure. Datasets built from those containers stay in your workspace with their classes, labels, and annotations, but their images cannot be loaded, previewed, or trained on until the same storage account is connected again.

Use the [REST API](../api/index.md) with the integration ID returned by `GET /api/integrations/buckets`:

```bash
curl -X DELETE \
  -H "Authorization: Bearer YOUR_API_KEY" \
  https://platform.ultralytics.com/api/integrations/buckets/INTEGRATION_ID
```

To revoke access at the source instead, rotate the storage account access keys in Azure.

## Current Limitations

Azure-backed datasets currently exclude features that require Platform-owned copies of your images: auto-annotation, [clustering analysis](../data/datasets.md#clustering), dataset cloning, and immutable [version snapshots](../data/datasets.md#versions-tab).

Deleting an Azure-backed dataset, or individual images from it, removes Platform's references only — your blobs are never touched.

Also see the [Google Cloud Storage](google-cloud-storage.md) and [Amazon S3](amazon-s3.md) integrations.
