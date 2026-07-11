---
comments: true
description: Connect Google Cloud Storage, Amazon S3, or Azure Blob Storage and train YOLO models on your data without uploading a copy to Ultralytics Platform.
keywords: Ultralytics Platform, cloud storage, Google Cloud Storage, GCS, Amazon S3, Azure Blob Storage, dataset import, YOLO, computer vision, integrations
title: Cloud Storage Datasets - Ultralytics Platform
---

# Cloud Storage Integration

The cloud storage integration connects [Google Cloud Storage](https://cloud.google.com/storage), [Amazon S3](https://aws.amazon.com/s3/), and [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs) to [Ultralytics Platform](https://platform.ultralytics.com). Your images stay in your own buckets and containers — Platform indexes them in place, so you can browse, annotate, and train without uploading a copy.

All three providers share one integration model: connect a credential once, then browse every connected location through the same interface and create datasets from any folder.

## Connect a Provider

1. Go to **Settings > [Integrations](index.md)** and find the **Google Cloud**, **Amazon S3**, or **Microsoft Azure** card.
2. Click **Connect** and paste the provider credential:
    - **Google Cloud Storage** — a service account JSON key with read access to your buckets.
    - **Amazon S3** — an access key ID, secret access key, and bucket region for a dedicated IAM user with only list and read access. Never use root credentials.
    - **Azure Blob Storage** — a storage account connection string from **Security + networking > Access keys**.
3. Platform lists the buckets or containers the credential can read. Select the ones you want to connect, or enter a name manually if the credential doesn't permit discovery.
4. Click **Connect**. Platform verifies it can list and read each selected location before saving anything.

Reconnecting the same credential later adds new locations to the existing integration. A saved credential is only replaced once its replacement can still read every location you've already connected.

!!! tip "Use least-privilege, read-only credentials"

    Platform only ever reads from your storage — it never writes, modifies, or deletes your objects. Grant the credential list and read permissions only (for example `Storage Object Viewer` on GCS, `s3:ListBucket` + `s3:GetObject` on AWS).

!!! note "Credential security"

    Credentials are encrypted at rest with AES-256-GCM, are never returned to the browser, and never enter training job payloads. To revoke access, rotate or delete the key with your cloud provider.

## Create a Dataset from Cloud Storage

1. Click **New Dataset** and open the **Cloud storage** tab.
2. Pick a connected location and browse to the folder containing your data.
3. Confirm the folder, adjust the dataset name, and create the dataset.

Platform lists the folder once and indexes what it finds:

- **Images** — `.jpg`, `.jpeg`, `.png`, `.webp`, and `.avif` files are indexed with their dimensions read from bounded header requests. Source pixels are never copied.
- **Labels** — YOLO `.txt` sidecars are parsed into Platform annotations, matched by the standard `images/` → `labels/` layout or as same-folder siblings.
- **Metadata** — a `data.yaml`/`data.yml` provides class names, task type, and pose keypoint shape, exactly like an [archive upload](../data/datasets.md#supported-formats).
- **Splits** — `train`, `val`, and `test` folder names in the object path assign splits automatically.

The dataset then behaves like any other: browse and [annotate](../data/annotation.md) it, set it public or private, share it with your [team](../account/teams.md), and [train](../train/index.md) on it through managed training. Original images are streamed on demand through Platform; indexed images do not consume your Platform [storage quota](../account/billing.md).

!!! note "Limits"

    A single import indexes up to 50,000 objects, and label or YAML files up to 1 MB each. Larger folders should be split across multiple datasets.

!!! warning "Keep indexed objects immutable"

    Every indexed image is pinned to the exact object revision that was read, and Platform fails closed if an object changes underneath it. Add new files instead of overwriting existing ones.

## Failed Imports

If an import fails — an empty folder, a typo in the path, or revoked permissions — the dataset shows the error on its page. Editors can click **Retry import** to restart it with the stored location, or create a new dataset pointing at the corrected folder.

## Training

Managed training works through the normal training flow. Workers download the pinned originals into temporary job storage for the run and remove them with job cleanup — your provider credentials never reach compute.

## Current Limitations

Cloud-backed datasets currently exclude features that require Platform-owned copies of your images: auto-annotation, [clustering analysis](../data/datasets.md#clustering), dataset cloning, and immutable [version snapshots](../data/datasets.md#versions-tab). These are planned to arrive as the shared boundaries learn to resolve external originals.

Deleting a cloud-backed dataset, or individual images from it, removes Platform's references only — your objects are never touched.
