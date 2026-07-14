---
plans: [pro, enterprise]
comments: true
description: Connect Google Cloud Storage to Ultralytics Platform and train YOLO models on images in your GCS buckets without uploading a copy.
keywords: Ultralytics Platform, Google Cloud Storage, GCS, GCS bucket, service account, dataset import, YOLO, computer vision, cloud storage
title: Google Cloud Storage Datasets - Ultralytics Platform
---

# Google Cloud Storage Integration

The [Google Cloud Storage](https://cloud.google.com/storage) integration connects your GCS buckets to [Ultralytics Platform](https://platform.ultralytics.com). Your images stay in your buckets — Platform indexes them in place, so you can browse, annotate, and train YOLO models without uploading a copy.

!!! note "Pro feature"

    Google Cloud Storage datasets require a [Pro or Enterprise plan](../account/billing.md#plans). Free workspaces see the integration and are prompted to upgrade when connecting. Existing Google Cloud Storage datasets stay fully accessible if a subscription ends — only new connections and imports require Pro.

## Create a Read-Only Service Account

Platform only ever reads from your storage — it never writes, modifies, or deletes your objects. Create a dedicated [service account](https://cloud.google.com/iam/docs/service-account-overview) with read access only:

1. In the Google Cloud console, go to **IAM & Admin > Service Accounts** and create a service account.
2. Grant it the **Storage Object Viewer** (`roles/storage.objectViewer`) role on the buckets you want to connect.
3. Open the service account, choose **Keys > Add key > Create new key**, select **JSON**, and download the key file.

## Connect to Platform

1. Go to **Settings > [Integrations](index.md)** and find the **Google Cloud** card.
2. Click **Connect** and paste the contents of the service account JSON key.
3. Platform lists the buckets the service account can read. Select the ones to connect, or enter a bucket name manually if the account can't list buckets.
4. Click **Connect**. Platform verifies it can list and read each selected bucket before saving anything.

Reconnecting the same service account later adds new buckets to the existing integration. A saved credential is only replaced once its replacement can still read every bucket you've already connected.

!!! note "Credential security"

    Credentials are encrypted at rest with AES-256-GCM, are never returned to the browser, and never enter training job payloads. To revoke access, delete the service account key in Google Cloud.

## Create a Dataset from a GCS Bucket

1. Click **New Dataset** and open the **Cloud storage** tab.
2. Pick a connected bucket and browse to the folder containing your data.
3. Confirm the folder, adjust the dataset name, and create the dataset.

Platform lists the folder once and indexes what it finds:

- **Images** — `.jpg`, `.jpeg`, `.png`, `.webp`, and `.avif` objects are indexed with dimensions read from bounded header requests. Source pixels are never copied out of your bucket.
- **Labels** — YOLO `.txt` sidecars are parsed into Platform annotations, matched by the standard `images/` → `labels/` layout or as same-folder siblings.
- **Metadata** — a `data.yaml`/`data.yml` provides class names, task type, and pose keypoint shape, exactly like an [archive upload](../data/datasets.md#supported-formats).
- **Splits** — `train`, `val`, and `test` folder names in the object path assign splits automatically.

The dataset then behaves like any other: browse and [annotate](../data/annotation.md) it, set it public or private, share it with your [team](../account/teams.md), and [train](../train/index.md) on it through managed training. Originals are streamed on demand, and indexed images do not consume your Platform [storage quota](../account/billing.md).

!!! note "Limits"

    A single import indexes up to 50,000 objects, and label or YAML files up to 1 MB each. Larger buckets should be split across multiple datasets.

!!! warning "Keep indexed objects immutable"

    Every indexed image is pinned to its GCS object generation, and Platform fails closed if an object changes underneath it. Add new objects instead of overwriting existing ones.

## Failed Imports

If an import fails — an empty folder, a typo in the path, or revoked permissions — the dataset shows the error on its page. Editors can click **Retry import** to restart it with the stored bucket and folder, or create a new dataset pointing at the corrected path.

## Training

Managed training works through the normal training flow. Workers download the pinned originals into temporary job storage for the run and remove them with job cleanup — your Google Cloud credentials never reach compute.

## Current Limitations

GCS-backed datasets currently exclude features that require Platform-owned copies of your images: auto-annotation, [clustering analysis](../data/datasets.md#clustering), dataset cloning, and immutable [version snapshots](../data/datasets.md#versions-tab).

Deleting a GCS-backed dataset, or individual images from it, removes Platform's references only — your objects are never touched.

Also see the [Amazon S3](amazon-s3.md) and [Azure Blob Storage](azure-blob-storage.md) integrations.
