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

!!! note "Bucket discovery needs one more permission"

    `roles/storage.objectViewer` covers everything Platform does with your data, but it does not allow listing the
    buckets in a project. Add a role carrying `storage.buckets.list` if you want Platform to find your buckets for you;
    otherwise type each bucket name manually when connecting.

## Connect to Platform

1. Go to **Settings > [Integrations](index.md)** and select **Google Cloud Storage** from the integration list.
2. Paste the contents of the service account JSON key.
3. Click **Find available buckets**, then select the buckets to connect. If the service account can't list buckets,
   enter a known bucket name manually.
4. Click **Connect**. Platform verifies it can list and read each selected bucket before saving anything.

![Ultralytics Platform Google Cloud Storage Integration Settings](https://cdn.ul.run/i/20245971f25b11765202ff542a4faa68.avif)<!-- screenshot -->

You need the workspace admin or owner [role](../account/teams.md#roles-and-permissions) to connect cloud storage. One
connection carries up to 50 buckets, and discovery lists up to 300 of the buckets the service account can see.

Your browser reads the pasted key file and sends only the three fields Platform needs — `client_email`, `private_key`,
and `project_id`. The rest of the JSON never leaves the page.

Reconnecting the same service account later adds new buckets to the existing integration. A saved credential is only replaced once its replacement can still read every bucket you've already connected.

!!! note "Credential security"

    Credentials are encrypted at rest with AES-256-GCM, are never returned to the browser, and are never exposed to training workloads. To revoke access, delete the service account key in Google Cloud.

## Create a Dataset from a GCS Bucket

1. Click **New Dataset** and open the **Cloud** tab.
2. Pick a connected bucket and browse to the folder containing your data.
3. Confirm the folder, adjust the dataset name, and create the dataset.

Platform lists the folder once and indexes what it finds:

- **Images** — `.jpg`, `.jpeg`, `.png`, `.webp`, and `.avif` objects are indexed with dimensions read through bounded
  requests. Platform does not persist a second copy of the source image.
- **Labels** — YOLO `.txt` sidecars are parsed into Platform annotations, matched by the standard `images/` → `labels/` layout or as same-folder siblings.
- **Metadata** — a YAML file provides class names and pose keypoint shape, exactly like an [archive upload](../data/datasets.md#supported-formats). `data.yaml` and `data.yml` are preferred when the folder holds several.
- **Task** — a sample of the label files decides the task, so segment, pose, and OBB folders are recognized from their label shape rather than the task you picked in the dialog.
- **Splits** — `train`, `val`, and `test` folder names in the object path assign splits automatically.

The dataset then behaves like any other: browse and [annotate](../data/annotation.md) it, set it public or private, share it with your [team](../account/teams.md), and [train](../train/index.md) on it through managed training. Originals are streamed on demand, and indexed images do not consume your Platform [storage quota](../account/billing.md).

!!! note "Limits"

    A single import indexes up to 50,000 objects, and label or YAML files up to 1 MB each. Larger buckets should be split across multiple datasets.

!!! warning "Keep indexed objects immutable"

    Every indexed image is pinned to its GCS object generation, and Platform fails closed if an object changes underneath it. Add new objects instead of overwriting existing ones.

## Failed Imports

If an import fails — an empty folder, a typo in the path, or revoked permissions — the dataset shows the error on its page. Editors can click **Retry import** to restart it with the stored bucket and folder, or create a new dataset pointing at the corrected path.

A retry re-lists the folder rather than resuming: objects added since the first attempt are picked up, and objects that are no longer there are dropped from the dataset.

## Training

Managed training works through the normal training flow. Training uses Platform's own copies of the pinned images for the duration of the run, and your Google Cloud credentials are never exposed to training workloads.

## Disconnect a Connection

Disconnecting deletes the stored credentials without touching anything in Google Cloud. Datasets built from those buckets stay in your workspace with their classes, labels, and annotations, but their images cannot be loaded, previewed, or trained on until the same service account is connected again.

Use the [REST API](../api/index.md) with the integration ID returned by `GET /api/integrations/buckets`:

```bash
curl -X DELETE \
  -H "Authorization: Bearer YOUR_API_KEY" \
  https://platform.ultralytics.com/api/integrations/buckets/INTEGRATION_ID
```

To revoke access at the source instead, delete the service account key in Google Cloud.

## Current Limitations

GCS-backed datasets currently exclude features that require Platform-owned copies of your images: auto-annotation, [clustering analysis](../data/datasets.md#clustering), dataset cloning, and immutable [version snapshots](../data/datasets.md#versions-tab).

Deleting a GCS-backed dataset, or individual images from it, removes Platform's references only — your objects are never touched.

Also see the [Amazon S3](amazon-s3.md) and [Azure Blob Storage](azure-blob-storage.md) integrations.
