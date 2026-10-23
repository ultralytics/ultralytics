---
plans: [pro, enterprise]
comments: true
description: Connect Google Cloud Storage to Ultralytics Platform and train YOLO models on images in your GCS buckets without uploading a copy.
keywords: Ultralytics Platform, Google Cloud Storage, GCS, GCS bucket, service account, dataset import, YOLO, computer vision, cloud storage
title: Google Cloud Storage Datasets - Ultralytics Platform
---

# Google Cloud Storage Integration

The [Google Cloud Storage](https://cloud.google.com/storage) integration connects your GCS buckets to [Ultralytics Platform](https://platform.ultralytics.com). Your images stay in your buckets — Platform indexes them in place, so you can browse, annotate, and train YOLO models without uploading a copy. With [AI features](#ai-features) on, Platform can also use them for smart annotation, auto-annotation, similar-image search, and cloud training.

!!! note "Pro feature"

    Google Cloud Storage datasets require a [Pro or Enterprise plan](../account/billing.md#plans). Free workspaces see the integration and are prompted to upgrade when connecting. Existing Google Cloud Storage datasets stay fully accessible if a subscription ends — only new connections and imports require Pro.

## Create a Service Account

Read access covers everything except [image generation](#image-generation), which also creates objects. Platform never modifies or deletes your objects. Create a dedicated [service account](https://docs.cloud.google.com/iam/docs/service-account-overview):

1. In the Google Cloud console, go to **IAM & Admin > Service Accounts** and create a service account.
2. Grant it the **Storage Object Viewer** (`roles/storage.objectViewer`) role on the buckets you want to connect. For
   [image generation](#image-generation), also grant **Storage Object Creator** (`roles/storage.objectCreator`), which
   creates objects but cannot overwrite or delete them.
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
4. Click **Connect**. Platform verifies it can list and read each selected bucket before saving anything, and asks
   Google Cloud whether the service account can also create objects. Nothing is written.

![Ultralytics Platform Google Cloud Storage Integration Settings](https://cdn.ul.run/i/e6224d9cf7306aec08d6cc53b3c7d70c.avif)<!-- screenshot -->

Each connected bucket then shows **Read and write** or **Read only**, and each credential has an [AI features](#ai-features) switch. After you change its roles, connect the same service account again to recheck write access.

You need the workspace admin or owner [role](../account/teams.md#roles-and-permissions) to connect cloud storage. You
can add up to 50 buckets at a time, and discovery lists up to 300 of the buckets the service account can see.

Your browser reads the pasted key file and sends only the three fields Platform needs — `client_email`, `private_key`,
and `project_id`. The rest of the JSON never leaves the page.

Reconnecting the same service account later adds new buckets to the existing integration. A saved credential is only replaced once its replacement can still read every bucket you've already connected. If any bucket can't be read, the error names it, including buckets connected earlier with the same service account.

!!! note "Credential security"

    Credentials are encrypted at rest with AES-256-GCM, are never returned to the browser, and are never exposed to training workloads. To revoke access, delete the service account key in Google Cloud.

## AI Features

Each connected credential has an **AI features** switch in **Settings > Integrations**, on by default. While it's on, Ultralytics GPUs and AI model providers may use that credential's images for:

- [SAM smart annotation](../data/annotation.md#sam-smart-annotation), and single-image and batch auto-annotation
- [Similar-image search](../data/datasets.md#find-similar-images) and [clustering analysis](../data/datasets.md#clustering)
- [Cloud training](../train/cloud-training.md)
- [Image generation](#image-generation), on buckets with write access

AI model providers are involved only for image generation, whose prompts Google Gemini plans from the source image, and for paid annotation models, which run on your workspace's own provider key. Platform keeps no copy of your images: features read them on demand, cloud training uses temporary copies for the duration of a run, and similar-image search keeps an embedding of each image, which is deleted when the image leaves Platform.

Turn the switch off to keep images off Ultralytics GPUs and AI model providers. Turning it off stops new AI work; work already running finishes. Datasets on that credential then show **AI features off** with a link back to these settings, and each unavailable feature explains why. Manual annotation, sharing, and local training keep working, and the labels, models, and search results you already created are kept.

You need the workspace admin or owner [role](../account/teams.md#roles-and-permissions) to change the switch. You can also set it with the [REST API](../api/index.md#set-ai-features), using the integration ID returned by `GET /api/integrations/buckets`:

```bash
curl -X PATCH \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"processing": false}' \
  https://platform.ultralytics.com/api/integrations/buckets/INTEGRATION_ID
```

## Image Generation

[Generate similar images](../data/datasets.md#generate-similar-images) works on Google Cloud Storage datasets whose bucket shows **Read and write** while AI features are on. Images you add are written to a `generated/` folder inside the dataset's folder, then added to the dataset like any other indexed image. Platform only creates new objects there and never overwrites or deletes existing ones. On read-only buckets, the action explains that it needs write access.

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

    A single import indexes up to 500,000 objects, and label or YAML files up to 1 MB each. Larger buckets should be split across multiple datasets.

!!! warning "Keep indexed objects immutable"

    Every indexed image is pinned to its GCS object generation, and Platform fails closed if an object changes underneath it. Add new objects instead of overwriting existing ones.

## Failed Imports

If an import fails — an empty folder, a typo in the path, or revoked permissions — the dataset shows the error on its page. Editors can click **Retry import** to restart it with the stored bucket and folder, or create a new dataset pointing at the corrected path.

A retry re-lists the folder rather than resuming: objects added since the first attempt are picked up, and objects that are no longer there are dropped from the dataset.

## Training

Managed training works through the normal training flow while [AI features](#ai-features) are on. Training uses Platform's own copies of the pinned images for the duration of the run, and your Google Cloud credentials are never exposed to training workloads. With AI features off, the training dialog explains why cloud training is unavailable and offers local training instead.

## Disconnect a Connection

Disconnecting deletes the stored credentials without touching anything in Google Cloud. Datasets built from those buckets stay in your workspace with their classes, labels, and annotations, but their images cannot be loaded, previewed, or trained on until the same service account is connected again. You can still rename them, edit their labels, or delete them while disconnected.

To disconnect, go to **Settings > Integrations**, select **Google Cloud Storage**, click **Disconnect** next to the credential, and confirm. Opening one of its datasets then shows **Google Cloud Storage Disconnected** with a link back to these settings, and the dataset returns automatically once you reconnect.

You can also disconnect with the [REST API](../api/index.md), using the integration ID returned by `GET /api/integrations/buckets`:

```bash
curl -X DELETE \
  -H "Authorization: Bearer YOUR_API_KEY" \
  https://platform.ultralytics.com/api/integrations/buckets/INTEGRATION_ID
```

```python
from ultralytics_platform import Platform

client = Platform()  # reads ULTRALYTICS_API_KEY
integrations = client.storage_integrations.list()
client.storage_integrations.delete("INTEGRATION_ID")
```

To revoke access at the source instead, delete the service account key in Google Cloud.

## Current Limitations

GCS-backed datasets exclude features that need Platform-owned copies of your images: dataset cloning, immutable [version snapshots](../data/datasets.md#versions-tab), [face blurring](../data/datasets.md#blur-faces), and pasting images from other datasets. Smart annotation, auto-annotation, similar-image search, clustering, cloud training, and image generation need [AI features](#ai-features) on.

Deleting a GCS-backed dataset, or individual images from it, removes Platform's references only — your objects are never touched.

Also see the [Amazon S3](amazon-s3.md) and [Azure Blob Storage](azure-blob-storage.md) integrations.

## FAQ

### Does Platform copy my images?

No. Platform lists the bucket once and indexes the image dimensions and labels it finds. Originals are streamed on demand for browsing and annotation. With [AI features](#ai-features) on, Ultralytics GPUs read them on demand, managed training uses temporary copies for the duration of a run, and similar-image search keeps an embedding of each image, which is deleted with the image. The only objects Platform ever writes are the generated images you add, under `generated/`. Indexed images do not count against your [storage quota](../account/billing.md).

### What happens if I add or change files in the bucket?

A dataset is indexed once when it is created. New objects are not picked up until you create a new dataset or retry a failed import, and every indexed image is pinned to its version, so overwriting an object makes Platform fail closed on that image. Add new objects instead of replacing existing ones.

### Why is smart annotation or clustering unavailable on my GCS dataset?

Check that [AI features](#ai-features) are on for its credential in **Settings > Integrations**, and that the credential is still connected; the dataset header shows **AI features off** when they're off. Dataset cloning, version snapshots, face blurring, and pasting images stay unavailable on connected datasets.

### Can Platform write to my bucket?

Only for image generation, and only when the service account can create objects and AI features are on. Platform creates new objects under `generated/` in the dataset's folder and never overwrites or deletes existing ones.

### How do I revoke Platform's access?

Disconnect the integration in **Settings > Integrations** to delete the stored credentials, or delete the service account key in Google Cloud. Neither action touches the data in your buckets. To keep images off Ultralytics GPUs without disconnecting, turn off [AI features](#ai-features).
