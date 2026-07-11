---
comments: true
description: Connect Amazon S3 to Ultralytics Platform and train YOLO models on images in your S3 buckets without uploading a copy.
keywords: Ultralytics Platform, Amazon S3, AWS S3, S3 bucket, IAM access key, dataset import, YOLO, computer vision, cloud storage
title: Amazon S3 Datasets - Ultralytics Platform
---

# Amazon S3 Integration

The [Amazon S3](https://aws.amazon.com/s3/) integration connects your S3 buckets to [Ultralytics Platform](https://platform.ultralytics.com). Your images stay in your buckets — Platform indexes them in place, so you can browse, annotate, and train YOLO models without uploading a copy.

## Create a Read-Only IAM User

Platform only ever reads from your storage — it never writes, modifies, or deletes your objects. Use a dedicated [IAM user](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users.html) with list and read access only — never root credentials:

1. In the AWS console, go to **IAM > Users** and create a user with no console access.
2. Attach a policy granting only list and read access to the buckets you want to connect:

    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            { "Effect": "Allow", "Action": "s3:ListAllMyBuckets", "Resource": "*" },
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetObject"],
                "Resource": ["arn:aws:s3:::my-bucket", "arn:aws:s3:::my-bucket/*"]
            }
        ]
    }
    ```

    `s3:ListAllMyBuckets` is optional — it lets Platform discover your buckets so you don't have to type their names.

3. Open the user's **Security credentials** tab, create an **access key**, and copy the access key ID and secret access key.

## Connect to Platform

1. Go to **Settings > [Integrations](index.md)** and find the **Amazon S3** card.
2. Click **Connect** and enter the access key ID, secret access key, and the bucket region (for example `us-east-1`).
3. Platform lists the buckets the credential can read. Select the ones to connect, or enter a bucket name manually if the policy doesn't permit discovery.
4. Click **Connect**. Platform verifies it can list and read each selected bucket before saving anything.

Reconnecting the same IAM user later adds new buckets to the existing integration. A saved credential is only replaced once its replacement can still read every bucket you've already connected.

!!! note "One region per connection"

    A connection reads buckets in the region you enter. If your buckets live in several regions, connect once per region.

!!! note "Credential security"

    Credentials are encrypted at rest with AES-256-GCM, are never returned to the browser, and never enter training job payloads. To revoke access, deactivate the access key in AWS IAM.

## Create a Dataset from an S3 Bucket

1. Click **New Dataset** and open the **Cloud storage** tab.
2. Pick a connected bucket and browse to the folder containing your data.
3. Confirm the folder, adjust the dataset name, and create the dataset.

Platform lists the folder once and indexes what it finds:

- **Images** — `.jpg`, `.jpeg`, `.png`, `.webp`, and `.avif` objects are indexed with dimensions read from bounded header requests. Source pixels are never copied out of your bucket.
- **Labels** — YOLO `.txt` sidecars are parsed into Platform annotations, matched by the standard `images/` → `labels/` layout or as same-folder siblings.
- **Metadata** — a `data.yaml`/`data.yml` provides class names, task type, and pose keypoint shape, exactly like an [archive upload](../data/datasets.md#supported-formats).
- **Splits** — `train`, `val`, and `test` folder names in the object key assign splits automatically.

The dataset then behaves like any other: browse and [annotate](../data/annotation.md) it, set it public or private, share it with your [team](../account/teams.md), and [train](../train/index.md) on it through managed training. Originals are streamed on demand, and indexed images do not consume your Platform [storage quota](../account/billing.md).

!!! note "Limits"

    A single import indexes up to 50,000 objects, and label or YAML files up to 1 MB each. Larger buckets should be split across multiple datasets.

!!! warning "Keep indexed objects immutable"

    Every indexed image is pinned to its S3 object ETag, and Platform fails closed if an object changes underneath it. Add new objects instead of overwriting existing ones.

## Failed Imports

If an import fails — an empty folder, a typo in the path, or revoked permissions — the dataset shows the error on its page. Editors can click **Retry import** to restart it with the stored bucket and folder, or create a new dataset pointing at the corrected path.

## Training

Managed training works through the normal training flow. Workers download the pinned originals into temporary job storage for the run and remove them with job cleanup — your AWS credentials never reach compute.

## Current Limitations

S3-backed datasets currently exclude features that require Platform-owned copies of your images: auto-annotation, [clustering analysis](../data/datasets.md#clustering), dataset cloning, and immutable [version snapshots](../data/datasets.md#versions-tab).

Deleting an S3-backed dataset, or individual images from it, removes Platform's references only — your objects are never touched.

Also see the [Google Cloud Storage](google-cloud-storage.md) and [Azure Blob Storage](azure-blob-storage.md) integrations.
