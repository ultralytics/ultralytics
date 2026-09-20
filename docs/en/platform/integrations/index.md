---
plans: [free, pro, enterprise]
title: Platform Integrations
comments: true
description: Connect Ultralytics Platform to Slack, existing tools, cloud storage, and Enterprise On Premise compute and datasets.
keywords: Ultralytics Platform, integrations, Slack, alerts, data import, Roboflow, Labelbox, LabelMe, CVAT, Label Studio, cloud storage, GCS, Amazon S3, Azure Blob Storage, On Premise, dataset migration, YOLO, computer vision
---

# Integrations

[Ultralytics Platform](https://platform.ultralytics.com) [integrations](../../integrations/index.md) connect your workspace to other tools and services you already use. Send job results to Slack, import existing datasets with a single API key, or connect your cloud storage and use the data where it lives.

![Ultralytics Platform Settings Integrations Tab](https://cdn.ul.run/i/ff5b55316ea85e8eadcffa272698239c.avif)<!-- screenshot -->

## Accessing Integrations

All integrations are managed from your account settings:

1. Go to **Settings > Integrations**
2. Pick a service from the list on the left — grouped into **Infrastructure**, **Notifications**, and **Imports** — or type its name in the search box
3. Follow the connection prompts

Roboflow imports start with a preview so you can review what will be transferred and confirm that you have enough
[storage](../account/billing.md). Labelbox, LabelMe, CVAT, and Label Studio need no connection at all — upload the
exported dataset and Platform reads it directly. Cloud storage connections verify list and read access on every
selected bucket before anything is saved.

## Available Integrations

| Integration                                         | Category       | What it does                                                                 |
| --------------------------------------------------- | -------------- | ---------------------------------------------------------------------------- |
| [**On Premise**](on-premise.md)                     | Infrastructure | Indexes and trains on datasets that never leave your own computer            |
| [**Amazon S3**](amazon-s3.md)                       | Infrastructure | Indexes datasets in place from your S3 buckets                               |
| [**Google Cloud Storage**](google-cloud-storage.md) | Infrastructure | Indexes datasets in place from your GCS buckets                              |
| [**Azure Blob Storage**](azure-blob-storage.md)     | Infrastructure | Indexes datasets in place from your blob containers                          |
| [**Slack**](slack.md)                               | Notifications  | Posts selected training, export, and deployment results to one Slack channel |
| [**Roboflow**](roboflow.md)                         | Imports        | Imports every supported dataset in a Roboflow workspace from an API key      |
| [**Labelbox**](labelbox.md)                         | Imports        | Reads Labelbox NDJSON exports as datasets                                    |
| [**LabelMe**](labelme.md)                           | Imports        | Imports the YOLO export produced by the LabelMe Toolkit                      |
| [**CVAT**](cvat.md)                                 | Imports        | Imports CVAT Ultralytics YOLO and COCO exports — direct import coming soon   |
| [**Label Studio**](label-studio.md)                 | Imports        | Imports Label Studio YOLO and COCO exports — direct import coming soon       |

## Plans and Permissions

Slack alerts and every dataset import work on all plans. Google Cloud Storage, Amazon S3, and Azure Blob Storage
connections require a [Pro or Enterprise plan](../account/billing.md#plans), and On Premise requires an active
Enterprise plan.

Connecting, changing, or disconnecting cloud storage and Slack requires the workspace admin or owner
[role](../account/teams.md#roles-and-permissions). Importing datasets and connecting an On Premise host require the
editor role.

## FAQ

### Which integrations need a connection and which work from an upload?

On Premise, Amazon S3, Google Cloud Storage, Azure Blob Storage, and Slack are live connections managed from **Settings > Integrations**. Roboflow imports from an API key that is used once and discarded. Labelbox, LabelMe, CVAT, and Label Studio need no connection at all: export from the tool and upload the file as a new dataset.

### Do cloud storage datasets count against my Platform storage?

No. Cloud storage and On Premise datasets are indexed in place, so the images are streamed on demand and do not consume your [storage quota](../account/billing.md). Only labels, classes, and annotations are stored in Platform.

### Which plan do I need?

Slack alerts and every dataset import work on all plans. Cloud storage connections require a [Pro or Enterprise plan](../account/billing.md#plans), and On Premise requires an active Enterprise plan.

### Can I disconnect an integration without losing my datasets?

Yes. Disconnecting cloud storage or On Premise deletes the stored credentials but keeps the datasets, classes, labels, and annotations in your workspace. Their images become available again when you reconnect the same storage account or host.
