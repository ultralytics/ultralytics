---
plans: [free, pro, enterprise]
title: Platform Integrations
comments: true
description: Connect Ultralytics Platform to Slack, existing tools, cloud storage, and Enterprise On Premise compute and datasets.
keywords: Ultralytics Platform, integrations, Slack, alerts, data import, Roboflow, Ultralytics HUB, cloud storage, GCS, Amazon S3, Azure Blob Storage, On Premise, dataset migration, YOLO, computer vision
---

# Integrations

[Ultralytics Platform](https://platform.ultralytics.com) [integrations](../../integrations/index.md) connect your workspace to other tools and services you already use. Send job results to Slack, import existing datasets with a single API key, or connect your cloud storage and use the data where it lives.

![Ultralytics Platform Settings Integrations Tab](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/settings-integrations-tab.avif)<!-- screenshot -->

## Accessing Integrations

All integrations are managed from your account settings:

1. Go to **Settings > Integrations**
2. Select the service from the integration list on the left
3. Follow the connection prompts

HUB and Roboflow imports start with a preview so you can review what will be transferred and confirm that you have
enough [storage](../account/billing.md). Cloud storage connections verify list and read access before anything is saved.

## Available Integrations

| Integration                                         | What it does                                        |
| --------------------------------------------------- | --------------------------------------------------- |
| [**Slack**](slack.md)                               | Posts selected job results to one Slack channel     |
| [**Ultralytics HUB**](ultralytics-hub.md)           | Imports datasets, projects, models, and balance     |
| [**Roboflow**](roboflow.md)                         | Imports datasets                                    |
| [**Google Cloud Storage**](google-cloud-storage.md) | Indexes datasets in place from your GCS buckets     |
| [**Amazon S3**](amazon-s3.md)                       | Indexes datasets in place from your S3 buckets      |
| [**Azure Blob Storage**](azure-blob-storage.md)     | Indexes datasets in place from your blob containers |
| [**On Premise**](on-premise.md)                     | Runs local CPU/GPU workers while pixels stay local  |

Slack alerts are available on every plan. Google Cloud Storage, Amazon S3, and Azure Blob Storage connections require a
[Pro or Enterprise plan](../account/billing.md#plans).
