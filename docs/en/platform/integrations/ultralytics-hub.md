---
comments: true
description: Import your datasets, projects, models, and account balance from Ultralytics HUB into Ultralytics Platform with a single API key.
keywords: Ultralytics Platform, Ultralytics HUB, HUB import, dataset migration, integrations, YOLO, computer vision
---

# Ultralytics HUB Integration

The Ultralytics HUB integration transfers everything from your [Ultralytics HUB](https://hub.ultralytics.com) account into [Ultralytics Platform](https://platform.ultralytics.com) in one step — your datasets, projects, models, and account balance.

## Import from Ultralytics HUB

1. Go to **Settings > [Integrations](index.md)** and find the **Ultralytics HUB** card.
2. Paste your **Ultralytics HUB API Key** and click **Import**.
3. Review the **Import from Ultralytics HUB?** preview dialog, which summarizes what will be transferred:
    - Number of datasets, projects, and models
    - Credits to be transferred, if any
    - Storage required, checked against your remaining storage
4. Click **Confirm** to start the import.

![Ultralytics Platform Settings Integrations Hub Import Dialog](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/platform/settings-integrations-hub-import-dialog.avif)

A progress bar tracks the import while it runs. When it finishes, the card shows a summary of everything that was imported.

## What Gets Imported

| Item         | Imported                         |
| ------------ | -------------------------------- |
| **Datasets** | Images, classes, and annotations |
| **Projects** | Projects                         |
| **Models**   | Trained models                   |
| **Credits**  | Credits                          |

!!! note "Where to find your HUB API key"

    Your Ultralytics HUB API key is available in your HUB account settings at [hub.ultralytics.com](https://hub.ultralytics.com). The key is used only to run the import — it is not stored.
