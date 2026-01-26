---
comments: true
description: Learn how to organize and manage projects in Ultralytics Platform for efficient model development.
keywords: Ultralytics Platform, projects, model management, experiment tracking, YOLO
---

# Projects

[Ultralytics Platform](https://platform.ultralytics.com) projects provide an effective solution for organizing and managing your models. Group related models together to facilitate easier management, comparison, and development.

## Create Project

Navigate to **Projects** in the sidebar and click **Create Project**.

<!-- Screenshot: platform-projects-list.avif -->

??? tip "Quick Create"

    You can also create a project from the Home page quick actions.

Enter your project details:

- **Name**: A descriptive name for your project
- **Description**: Optional notes about the project purpose
- **Image**: Optional cover image for recognition

<!-- Screenshot: platform-projects-create.avif -->

Click **Create** to finalize. Your new project appears in the Projects list.

<!-- Screenshot: platform-projects-detail.avif -->

## Project Contents

Each project contains:

| Section      | Description                                  |
| ------------ | -------------------------------------------- |
| **Models**   | Trained checkpoints and their metrics        |
| **Charts**   | Compare model performance across experiments |
| **Activity** | History of changes and events                |
| **Settings** | Project configuration                        |

## Edit Project

Update project name, description, or settings:

1. Open project actions menu
2. Click **Edit**
3. Make changes
4. Click **Save**

<!-- Screenshot: platform-projects-settings.avif -->

## Delete Project

Remove a project you no longer need:

1. Open project actions menu
2. Click **Delete**
3. Confirm deletion

!!! warning "Cascading Delete"

    Deleting a project also deletes all models inside it. This action moves items to Trash where they can be restored within 30 days.

## Activity Log

Track all changes and events in your project:

- Model uploads and training starts
- Export jobs and downloads
- Settings changes

<!-- Screenshot: platform-projects-activity.avif -->

The activity log helps you:

- Audit who made changes
- Track experiment progress
- Debug issues

## Compare Models

Compare model performance across experiments using the **Charts** tab:

1. Navigate to your project
2. Click the **Charts** tab
3. View metrics comparison across all models

Available comparisons:

| Metric        | Description                                         |
| ------------- | --------------------------------------------------- |
| **Loss**      | Training and validation loss curves                 |
| **mAP50**     | Mean Average Precision at IoU 0.50                  |
| **mAP50-95**  | Mean Average Precision at IoU 0.50-0.95             |
| **Precision** | True positives / (True positives + False positives) |
| **Recall**    | True positives / (True positives + False negatives) |

!!! tip "Interactive Charts"

    - Hover to see exact values
    - Click legend items to hide/show models
    - Drag to zoom into specific regions

## Transfer Models

Move models between projects:

1. Open model actions menu
2. Click **Transfer**
3. Select destination project
4. Click **Save**

## FAQ

### How many models can a project contain?

There's no hard limit on models per project. However, for better organization, we recommend:

- Group related experiments (same dataset/task)
- Archive old experiments
- Use meaningful project names

### Can I restore a deleted project?

Yes, deleted projects go to Trash and can be restored within 30 days:

1. Go to Settings > Trash
2. Find the project
3. Click **Restore**
