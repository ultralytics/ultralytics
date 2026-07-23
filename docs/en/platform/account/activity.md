---
plans: [free, pro, enterprise]
title: Account Activity Feed
comments: true
description: Track all account activity and events on Ultralytics Platform with the activity feed, including training, uploads, and system events.
keywords: Ultralytics Platform, activity feed, audit log, notifications, event tracking, activity history
---

# Activity Feed

[Ultralytics Platform](https://platform.ultralytics.com) provides a comprehensive activity feed that tracks all events and actions across your account. Monitor training progress and system events in one centralized location.

![Ultralytics Platform Activity Page Inbox Tab With Event List](https://cdn.ul.run/i/afb598587971d5275b5c050911c2deb5.avif)<!-- screenshot -->

## Overview

The Activity Feed serves as your central hub for:

- **Training updates**: Job started, completed, failed, or cancelled
- **Data changes**: Datasets created, modified, or deleted
- **Model events**: Model creation, exports, and deployments
- **Project events**: Project creation, updates, and deletion
- **API key events**: Key creation and revocation
- **Settings changes**: Profile and account updates
- **System alerts**: Onboarding and account notifications

## Accessing Activity

Navigate to the Activity Feed in any of the following ways:

1. Click the activity indicator in the top navigation bar
2. Open the profile menu at the bottom of the sidebar and select **Activity**
3. Navigate directly to `/activity`

![Ultralytics Platform Activity Page Inbox With Search And Date Filter](https://cdn.ul.run/i/be73b7964145c52f6e6c701f6ef9e1bf.avif)<!-- screenshot -->

## Activity Types

The Platform tracks the following resource types and actions:

| Resource Type  | Description                                 | Icon Color          |
| -------------- | ------------------------------------------- | ------------------- |
| **project**    | [Project](../train/projects.md) events      | Blue                |
| **dataset**    | [Dataset](../data/datasets.md) events       | Green               |
| **model**      | [Model](../train/models.md) events          | Purple              |
| **training**   | Training job events                         | Blue/Green/Red/Gray |
| **settings**   | Account settings changes                    | Gray                |
| **api_key**    | [API key](api-keys.md) creation/revocation  | Amber               |
| **export**     | Model export events                         | Amber               |
| **deployment** | [Deployment](../deploy/endpoints.md) events | Blue                |
| **onboarding** | Onboarding completion                       | Green               |

### Action Types

Each event includes one of the following action types:

| Action        | Description                                          |
| ------------- | ---------------------------------------------------- |
| **created**   | Resource was created                                 |
| **updated**   | Resource was modified                                |
| **deleted**   | Resource was permanently deleted                     |
| **trashed**   | Resource was moved to trash                          |
| **restored**  | Resource was restored from trash                     |
| **started**   | Training or export job was started                   |
| **completed** | Training or export job finished successfully         |
| **failed**    | Training or export job failed                        |
| **cancelled** | Training or export job was cancelled                 |
| **uploaded**  | Data was uploaded (images, model weights)            |
| **shared**    | Resource visibility changed to public                |
| **unshared**  | Resource visibility changed to private               |
| **exported**  | Model was exported to a deployment format            |
| **cloned**    | Resource was cloned to another location              |
| **analyzed**  | Dataset analysis (embeddings/clustering) was started |

## Inbox and Archive

Organize your activity with two tabs:

### Inbox

The Inbox shows recent activity:

- New events appear here automatically
- Unseen events are highlighted with a colored background
- Events are automatically marked as seen when you view the page
- Click **Archive** on individual events to move them out of Inbox

### Archive

Move events to Archive to keep your Inbox clean:

- Click **Archive** on individual events
- Click **Archive all** to archive all Inbox events at once
- Access archived events via the `Archive` tab
- Click **Restore** on archived events to move them back to Inbox

## Search and Filtering

Find specific events quickly:

### Search

Use the search bar to find events by resource name or resource type.

### Date Range

Filter by time period using the date range picker:

- Select a start and end date
- The page defaults to the last 30 days
- Custom date ranges supported

![Ultralytics Platform Activity Page Date Range Picker Expanded](https://cdn.ul.run/i/32f81d792319c4b8fe0d31f2958f27b3.avif)<!-- screenshot -->

## Event Details

Each row displays:

| Field           | Description                                             |
| --------------- | ------------------------------------------------------- |
| **Event**       | Action and resource type (for example, Created Project) |
| **Resource**    | Recorded resource name                                  |
| **Time**        | Event timestamp                                         |
| **User email**  | Account member that performed the action                |
| **Resource ID** | Recorded resource identifier                            |
| **Actions**     | Undo, Archive, or Restore when available                |

## Undo Support

Recent settings changes support undo directly from the Activity feed:

- Click **Undo** next to the matching settings event to restore the previous value.
- Undo remains available for **one hour** in the browser session where the change was made. It does not persist after
  reloading or opening another browser.

## Pagination

The Activity feed supports pagination:

- Default page size: 20 events
- Navigate between pages using the pagination controls
- Page size is configurable via URL query parameter

## Export Activity

Click **Export** to download the events in the current Inbox or Archive view as JSON. The export respects the active
search and date filters.

## FAQ

### Can I export my activity history?

Yes. Click **Export** on the Activity page to download the current filtered view, or use the GDPR data export feature
in [`Settings > Profile`](settings.md#gdpr-compliance) to download account metadata including activity history.

### What happens to activity when I delete a resource?

The recorded event keeps its action, resource name, resource ID, time, and user email. The resource itself is no longer
available after permanent deletion.

### Does activity work with team workspaces?

Yes, the Activity feed shows events for the currently active workspace. Switch workspaces in the sidebar to see activity for different workspaces.
