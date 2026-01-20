---
comments: true
description: Track all account activity and events on Ultralytics Platform with the activity feed, including training, uploads, and system events.
keywords: Ultralytics Platform, activity feed, audit log, notifications, event tracking, activity history
---

# Activity Feed

[Ultralytics Platform](https://platform.ultralytics.com) provides a comprehensive activity feed that tracks all events and actions across your account. Monitor training progress and system events in one centralized location.

<!-- Screenshot: platform-activity-overview.avif -->

## Overview

The Activity Feed serves as your central hub for:

- **Training updates**: Job started, completed, failed, or cancelled
- **Data changes**: Datasets uploaded, modified, or deleted
- **Model events**: Exports, deployments, and inference activity
- **System alerts**: Billing, storage, and account notifications

## Accessing Activity

Navigate to the Activity Feed:

1. Click your profile icon in the top right
2. Select **Activity** from the dropdown
3. Or navigate to **Settings > Activity**

<!-- Screenshot: platform-activity-feed.avif -->

## Activity Types

The Platform tracks the following event types:

| Event Type    | Description                           | Icon  |
| ------------- | ------------------------------------- | ----- |
| **created**   | New resource created                  | +     |
| **updated**   | Resource modified                     | edit  |
| **deleted**   | Resource permanently removed          | trash |
| **trashed**   | Resource moved to trash (recoverable) | trash |
| **restored**  | Resource restored from trash          | undo  |
| **started**   | Training or export job started        | play  |
| **completed** | Job finished successfully             | check |
| **failed**    | Job encountered an error              | error |
| **cancelled** | Job stopped by user                   | stop  |
| **uploaded**  | File or dataset uploaded              | cloud |
| **exported**  | Model exported to format              | save  |
| **cloned**    | Resource duplicated                   | copy  |

## Inbox and Archive

Organize your activity with tabs:

### Inbox

The Inbox shows recent, unread activity:

- New events appear here automatically
- Unread events are highlighted
- Click an event to view details and mark as seen

### Archive

Move events to Archive to keep your Inbox clean:

1. Select events to archive
2. Click **Archive**
3. Access archived events via the Archive tab

!!! tip "Bulk Actions"

    Select multiple events using checkboxes to archive or mark as seen in bulk.

## Search and Filtering

Find specific events quickly:

### Search

Use the search bar to find events by:

- Resource name (dataset, model, project)
- Event description

### Filters

Filter events by type:

| Filter       | Shows                               |
| ------------ | ----------------------------------- |
| **All**      | All activity types                  |
| **Training** | Training started, completed, failed |
| **Uploads**  | Dataset and model uploads           |
| **Exports**  | Model export activity               |
| **System**   | Billing, storage, account events    |

### Date Range

Filter by time period:

- **Today**: Events from today
- **This Week**: Events from the past 7 days
- **This Month**: Events from the past 30 days
- **Custom**: Select specific date range

<!-- Screenshot: platform-activity-filters.avif -->

## Event Details

Click an event to view details:

| Field           | Description                       |
| --------------- | --------------------------------- |
| **Timestamp**   | When the event occurred           |
| **User**        | Who triggered the event           |
| **Resource**    | What was affected (with link)     |
| **Description** | Detailed event information        |
| **Metadata**    | Additional context (job ID, etc.) |

## Mark as Seen

Mark events as seen to track what you've reviewed:

- Click the checkmark icon on individual events
- Use **Mark All Seen** to clear all unread indicators
- Seen events remain accessible but are no longer highlighted

## API Access

Access activity programmatically via the REST API:

```bash
# List activity
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://platform.ultralytics.com/api/activity

# Filter by date range
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "https://platform.ultralytics.com/api/activity?startDate=2024-01-01&endDate=2024-01-31"

# Mark events as seen
curl -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  https://platform.ultralytics.com/api/activity/mark-seen

# Archive events
curl -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  https://platform.ultralytics.com/api/activity/archive
```

See [REST API Reference](../api/index.md#activity-api) for complete documentation.

## FAQ

### How long is activity history retained?

Activity history is retained indefinitely for your account. Archived events are also kept permanently.

### Can I export my activity history?

Yes, use the GDPR data export feature in Settings > Privacy to download all account data including activity history.

### Can I disable activity notifications?

Activity events are always logged for audit purposes. Email notifications can be configured in Settings > Notifications.

### What happens to activity when I delete a resource?

The activity event remains in your history with a note that the resource was deleted. You can still see what happened even after deletion.
