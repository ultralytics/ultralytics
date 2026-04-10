---
comments: true
description: Learn how to recover deleted projects, datasets, and models from Trash on Ultralytics Platform with the 30-day soft delete policy.
keywords: Ultralytics Platform, trash, restore, soft delete, recover, deleted items, data recovery
---

# Trash and Restore

[Ultralytics Platform](https://platform.ultralytics.com) implements a 30-day soft delete policy, allowing you to recover accidentally deleted projects, datasets, and models. Deleted items are moved to Trash where they can be restored before permanent deletion.

<!-- Screenshot: platform-trash-overview.avif -->

## Soft Delete Policy

When you delete a resource on the Platform:

1. **Immediate**: Item moves to Trash (not permanently deleted)
2. **30 Days**: Item remains recoverable in Trash
3. **After 30 Days**: Item is permanently deleted automatically

!!! success "Recovery Window"

    You have 30 days to restore any deleted item. After this period, the item and all associated data are permanently removed and cannot be recovered.

## Accessing Trash

Navigate to your Trash:

1. Go to **Settings** (gear icon)
2. Click **Trash** in the sidebar
3. Or navigate directly to Settings > Trash

<!-- Screenshot: platform-trash-list.avif -->

## Trash Contents

The Trash shows all soft-deleted resources:

| Resource Type | What's Included When Deleted               |
| ------------- | ------------------------------------------ |
| **Projects**  | Project + all models inside                |
| **Datasets**  | Dataset + all images and annotations       |
| **Models**    | Model weights + training history + exports |

### Viewing Trash Items

Each item in Trash displays:

- **Name**: Original resource name
- **Type**: Project, Dataset, or Model
- **Deleted**: Date and time of deletion
- **Expires**: When permanent deletion occurs
- **Size**: Storage used by the item

## Restoring Items

Recover a deleted item:

1. Navigate to **Settings > Trash**
2. Find the item you want to restore
3. Click the **Restore** button
4. Confirm restoration

<!-- Screenshot: platform-trash-restore.avif -->

The item returns to its original location with all data intact.

### Restore Behavior

| Resource | Restore Behavior                                                             |
| -------- | ---------------------------------------------------------------------------- |
| Project  | Restores project and all contained models                                    |
| Dataset  | Restores dataset with all images and annotations                             |
| Model    | Restores model to original project (or orphaned if project was also deleted) |

!!! note "Parent Dependency"

    If you deleted both a project and its models, restore the project first. This automatically restores all models that were inside it.

## Permanent Deletion

### Automatic Deletion

Items in Trash are automatically and permanently deleted after 30 days. This process:

- Runs daily
- Removes items older than 30 days
- Frees up storage space
- Cannot be reversed

### Empty Trash

Permanently delete all items immediately:

1. Navigate to **Settings > Trash**
2. Click **Empty Trash**
3. Confirm the action

!!! warning "Irreversible Action"

    Emptying Trash permanently deletes all items immediately. This action cannot be undone and all data will be lost.

### Delete Single Item Permanently

To permanently delete one item without waiting:

1. Find the item in Trash
2. Click the **Delete Permanently** button
3. Confirm deletion

## Storage and Trash

Items in Trash still count toward your storage quota:

| Scenario             | Storage Impact                 |
| -------------------- | ------------------------------ |
| Delete item          | Storage remains allocated      |
| Restore item         | No change (was still counting) |
| Permanent deletion   | Storage freed                  |
| 30-day auto-deletion | Storage freed automatically    |

!!! tip "Free Up Storage"

    If you're running low on storage, empty Trash or permanently delete specific items to immediately reclaim space.

## API Access

Manage Trash programmatically via the REST API:

```bash
# List items in Trash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://platform.ultralytics.com/api/trash

# Restore an item
curl -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"itemId": "item_abc123", "type": "dataset"}' \
  https://platform.ultralytics.com/api/trash

# Empty Trash (permanently delete all)
curl -X POST -H "Authorization: Bearer YOUR_API_KEY" \
  https://platform.ultralytics.com/api/trash/empty
```

See [REST API Reference](../api/index.md#trash-api) for complete documentation.

## FAQ

### Can I restore an item after 30 days?

No. After 30 days, items are permanently deleted and cannot be recovered. Make sure to restore important items before the expiration date shown in Trash.

### What happens when I delete a project with models?

Both the project and all models inside it move to Trash together. Restoring the project restores all its models. You can also restore individual models separately.

### Do items in Trash count toward storage?

Yes, items in Trash continue to use storage quota. To free up space, permanently delete items or empty Trash.

### Can I recover a model if its project was permanently deleted?

No. If a project is permanently deleted, all models that were inside it are also permanently deleted. Always restore items before the 30-day window expires.

### How do I know when an item will be permanently deleted?

Each item in Trash shows an "Expires" date indicating when automatic permanent deletion will occur.
