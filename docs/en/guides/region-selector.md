---
comments: true
description: Learn how to interactively select polygon regions on video frames using Ultralytics YOLO RegionSelector for defining custom detection zones.
keywords: Ultralytics, YOLO, region selector, polygon selection, interactive region, object detection zones
---

# Interactive Region Selection using Ultralytics YOLO 🚀

This guide covers the Region Selector tool, part of the [Ultralytics Solutions](https://docs.ultralytics.com/solutions/) suite.

## What is Region Selector?

The Region Selector is an interactive tool that allows you to draw polygon regions on the first frame of a video using mouse clicks. This is particularly useful for defining custom detection zones, counting regions, or areas of interest for various computer vision tasks like [object counting](object-counting.md), [heatmaps](heatmaps.md), or [region counting](region-counting.md).

## Key Features

- **Interactive polygon drawing** on video frames
- **Visual feedback** with numbered points and connecting lines
- **Undo capability** to remove incorrectly placed points
- **Display detection** for headless environment compatibility

## Mouse and Keyboard Controls

| Control         | Action                                            |
| --------------- | ------------------------------------------------- |
| **Left Click**  | Add a new point to the polygon                    |
| **Right Click** | Remove the last added point (undo)                |
| **Enter**       | Confirm the selection (minimum 3 points required) |
| **ESC**         | Cancel selection and exit                         |

## Headless Environment Handling

The Region Selector requires a display to show the interactive window. When running in headless environments (servers or Docker containers without a display), the tool will:

1. Check for display availability using `check_imshow()`
2. Raise a `RuntimeError` with a clear message if no display is available
3. Suggest providing region coordinates directly as an alternative

!!! warning "Headless Environments"

    If you're running on a headless server, you cannot use the interactive selector. Instead, define your region coordinates manually:

    ```python
    # Define polygon points manually for headless environments
    region = [(100, 200), (400, 200), (400, 500), (100, 500)]
    ```

## Usage Examples

!!! example "Region Selection"

    === "Python (Interactive)" ===

        ```python
        from ultralytics.solutions import select_region

        # Open interactive selector on video's first frame
        region = select_region("path/to/video.mp4")

        if region:
            print(f"Selected region: {region}")
            # Use region with other solutions like ObjectCounter, RegionCounter, etc.
        else:
            print("Selection cancelled")
        ```

    === "Using with Object Counter" ===

        ```python
        from ultralytics import solutions

        # First, select your counting region interactively
        region = solutions.select_region("path/to/video.mp4")

        if region:
            # Then use it with ObjectCounter
            counter = solutions.ObjectCounter(
                model="yolo11n.pt",
                region=region,
                show=True,
            )
        ```

## Integration with Other Solutions

The selected region (a list of point tuples) can be used with various Ultralytics solutions:

| Solution                                | Use Case                                 |
| --------------------------------------- | ---------------------------------------- |
| [Object Counter](object-counting.md)    | Count objects within the region          |
| [Region Counter](region-counting.md)    | Count objects within the region          |
| [Heatmap](heatmaps.md)                  | Visualize activity within specific areas |
| [Queue Management](queue-management.md) | Monitor queue areas                      |

## Arguments

| Argument     | Type  | Default | Description            |
| ------------ | ----- | ------- | ---------------------- |
| `video_path` | `str` | —       | Path to the video file |

**Returns:** `list` of tuples `[(x1, y1), (x2, y2), ...]` or `None` if cancelled.
