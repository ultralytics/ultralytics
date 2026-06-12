---
comments: true
description: Count objects in multiple user-defined polygonal zones of a video with the Ultralytics YOLO26 RegionCounter solution and read the counts in your code.
keywords: region counting, zone counting, object counting in regions, RegionCounter, Ultralytics YOLO26, count objects in a zone, polygon region, video analytics, people counting, vehicle counting
---

# Object Counting in Regions with Ultralytics YOLO26

The [RegionCounter solution](../reference/solutions/region_counter.md) in [Ultralytics YOLO26](../models/yolo26.md) tracks objects across video frames and, on every frame, counts the objects whose bounding-box center falls inside each region you define. Each region is drawn on the frame with its own live count, so you can monitor several zones, such as store aisles, road lanes, or production areas, with a single Python call or CLI command.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/mzLfC13ISF4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Counting in Different Regions using Ultralytics YOLO | Ultralytics Solutions
</p>

## Why Use Region Counting?

- **Several zones in one pass:** Pass a dictionary of named polygons and each zone gets an independent count drawn on the frame, so one video stream covers every area you care about.
- **Tracking built in:** The solution runs YOLO26 detection and [object tracking](../modes/track.md) internally, so there is no separate tracking pipeline to wire up.
- **Counts you can read programmatically:** Every processed frame returns a results object with a dictionary of region counts, ready for dashboards, alerts, or logging.

## Real World Applications

|                                                                                                Retail                                                                                                 |                                                                                     Market Streets                                                                                      |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLO counting people inside drawn polygon regions in a retail store](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/people-counting-different-region-ultralytics-yolov8.avif) | ![Ultralytics YOLO region counts overlaid on a crowded market street](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/crowd-counting-different-region-ultralytics-yolov8.avif) |
|                                                                              Ultralytics YOLO People Counting in Regions                                                                              |                                                                       Ultralytics YOLO Crowd Counting in Regions                                                                        |

## How to Count Objects in Regions with Ultralytics YOLO

Run the RegionCounter solution on a video source from the CLI or Python. A region can be a list of points (a single zone) or a dictionary that maps zone names to point lists. The Python example below counts objects in two named regions and writes the processed frames to an output video file:

!!! example "Region counting using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Run a region counting example
        yolo solutions region show=True

        # Pass a source video
        yolo solutions region source="path/to/video.mp4"

        # Pass a custom region
        yolo solutions region region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Pass region as list
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

        # Pass region as dictionary
        region_points = {
            "region-01": [(50, 50), (250, 50), (250, 250), (50, 250)],
            "region-02": [(640, 640), (780, 640), (780, 720), (640, 720)],
        }

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("region_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize region counter object
        regioncounter = solutions.RegionCounter(
            show=True,  # display the frame
            region=region_points,  # pass region points
            model="yolo26n.pt",  # model for counting in regions, e.g., yolo26s.pt
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = regioncounter(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

## `RegionCounter()` Arguments

Here's a table with the `RegionCounter` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

The `RegionCounter` solution enables the use of object tracking parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the following visualization settings are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## How Region Counting Works

The [RegionCounter solution](../reference/solutions/region_counter.md) turns each region you pass into a polygon, with at least three points per region. A plain list of points becomes a single zone named `Region#01`, while a dictionary keeps your own zone names; each zone is drawn in its own color from the Ultralytics palette. Every frame is then processed in three steps:

1. YOLO26 tracking detects and follows every object in the frame
2. For each tracked object, the solution computes the bounding-box center and checks which region polygons contain it
3. Each region's count is drawn at its center, and the counts reset before the next frame

Because counts restart every frame, the number shown on a region is its current occupancy, meaning how many objects are inside it right now, not a running total. To count objects crossing a boundary cumulatively, use the [ObjectCounter solution](object-counting.md) instead.

Each call also returns a results object whose `plot_im` attribute holds the annotated frame and whose `region_counts` dictionary maps region names to their counts, so you can feed the numbers into your own application logic.

## Conclusion

The Ultralytics YOLO26 RegionCounter solution monitors live object counts in as many named zones as you need with a few lines of code. To go further, count boundary crossings with [object counting](object-counting.md), follow objects inside a single zone with [TrackZone](trackzone.md), or explore the other [Ultralytics Solutions](../solutions/index.md).

## FAQ

### How does object counting in regions work in Ultralytics YOLO26?

Ultralytics YOLO26 counts objects in regions by tracking every object in the frame and testing whether its bounding-box center lies inside each user-defined polygon. The count shown on a region is its current occupancy for that frame, not a cumulative total. Configuration options are listed in the [`RegionCounter()` Arguments](#regioncounter-arguments) section.

### How do I count objects in multiple regions at once?

Pass the `region` argument as a dictionary that maps zone names to point lists, as in the [main example](#how-to-count-objects-in-regions-with-ultralytics-yolo):

```python
from ultralytics import solutions

region_points = {
    "region-01": [(50, 50), (250, 50), (250, 250), (50, 250)],
    "region-02": [(640, 640), (780, 640), (780, 720), (640, 720)],
}
regioncounter = solutions.RegionCounter(region=region_points, model="yolo26n.pt", show=True)
```

Each named zone is drawn in its own color with an independent count.

### How do I get the region counts programmatically?

The object returned by each `regioncounter(im0)` call carries a `region_counts` dictionary that maps region names to their counts, alongside `plot_im` (the annotated frame) and `total_tracks` (the number of tracked objects). Print the results object to inspect all available fields.

### Can I count only specific object classes in a region?

Yes. Pass the `classes` argument with the class indices to keep, for example `classes=[0]` to count only persons with a COCO-pretrained model. The full list of tracking-related arguments is in the [`RegionCounter()` Arguments](#regioncounter-arguments) section.

### What is the difference between RegionCounter and ObjectCounter?

`RegionCounter` shows how many objects are inside each polygonal zone on the current frame, while [ObjectCounter](object-counting.md) counts objects cumulatively as they cross a line or region boundary, tracking in and out totals. Use `RegionCounter` for live occupancy of one or more zones and `ObjectCounter` for entry and exit counting; line-shaped regions are supported by `ObjectCounter` only.
