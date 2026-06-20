---
comments: true
description: Generate real-time object tracking heatmaps on video with the Ultralytics YOLO26 Heatmap solution to visualize traffic flow and crowd movement patterns.
keywords: Ultralytics, YOLO26, heatmap, object tracking heatmap, video heatmap, traffic flow visualization, crowd movement analysis, Heatmap solution, OpenCV colormaps, computer vision
---

# Object Tracking Heatmaps with Ultralytics YOLO26

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-generate-heatmaps-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Heatmaps In Colab"></a>

The [Heatmap solution](../reference/solutions/heatmap.md) in [Ultralytics YOLO26](../models/yolo26.md) tracks objects across video frames and overlays their accumulated movement intensity onto each frame, so busy areas glow in warm colors while quiet areas stay cool. Built on YOLO26 [object tracking](../modes/track.md), it turns any video into a spatial activity map that reveals traffic flow, crowd movement, and dwell zones with a single Python call or CLI command.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/4ezde5-nZZw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Heatmaps using Ultralytics YOLO26
</p>

## Why Use Heatmaps for Video Analytics?

- **Spot activity patterns at a glance:** Intensity accumulates wherever tracked objects spend time, so high-traffic lanes, popular shelves, or crowd bottlenecks stand out without manual frame-by-frame review.
- **Tracking built in:** The solution runs YOLO26 detection and tracking internally, so there is no separate tracking pipeline to wire up.
- **Counting in the same pass:** Pass a `region` to count objects entering and exiting a zone while the heatmap builds, combining two analytics tasks in one run.

## Real World Applications

|                                                                                Transportation                                                                                |                                                                                Retail                                                                                 |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLO heatmap overlay showing vehicle traffic density](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-transportation-heatmap.avif) | ![Ultralytics YOLO heatmap overlay showing retail customer movement](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-retail-heatmap.avif) |
|                                                                   Ultralytics YOLO Transportation Heatmap                                                                    |                                                                    Ultralytics YOLO Retail Heatmap                                                                    |

## How to Generate Heatmaps with Ultralytics YOLO

Run the Heatmap solution on a video source from the CLI or Python. The Python example writes the processed frames to an output video file:

!!! example "Heatmaps using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Run a heatmap example
        yolo solutions heatmap show=True

        # Pass a source video
        yolo solutions heatmap source="path/to/video.mp4"

        # Pass a custom colormap
        yolo solutions heatmap colormap=cv2.COLORMAP_INFERNO

        # Heatmaps + object counting
        yolo solutions heatmap region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # For object counting with heatmap, you can pass region points.
        # region_points = [(20, 400), (1080, 400)]                                      # line points
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # rectangle region
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon points

        # Initialize heatmap object
        heatmap = solutions.Heatmap(
            show=True,  # display the output
            model="yolo26n.pt",  # path to the YOLO26 model file
            colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
            # region=region_points,  # object counting with heatmaps, you can pass region_points
            # classes=[0, 2],  # generate heatmap for specific classes, e.g., person and car.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = heatmap(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

## `Heatmap()` Arguments

Here's a table with the `Heatmap` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "colormap", "show_in", "show_out", "region"]) }}

You can also apply different `track` arguments in the `Heatmap` solution.

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the supported visualization arguments are listed below:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

### Heatmap Colormaps

The `colormap` argument accepts any [OpenCV colormap](https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html). Pass the constant from the `cv2` module, for example `colormap=cv2.COLORMAP_INFERNO`:

| Colormap Name                   | Description                            |
| ------------------------------- | -------------------------------------- |
| `cv2.COLORMAP_AUTUMN`           | Autumn color map                       |
| `cv2.COLORMAP_BONE`             | Bone color map                         |
| `cv2.COLORMAP_JET`              | Jet color map                          |
| `cv2.COLORMAP_WINTER`           | Winter color map                       |
| `cv2.COLORMAP_RAINBOW`          | Rainbow color map                      |
| `cv2.COLORMAP_OCEAN`            | Ocean color map                        |
| `cv2.COLORMAP_SUMMER`           | Summer color map                       |
| `cv2.COLORMAP_SPRING`           | Spring color map                       |
| `cv2.COLORMAP_COOL`             | Cool color map                         |
| `cv2.COLORMAP_HSV`              | HSV (Hue, Saturation, Value) color map |
| `cv2.COLORMAP_PINK`             | Pink color map                         |
| `cv2.COLORMAP_HOT`              | Hot color map                          |
| `cv2.COLORMAP_PARULA`           | Parula color map                       |
| `cv2.COLORMAP_MAGMA`            | Magma color map                        |
| `cv2.COLORMAP_INFERNO`          | Inferno color map                      |
| `cv2.COLORMAP_PLASMA`           | Plasma color map                       |
| `cv2.COLORMAP_VIRIDIS`          | Viridis color map                      |
| `cv2.COLORMAP_CIVIDIS`          | Cividis color map                      |
| `cv2.COLORMAP_TWILIGHT`         | Twilight color map                     |
| `cv2.COLORMAP_TWILIGHT_SHIFTED` | Shifted Twilight color map             |
| `cv2.COLORMAP_TURBO`            | Turbo color map                        |
| `cv2.COLORMAP_DEEPGREEN`        | Deep Green color map                   |

## How Heatmaps Work

The [Heatmap solution](../reference/solutions/heatmap.md) extends the [ObjectCounter](../reference/solutions/object_counter.md) class. On the first processed frame it creates a blank intensity layer matching the frame size. Each frame is then processed in two steps:

1. YOLO26 tracking detects and follows every object in the frame
2. For each tracked object, the heatmap intensity increases within a circular region centered in its bounding box

Once per frame, the accumulated intensity layer is normalized, colorized with the selected colormap, and blended with the original frame. The overlay appears as soon as at least one object is tracked; frames without tracked objects are shown without the heatmap overlay.

The result is a dynamic visualization that builds up over time, revealing traffic patterns, crowd movements, or other spatial behaviors in your video data. When a `region` is set, the solution also [counts objects](object-counting.md) entering and exiting that region while the heatmap builds.

## Conclusion

The Ultralytics YOLO26 Heatmap solution turns object tracking results into an intuitive activity overlay with a few lines of code. To go further, combine it with [object counting](object-counting.md), explore the other [Ultralytics Solutions](../solutions/index.md), or read about the underlying [tracking mode](../modes/track.md).

## FAQ

### How does Ultralytics YOLO26 generate heatmaps from a video?

Ultralytics YOLO26 generates heatmaps by tracking objects across video frames and accumulating an intensity value at each tracked object's location, then colorizing the result and blending it with the original frame. Areas where objects appear frequently or linger build up higher intensity and render in warmer colors. For configuration options, refer to the [`Heatmap()` Arguments](#heatmap-arguments) section.

### How do I save the heatmap output to a video file?

Use OpenCV's `cv2.VideoWriter` and write `results.plot_im` for every processed frame, as shown in the [main example](#how-to-generate-heatmaps-with-ultralytics-yolo). The `plot_im` attribute holds the frame with the heatmap overlay already applied.

### Can I combine heatmaps with object counting?

Yes. Pass a `region` argument to `Heatmap()` with line, rectangle, or polygon points, and the solution counts objects entering and exiting that region while the heatmap builds. The returned results include `in_count`, `out_count`, and per-class counts. See the [object counting guide](object-counting.md) for region configuration details.

### How can I visualize only specific object classes in heatmaps using Ultralytics YOLO26?

Pass the `classes` argument to `Heatmap()` with the class indices you want to keep. For example, `classes=[0, 2]` builds the heatmap only from persons and cars (COCO class indices 0 and 2):

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
heatmap = solutions.Heatmap(show=True, model="yolo26n.pt", classes=[0, 2])

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = heatmap(im0)
cap.release()
cv2.destroyAllWindows()
```

### What makes Ultralytics YOLO26 heatmaps different from other data visualization tools like those from [OpenCV](https://www.ultralytics.com/glossary/opencv) or Matplotlib?

Ultralytics YOLO26 heatmaps integrate [object detection](https://www.ultralytics.com/glossary/object-detection), tracking, intensity accumulation, and overlay rendering into a single call, while generic tools like OpenCV or Matplotlib require you to build that pipeline yourself. The solution processes video streams in real time and supports persistent tracking and customizable colormaps out of the box. For details on the underlying model, see the [YOLO26 model page](../models/yolo26.md).
