---
comments: true
description: Learn how to use Ultralytics YOLO11 for precise object counting in specified regions, enhancing efficiency across various applications.
keywords: object counting, regions, YOLO11, computer vision, Ultralytics, efficiency, accuracy, automation, real-time, applications, surveillance, monitoring
---

# Object Counting in Different Regions using Ultralytics YOLO 🚀

## What is Object Counting in Regions?

[Object counting](../guides/object-counting.md) in regions with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) involves precisely determining the number of objects within specified areas using advanced [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). This approach is valuable for optimizing processes, enhancing security, and improving efficiency in various applications.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/mzLfC13ISF4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Counting in Different Regions using Ultralytics YOLO11 | Ultralytics Solutions 🚀
</p>

## Advantages of Object Counting in Regions?

- **[Precision](https://www.ultralytics.com/glossary/precision) and Accuracy:** Object counting in regions with advanced computer vision ensures precise and accurate counts, minimizing errors often associated with manual counting.
- **Efficiency Improvement:** Automated object counting enhances operational efficiency, providing real-time results and streamlining processes across different applications.
- **Versatility and Application:** The versatility of object counting in regions makes it applicable across various domains, from manufacturing and surveillance to traffic monitoring, contributing to its widespread utility and effectiveness.

## Real World Applications

|                                                                                      Retail                                                                                       |                                                                                 Market Streets                                                                                  |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![People Counting in Different Region using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/people-counting-different-region-ultralytics-yolov8.avif) | ![Crowd Counting in Different Region using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/crowd-counting-different-region-ultralytics-yolov8.avif) |
|                                                           People Counting in Different Region using Ultralytics YOLO11                                                            |                                                           Crowd Counting in Different Region using Ultralytics YOLO11                                                           |

!!! example "Region Counting Example"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
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

        # Init RegionCounter
        region = solutions.RegionCounter(
            show=True,                 # display the frame
            region=region_points,      # pass region points
            model="yolo11n.pt",        # model for counting in regions i.e yolo11s.pt
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = region.count(im0)

            # Access the output
            # print(f"Region counts: , results['region_counts']")
            # print(f"Total tracks: , results['total_tracks']")

            video_writer.write(results["im0"])

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()     # destroy all opened windows
        ```

!!! tip "Ultralytics Example Code"

      The Ultralytics region counting module is available in our [examples section](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Region-Counter/yolov8_region_counter.py). You can explore this example for code customization and modify it to suit your specific use case.

### Argument `RegionCounter`

Here's a table with the `RegionCounter` arguments:

| Name         | Type    | Default                    | Description                                                                                                                                                                  |
| ------------ | ------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`      | `str`   | `None`                     | Path to Ultralytics YOLO Model File                                                                                                                                          |
| `region`     | `list`  | `[(20, 400), (1260, 400)]` | List of points defining the counting region.                                                                                                                                 |
| `line_width` | `int`   | `2`                        | Line thickness for bounding boxes.                                                                                                                                           |
| `show`       | `bool`  | `False`                    | Flag to control whether to display the video stream.                                                                                                                         |
| `tracker`    | `str`   | `botsort.yaml`             | Specifies the tracking algorithm to use, e.g., `bytetrack.yaml` or `botsort.yaml`.                                                                                           |
| `conf`       | `float` | `0.3`                      | Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives.                                                 |
| `iou`        | `float` | `0.5`                      | Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections.                   |
| `classes`    | `list`  | `None`                     | Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes.                                                                          |
| `max_det`    | `int`   | `300`                      | Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes. |
| `verbose`    | `bool`  | `True`                     | Controls the display of solutions results, providing a visual output of tracked objects.                                                                                     |

## FAQ

### What is object counting in specified regions using Ultralytics YOLO11?

Object counting in specified regions with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) involves detecting and tallying the number of objects within defined areas using advanced computer vision. This precise method enhances efficiency and [accuracy](https://www.ultralytics.com/glossary/accuracy) across various applications like manufacturing, surveillance, and traffic monitoring.

### How do I run the region based object counting script with Ultralytics YOLO11?

Follow these steps to run object counting in Ultralytics YOLO11:

1. Clone the Ultralytics repository and navigate to the directory:

    ```bash
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics/examples/YOLOv8-Region-Counter
    ```

2. Execute the region counting script:
    ```bash
    python yolov8_region_counter.py --source "path/to/video.mp4" --save-img
    ```

For more options, visit the [Run Region Counting](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Region-Counter/readme.md) section.

### Why should I use Ultralytics YOLO11 for object counting in regions?

Using Ultralytics YOLO11 for object counting in regions offers several advantages:

- **Precision and Accuracy:** Minimizes errors often seen in manual counting.
- **Efficiency Improvement:** Provides real-time results and streamlines processes.
- **Versatility and Application:** Applies to various domains, enhancing its utility.

Explore deeper benefits in the [Advantages](#advantages-of-object-counting-in-regions) section.

### What are some real-world applications of object counting in regions?

Object counting with Ultralytics YOLO11 can be applied to numerous real-world scenarios:

- **Retail:** Counting people for foot traffic analysis.
- **Market Streets:** Crowd density management.

Explore more examples in the [Real World Applications](#real-world-applications) section.
