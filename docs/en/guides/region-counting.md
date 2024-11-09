---
comments: true
description: Learn how to use Ultralytics YOLOv8 for precise object counting in specified regions, enhancing efficiency across various applications.
keywords: object counting, regions, YOLOv8, computer vision, Ultralytics, efficiency, accuracy, automation, real-time, applications, surveillance, monitoring
---

# Object Counting in Different Regions using Ultralytics YOLOv8 🚀

## What is Object Counting in Regions?

[Object counting](../guides/object-counting.md) in regions with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) involves precisely determining the number of objects within specified areas using advanced [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). This approach is valuable for optimizing processes, enhancing security, and improving efficiency in various applications.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/okItf1iHlV8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLOv8 Object Counting in Multiple & Movable Regions
</p>

## Advantages of Object Counting in Regions?

- **[Precision](https://www.ultralytics.com/glossary/precision) and Accuracy:** Object counting in regions with advanced computer vision ensures precise and accurate counts, minimizing errors often associated with manual counting.
- **Efficiency Improvement:** Automated object counting enhances operational efficiency, providing real-time results and streamlining processes across different applications.
- **Versatility and Application:** The versatility of object counting in regions makes it applicable across various domains, from manufacturing and surveillance to traffic monitoring, contributing to its widespread utility and effectiveness.

## Real World Applications

|                                                                                      Retail                                                                                       |                                                                                 Market Streets                                                                                  |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![People Counting in Different Region using Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/people-counting-different-region-ultralytics-yolov8.avif) | ![Crowd Counting in Different Region using Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/crowd-counting-different-region-ultralytics-yolov8.avif) |
|                                                           People Counting in Different Region using Ultralytics YOLOv8                                                            |                                                           Crowd Counting in Different Region using Ultralytics YOLOv8                                                           |

!!! example "Region Counting Example"

    === "Python"

        ```python
         import cv2
         from ultralytics import solutions

         cap = cv2.VideoCapture("Path/to/video/file.mp4")
         assert cap.isOpened(), "Error reading video file"
         w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

         # Define region points
         # region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)] # Pass region as list

         # pass region as dictionary
         region_points = {
             "region-01": [(50, 50), (250, 50), (250, 250), (50, 250)],
             "region-02": [(640, 640), (780, 640), (780, 720), (640, 720)]
         }

         # Video writer
         video_writer = cv2.VideoWriter("region_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

         # Init Object Counter
         region = solutions.RegionCounter(
             show=True,
             region=region_points,
             model="yolo11n.pt",
         )

         # Process video
         while cap.isOpened():
             success, im0 = cap.read()
             if not success:
                 print("Video frame is empty or video processing has been successfully completed.")
                 break
             im0 = region.count(im0)
             video_writer.write(im0)

         cap.release()
         video_writer.release()
         cv2.destroyAllWindows()
        ```

!!! tip "Ultralytics Example Code"

      The Ultralytics region counting module is available in our [examples section](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Region-Counter/yolov8_region_counter.py). You can explore this example for code customization and modify it to suit your specific use case.

### Argument `RegionCounter`

Here's a table with the `RegionCounter` arguments:

| Name         | Type   | Default                    | Description                                          |
| ------------ | ------ | -------------------------- | ---------------------------------------------------- |
| `model`      | `str`  | `None`                     | Path to Ultralytics YOLO Model File                  |
| `region`     | `list` | `[(20, 400), (1260, 400)]` | List of points defining the counting region.         |
| `line_width` | `int`  | `2`                        | Line thickness for bounding boxes.                   |
| `show`       | `bool` | `False`                    | Flag to control whether to display the video stream. |

## FAQ

### What is object counting in specified regions using Ultralytics YOLOv8?

Object counting in specified regions with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) involves detecting and tallying the number of objects within defined areas using advanced computer vision. This precise method enhances efficiency and [accuracy](https://www.ultralytics.com/glossary/accuracy) across various applications like manufacturing, surveillance, and traffic monitoring.

### How do I run the object counting script with Ultralytics YOLOv8?

Follow these steps to run object counting in Ultralytics YOLOv8:

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

### Why should I use Ultralytics YOLOv8 for object counting in regions?

Using Ultralytics YOLOv8 for object counting in regions offers several advantages:

- **Precision and Accuracy:** Minimizes errors often seen in manual counting.
- **Efficiency Improvement:** Provides real-time results and streamlines processes.
- **Versatility and Application:** Applies to various domains, enhancing its utility.

Explore deeper benefits in the [Advantages](#advantages-of-object-counting-in-regions) section.

### Can the defined regions be adjusted during video playback?

Yes, with Ultralytics YOLOv8, regions can be interactively moved during video playback. Simply click and drag with the left mouse button to reposition the region. This feature enhances flexibility for dynamic environments. Learn more in the tip section for [movable regions](https://github.com/ultralytics/ultralytics/blob/33cdaa5782efb2bc2b5ede945771ba647882830d/examples/YOLOv8-Region-Counter/yolov8_region_counter.py#L39).

### What are some real-world applications of object counting in regions?

Object counting with Ultralytics YOLOv8 can be applied to numerous real-world scenarios:

- **Retail:** Counting people for foot traffic analysis.
- **Market Streets:** Crowd density management.

Explore more examples in the [Real World Applications](#real-world-applications) section.
