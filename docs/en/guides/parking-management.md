---
title: Parking Management with Ultralytics YOLO26
comments: true
description: Detect occupied and available parking spaces in real time with Ultralytics YOLO26. Track vehicles, monitor lot occupancy, and export annotated video.
keywords: parking management, YOLO26, Ultralytics, vehicle detection, real-time tracking, parking lot optimization, smart parking, parking occupancy detection, parking slot detection, ParkingManagement, ParkingPtsSelection, yolo26 parking, parking space detection
---

# Build a Parking Management System with Ultralytics YOLO26 🚀

## What Is a Parking Management System?

A parking management system built with [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) detects vehicles in real time to track which parking spaces are occupied or available, then reports live lot occupancy. It pairs YOLO26 [object detection](https://www.ultralytics.com/glossary/object-detection) with a JSON-defined parking layout so you can monitor a whole lot from a single video stream or camera feed.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hsimB10D6Y0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Build a Parking Management System with Ultralytics YOLO26 | Real-Time Spot Detection 🚗
</p>

## Advantages of Parking Management System

- **Efficiency**: Optimizes the use of parking spaces and reduces congestion across the lot.
- **Safety and Security**: Improves the safety of people and vehicles through continuous surveillance.
- **Reduced Emissions**: Manages traffic flow to minimize idle time and emissions in parking lots.

## Real World Applications

|                                                                      Parking Management System                                                                      |                                                                       Parking Management System                                                                       |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Parking lots Analytics Using Ultralytics YOLO26](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/parking-management-aerial-view-ultralytics-yolov8.avif) | ![Parking management top view using Ultralytics YOLO26](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/parking-management-top-view-ultralytics-yolov8.avif) |
|                                                       Parking management Aerial View using Ultralytics YOLO26                                                       |                                                         Parking management Top View using Ultralytics YOLO26                                                          |

## Parking Management System Code Workflow

??? note "Points selection is now easy"

    Choosing parking points is a critical and complex task in parking management systems. Ultralytics streamlines this process by providing a tool "Parking slots annotator" that lets you define parking lot areas, which can be utilized later for additional processing.

**Step-1:** Capture a frame from the video or camera stream where you want to manage the parking lot.

**Step-2:** Use the provided code to launch a graphical interface, where you can select an image and start outlining parking regions by mouse click to create polygons.

!!! example "Parking slots annotator Ultralytics YOLO"

    ??? note "Additional step for installing `tkinter`"

        Generally, `tkinter` comes pre-packaged with Python. However, if it did not, you can install it using the highlighted steps:

        - **Linux**: (Debian/Ubuntu): `sudo apt install python3-tk`
        - **Fedora**: `sudo dnf install python3-tkinter`
        - **Arch**: `sudo pacman -S tk`
        - **Windows**: Reinstall Python and enable the checkbox `tcl/tk and IDLE` on **Optional Features** during installation
        - **MacOS**: Reinstall Python from [https://www.python.org/downloads/macos/](https://www.python.org/downloads/macos/) or `brew install python-tk`

    === "Python"

        ```python
        from ultralytics import solutions

        solutions.ParkingPtsSelection()
        ```

**Step-3:** After defining the parking areas with polygons, click `save` to store the data as `bounding_boxes.json` in your working directory — the same filename the management script loads below.

![Ultralytics YOLO26 Points Selection Demo](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-points-selection-demo.avif)

**Step-4:** You can now run the parking management [solution](../solutions/index.md) with the code below.

!!! example "Parking Management using Ultralytics YOLO"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        # Video capture
        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("parking management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize parking management object
        parkingmanager = solutions.ParkingManagement(
            model="yolo26n.pt",  # path to model file
            json_file="bounding_boxes.json",  # path to parking annotations file
        )

        while cap.isOpened():
            ret, im0 = cap.read()
            if not ret:
                break

            results = parkingmanager(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "CLI"

        ```bash
        yolo solutions parking source="path/to/video.mp4" json_file="bounding_boxes.json" show=True
        ```

        !!! note
            Create parking zone annotations first using `ParkingPtsSelection()` in Python (Step 2 above), then pass the JSON file to the CLI command.

### `ParkingManagement` Arguments

Here's a table with the `ParkingManagement` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "json_file"]) }}

The `ParkingManagement` solution allows the use of several [`track`](../modes/track.md) parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization options are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## FAQ

### How does Ultralytics YOLO26 enhance parking management systems?

Ultralytics YOLO26 provides **real-time [vehicle detection](../tasks/detect.md)** and monitoring for parking systems, which optimizes the use of parking spaces, reduces congestion, and improves safety through continuous surveillance. Running it on a live camera feed also helps manage traffic flow and minimize vehicle idle time. For the full setup, see the [parking management code workflow](#parking-management-system-code-workflow).

### What are the benefits of using Ultralytics YOLO26 for smart parking?

Using Ultralytics YOLO26 for smart parking yields numerous benefits:

- **Efficiency**: Optimizes the use of parking spaces and decreases congestion.
- **Safety and Security**: Enhances surveillance and ensures the safety of vehicles and pedestrians.
- **Environmental Impact**: Helps reduce emissions by minimizing vehicle idle times. Explore more benefits in the [Advantages of Parking Management System section](#advantages-of-parking-management-system).

### How can I define parking spaces using Ultralytics YOLO26?

Defining parking spaces is straightforward with Ultralytics YOLO26:

1. Capture a frame from a video or camera stream.
2. Use the provided code to launch a GUI for selecting an image and drawing polygons to define parking spaces.
3. Save the labeled data in JSON format for further processing. For comprehensive instructions, check the selection of points section above.

### How do I customize the ParkingManagement solution for my parking lot?

You can tailor the `ParkingManagement` solution through its [arguments](#parkingmanagement-arguments). Point `model` at a custom-trained detector, restrict detections to specific vehicle `classes`, adjust the `conf` and `iou` thresholds, switch the `tracker`, use `line_width` to scale the on-frame labels and occupancy readout, or select the inference `device`. For related zone-based monitoring, see the [object counting](object-counting.md) guide.

### What are some real-world applications of Ultralytics YOLO26 in parking lot management?

Ultralytics YOLO26 is utilized in various real-world applications for parking lot management, including:

- **Parking Space Detection**: Accurately identifying available and occupied spaces.
- **Surveillance**: Enhancing security through real-time monitoring.
- **Traffic Flow Management**: Reducing idle times and congestion with efficient traffic handling. Images showcasing these applications can be found in [real-world applications](#real-world-applications).
