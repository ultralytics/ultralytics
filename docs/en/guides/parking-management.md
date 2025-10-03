---
comments: true
description: Optimize parking spaces and enhance safety with Ultralytics YOLO11. Explore real-time vehicle detection and smart parking solutions.
keywords: parking management, YOLO11, Ultralytics, vehicle detection, real-time tracking, parking lot optimization, smart parking
---

# Parking Management using Ultralytics YOLO11 🚀

## What is Parking Management System?

Parking management with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) ensures efficient and safe parking by organizing spaces and monitoring availability. YOLO11 can improve parking lot management through real-time vehicle detection, and insights into parking occupancy.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/WwXnljc7ZUM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Implement Parking Management Using Ultralytics YOLO 🚀
</p>

## Advantages of Parking Management System?

- **Efficiency**: Parking lot management optimizes the use of parking spaces and reduces congestion.
- **Safety and Security**: Parking management using YOLO11 improves the safety of both people and vehicles through surveillance and security measures.
- **Reduced Emissions**: Parking management using YOLO11 manages traffic flow to minimize idle time and emissions in parking lots.

## Real World Applications

|                                                                     Parking Management System                                                                      |                                                                      Parking Management System                                                                       |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Parking lots Analytics Using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/parking-management-aerial-view-ultralytics-yolov8.avif) | ![Parking management top view using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/parking-management-top-view-ultralytics-yolov8.avif) |
|                                                      Parking management Aerial View using Ultralytics YOLO11                                                       |                                                         Parking management Top View using Ultralytics YOLO11                                                         |

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

**Step-3:** After defining the parking areas with polygons, click `save` to store a JSON file with the data in your working directory.

![Ultralytics YOLO11 Points Selection Demo](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-points-selection-demo.avif)

**Step-4:** You can now utilize the provided code for parking management with Ultralytics YOLO.

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
            model="yolo11n.pt",  # path to model file
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

### `ParkingManagement` Arguments

Here's a table with the `ParkingManagement` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "json_file"]) }}

The `ParkingManagement` solution allows the use of several `track` parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization options are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## FAQ

### How does Ultralytics YOLO11 enhance parking management systems?

Ultralytics YOLO11 greatly enhances parking management systems by providing **real-time vehicle detection** and monitoring. This results in optimized usage of parking spaces, reduced congestion, and improved safety through continuous surveillance. The [Parking Management System](https://github.com/ultralytics/ultralytics) enables efficient traffic flow, minimizing idle times and emissions in parking lots, thereby contributing to environmental sustainability. For further details, refer to the [parking management code workflow](#parking-management-system-code-workflow).

### What are the benefits of using Ultralytics YOLO11 for smart parking?

Using Ultralytics YOLO11 for smart parking yields numerous benefits:

- **Efficiency**: Optimizes the use of parking spaces and decreases congestion.
- **Safety and Security**: Enhances surveillance and ensures the safety of vehicles and pedestrians.
- **Environmental Impact**: Helps reduce emissions by minimizing vehicle idle times. Explore more benefits in the [Advantages of Parking Management System section](#advantages-of-parking-management-system).

### How can I define parking spaces using Ultralytics YOLO11?

Defining parking spaces is straightforward with Ultralytics YOLO11:

1. Capture a frame from a video or camera stream.
2. Use the provided code to launch a GUI for selecting an image and drawing polygons to define parking spaces.
3. Save the labeled data in JSON format for further processing. For comprehensive instructions, check the selection of points section above.

### Can I customize the YOLO11 model for specific parking management needs?

Yes, Ultralytics YOLO11 allows customization for specific parking management needs. You can adjust parameters such as the **occupied and available region colors**, margins for text display, and much more. Utilizing the `ParkingManagement` class's [arguments](#parkingmanagement-arguments), you can tailor the model to suit your particular requirements, ensuring maximum efficiency and effectiveness.

### What are some real-world applications of Ultralytics YOLO11 in parking lot management?

Ultralytics YOLO11 is utilized in various real-world applications for parking lot management, including:

- **Parking Space Detection**: Accurately identifying available and occupied spaces.
- **Surveillance**: Enhancing security through real-time monitoring.
- **Traffic Flow Management**: Reducing idle times and congestion with efficient traffic handling. Images showcasing these applications can be found in [real-world applications](#real-world-applications).
