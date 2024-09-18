---
comments: true
description: Learn how to manage and optimize queues using Ultralytics YOLOv8 to reduce wait times and increase efficiency in various real-world applications.
keywords: queue management, YOLOv8, Ultralytics, reduce wait times, efficiency, customer satisfaction, retail, airports, healthcare, banks
---

# Queue Management using Ultralytics YOLOv8 🚀

## What is Queue Management?

Queue management using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) involves organizing and controlling lines of people or vehicles to reduce wait times and enhance efficiency. It's about optimizing queues to improve customer satisfaction and system performance in various settings like retail, banks, airports, and healthcare facilities.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/gX5kSRD56Gs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Implement Queue Management with Ultralytics YOLOv8 | Airport and Metro Station
</p>

## Advantages of Queue Management?

- **Reduced Waiting Times:** Queue management systems efficiently organize queues, minimizing wait times for customers. This leads to improved satisfaction levels as customers spend less time waiting and more time engaging with products or services.
- **Increased Efficiency:** Implementing queue management allows businesses to allocate resources more effectively. By analyzing queue data and optimizing staff deployment, businesses can streamline operations, reduce costs, and improve overall productivity.

## Real World Applications

|                                                                                            Logistics                                                                                            |                                                                            Retail                                                                             |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Queue management at airport ticket counter using Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/queue-management-airport-ticket-counter-ultralytics-yolov8.avif) | ![Queue monitoring in crowd using Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/queue-monitoring-crowd-ultralytics-yolov8.avif) |
|                                                               Queue management at airport ticket counter Using Ultralytics YOLOv8                                                               |                                                         Queue monitoring in crowd Ultralytics YOLOv8                                                          |

!!! example "Queue Management using YOLOv8 Example"

    === "Queue Manager"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")

        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

        queue = solutions.QueueManager(
            names=model.names,
            reg_pts=queue_region,
            line_thickness=3,
        )

        while cap.isOpened():
            success, im0 = cap.read()

            if success:
                tracks = model.track(im0, persist=True)
                out = queue.process_queue(im0, tracks)

                video_writer.write(im0)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            print("Video frame is empty or video processing has been successfully completed.")
            break

        cap.release()
        cv2.destroyAllWindows()
        ```

    === "Queue Manager Specific Classes"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")

        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

        queue = solutions.QueueManager(
            names=model.names,
            reg_pts=queue_region,
            line_thickness=3,
        )

        while cap.isOpened():
            success, im0 = cap.read()

            if success:
                tracks = model.track(im0, persist=True, classes=0)  # Only person class
                out = queue.process_queue(im0, tracks)

                video_writer.write(im0)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            print("Video frame is empty or video processing has been successfully completed.")
            break

        cap.release()
        cv2.destroyAllWindows()
        ```

### Arguments `QueueManager`

| Name             | Type             | Default                    | Description                                                                      |
| ---------------- | ---------------- | -------------------------- | -------------------------------------------------------------------------------- |
| `names`          | `dict`           | `model.names`              | A dictionary mapping class IDs to class names.                                   |
| `reg_pts`        | `list of tuples` | `[(20, 400), (1260, 400)]` | Points defining the counting region polygon. Defaults to a predefined rectangle. |
| `line_thickness` | `int`            | `2`                        | Thickness of the annotation lines.                                               |
| `view_img`       | `bool`           | `False`                    | Whether to display the image frames.                                             |
| `draw_tracks`    | `bool`           | `False`                    | Whether to draw tracks of the objects.                                           |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## FAQ

### How can I use Ultralytics YOLOv8 for real-time queue management?

To use Ultralytics YOLOv8 for real-time queue management, you can follow these steps:

1. Load the YOLOv8 model with `YOLO("yolov8n.pt")`.
2. Capture the video feed using `cv2.VideoCapture`.
3. Define the region of interest (ROI) for queue management.
4. Process frames to detect objects and manage queues.

Here's a minimal example:

```python
import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("path/to/video.mp4")
queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

queue = solutions.QueueManager(
    names=model.names,
    reg_pts=queue_region,
    line_thickness=3,
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        tracks = model.track(im0, show=False, persist=True, verbose=False)
        out = queue.process_queue(im0, tracks)
        cv2.imshow("Queue Management", im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
```

Leveraging Ultralytics [HUB](https://docs.ultralytics.com/hub/) can streamline this process by providing a user-friendly platform for deploying and managing your queue management solution.

### What are the key advantages of using Ultralytics YOLOv8 for queue management?

Using Ultralytics YOLOv8 for queue management offers several benefits:

- **Plummeting Waiting Times:** Efficiently organizes queues, reducing customer wait times and boosting satisfaction.
- **Enhancing Efficiency:** Analyzes queue data to optimize staff deployment and operations, thereby reducing costs.
- **Real-time Alerts:** Provides real-time notifications for long queues, enabling quick intervention.
- **Scalability:** Easily scalable across different environments like retail, airports, and healthcare.

For more details, explore our [Queue Management](https://docs.ultralytics.com/reference/solutions/queue_management/) solutions.

### Why should I choose Ultralytics YOLOv8 over competitors like TensorFlow or Detectron2 for queue management?

Ultralytics YOLOv8 has several advantages over TensorFlow and Detectron2 for queue management:

- **Real-time Performance:** YOLOv8 is known for its real-time detection capabilities, offering faster processing speeds.
- **Ease of Use:** Ultralytics provides a user-friendly experience, from training to deployment, via [Ultralytics HUB](https://docs.ultralytics.com/hub/).
- **Pretrained Models:** Access to a range of pretrained models, minimizing the time needed for setup.
- **Community Support:** Extensive documentation and active community support make problem-solving easier.

Learn how to get started with [Ultralytics YOLO](https://docs.ultralytics.com/quickstart/).

### Can Ultralytics YOLOv8 handle multiple types of queues, such as in airports and retail?

Yes, Ultralytics YOLOv8 can manage various types of queues, including those in airports and retail environments. By configuring the QueueManager with specific regions and settings, YOLOv8 can adapt to different queue layouts and densities.

Example for airports:

```python
queue_region_airport = [(50, 600), (1200, 600), (1200, 550), (50, 550)]
queue_airport = solutions.QueueManager(
    names=model.names,
    reg_pts=queue_region_airport,
    line_thickness=3,
)
```

For more information on diverse applications, check out our [Real World Applications](#real-world-applications) section.

### What are some real-world applications of Ultralytics YOLOv8 in queue management?

Ultralytics YOLOv8 is used in various real-world applications for queue management:

- **Retail:** Monitors checkout lines to reduce wait times and improve customer satisfaction.
- **Airports:** Manages queues at ticket counters and security checkpoints for a smoother passenger experience.
- **Healthcare:** Optimizes patient flow in clinics and hospitals.
- **Banks:** Enhances customer service by managing queues efficiently in banks.

Check our [blog on real-world queue management](https://www.ultralytics.com/blog/revolutionizing-queue-management-with-ultralytics-yolov8-and-openvino) to learn more.
