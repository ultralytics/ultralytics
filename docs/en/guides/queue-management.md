---
comments: true
description: Learn how to manage and optimize queues using Ultralytics YOLO11 to reduce wait times and increase efficiency in various real-world applications.
keywords: queue management, YOLO11, Ultralytics, reduce wait times, efficiency, customer satisfaction, retail, airports, healthcare, banks
---

# Queue Management using Ultralytics YOLO11 🚀

## What is Queue Management?

Queue management using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) involves organizing and controlling lines of people or vehicles to reduce wait times and enhance efficiency. It's about optimizing queues to improve customer satisfaction and system performance in various settings like retail, banks, airports, and healthcare facilities.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/gX5kSRD56Gs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Implement Queue Management with Ultralytics YOLO11 | Airport and Metro Station
</p>

## Advantages of Queue Management?

- **Reduced Waiting Times:** Queue management systems efficiently organize queues, minimizing wait times for customers. This leads to improved satisfaction levels as customers spend less time waiting and more time engaging with products or services.
- **Increased Efficiency:** Implementing queue management allows businesses to allocate resources more effectively. By analyzing queue data and optimizing staff deployment, businesses can streamline operations, reduce costs, and improve overall productivity.

## Real World Applications

|                                                                                            Logistics                                                                                            |                                                                            Retail                                                                             |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Queue management at airport ticket counter using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/queue-management-airport-ticket-counter-ultralytics-yolov8.avif) | ![Queue monitoring in crowd using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/queue-monitoring-crowd-ultralytics-yolov8.avif) |
|                                                               Queue management at airport ticket counter Using Ultralytics YOLO11                                                               |                                                         Queue monitoring in crowd Ultralytics YOLO11                                                          |

!!! example "Queue Management using YOLO11 Example"

    === "CLI"

        ```bash
        # Run a queue example
        yolo solutions queue show=True

        # Pass a source video
        yolo solutions queue source="path/to/video/file.mp4"

        # Pass queue coordinates
        yolo solutions queue region=[(20, 400), (1080, 404), (1080, 360), (20, 360)]
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")

        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Define queue region points
        queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]  # Define queue region points
        # queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]  # Define queue polygon points

        # Init Queue Manager
        queue = solutions.QueueManager(
            show=True,  # Display the output
            model="yolo11n.pt",  # Path to the YOLO11 model file
            region=queue_region,  # Pass queue region points
            # classes=[0, 2],  # If you want to count specific classes i.e person and car with COCO pretrained model.
            # line_width=2,  # Adjust the line width for bounding boxes and text display
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if success:
                out = queue.process_queue(im0)
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

| Name         | Type   | Default                    | Description                                          |
| ------------ | ------ | -------------------------- | ---------------------------------------------------- |
| `model`      | `str`  | `None`                     | Path to Ultralytics YOLO Model File                  |
| `region`     | `list` | `[(20, 400), (1260, 400)]` | List of points defining the queue region.            |
| `line_width` | `int`  | `2`                        | Line thickness for bounding boxes.                   |
| `show`       | `bool` | `False`                    | Flag to control whether to display the video stream. |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## FAQ

### How can I use Ultralytics YOLO11 for real-time queue management?

To use Ultralytics YOLO11 for real-time queue management, you can follow these steps:

1. Load the YOLO11 model with `YOLO("yolo11n.pt")`.
2. Capture the video feed using `cv2.VideoCapture`.
3. Define the region of interest (ROI) for queue management.
4. Process frames to detect objects and manage queues.

Here's a minimal example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

queue = solutions.QueueManager(
    model="yolo11n.pt",
    region=queue_region,
    line_width=3,
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        out = queue.process_queue(im0)
        cv2.imshow("Queue Management", im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
```

Leveraging Ultralytics [HUB](https://docs.ultralytics.com/hub/) can streamline this process by providing a user-friendly platform for deploying and managing your queue management solution.

### What are the key advantages of using Ultralytics YOLO11 for queue management?

Using Ultralytics YOLO11 for queue management offers several benefits:

- **Plummeting Waiting Times:** Efficiently organizes queues, reducing customer wait times and boosting satisfaction.
- **Enhancing Efficiency:** Analyzes queue data to optimize staff deployment and operations, thereby reducing costs.
- **Real-time Alerts:** Provides real-time notifications for long queues, enabling quick intervention.
- **Scalability:** Easily scalable across different environments like retail, airports, and healthcare.

For more details, explore our [Queue Management](https://docs.ultralytics.com/reference/solutions/queue_management/) solutions.

### Why should I choose Ultralytics YOLO11 over competitors like [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) or Detectron2 for queue management?

Ultralytics YOLO11 has several advantages over TensorFlow and Detectron2 for queue management:

- **Real-time Performance:** YOLO11 is known for its real-time detection capabilities, offering faster processing speeds.
- **Ease of Use:** Ultralytics provides a user-friendly experience, from training to deployment, via [Ultralytics HUB](https://docs.ultralytics.com/hub/).
- **Pretrained Models:** Access to a range of pretrained models, minimizing the time needed for setup.
- **Community Support:** Extensive documentation and active community support make problem-solving easier.

Learn how to get started with [Ultralytics YOLO](https://docs.ultralytics.com/quickstart/).

### Can Ultralytics YOLO11 handle multiple types of queues, such as in airports and retail?

Yes, Ultralytics YOLO11 can manage various types of queues, including those in airports and retail environments. By configuring the QueueManager with specific regions and settings, YOLO11 can adapt to different queue layouts and densities.

Example for airports:

```python
queue_region_airport = [(50, 600), (1200, 600), (1200, 550), (50, 550)]
queue_airport = solutions.QueueManager(
    model="yolo11n.pt",
    region=queue_region_airport,
    line_width=3,
)
```

For more information on diverse applications, check out our [Real World Applications](#real-world-applications) section.

### What are some real-world applications of Ultralytics YOLO11 in queue management?

Ultralytics YOLO11 is used in various real-world applications for queue management:

- **Retail:** Monitors checkout lines to reduce wait times and improve customer satisfaction.
- **Airports:** Manages queues at ticket counters and security checkpoints for a smoother passenger experience.
- **Healthcare:** Optimizes patient flow in clinics and hospitals.
- **Banks:** Enhances customer service by managing queues efficiently in banks.

Check our [blog on real-world queue management](https://www.ultralytics.com/blog/revolutionizing-queue-management-with-ultralytics-yolov8-and-openvino) to learn more.
