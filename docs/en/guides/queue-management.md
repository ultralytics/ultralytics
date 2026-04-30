---
comments: true
description: Learn how to manage and optimize queues using Ultralytics YOLO26 to reduce wait times and increase efficiency in various real-world applications.
keywords: queue management, YOLO26, Ultralytics, reduce wait times, efficiency, customer satisfaction, retail, airports, healthcare, banks
---

# Queue Management using Ultralytics YOLO26 🚀

## What is Queue Management?

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-monitor-objects-in-queue-using-queue-management-solution.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Queue Management In Colab"></a>

Queue management using [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) involves organizing and controlling lines of people or vehicles to reduce wait times and enhance efficiency. It's about optimizing queues to improve customer satisfaction and system performance in various settings like retail, banks, airports, and healthcare facilities.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gxr9SpYPLh0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Build a Queue Management System with Ultralytics YOLO | Retail, Bank & Crowd Use Cases 🚀
</p>

## Advantages of Queue Management

- **Reduced Waiting Times:** Queue management systems efficiently organize queues, minimizing wait times for customers. This leads to improved satisfaction levels as customers spend less time waiting and more time engaging with products or services.
- **Increased Efficiency:** Implementing queue management allows businesses to allocate resources more effectively. By analyzing queue data and optimizing staff deployment, businesses can streamline operations, reduce costs, and improve overall productivity.
- **Real-time Insights:** YOLO26-powered queue management provides instant data on queue lengths and wait times, enabling managers to make informed decisions quickly.
- **Enhanced Customer Experience:** By reducing frustration associated with long waits, businesses can significantly improve customer satisfaction and loyalty.

## Real World Applications

|                                                                                            Logistics                                                                                             |                                                                             Retail                                                                             |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Queue management at airport ticket counter using Ultralytics YOLO26](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/queue-management-airport-ticket-counter-ultralytics-yolov8.avif) | ![Queue monitoring in crowd using Ultralytics YOLO26](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/queue-monitoring-crowd-ultralytics-yolov8.avif) |
|                                                               Queue management at airport ticket counter Using Ultralytics YOLO26                                                                |                                                          Queue monitoring in crowd Ultralytics YOLO26                                                          |

!!! example "Queue Management using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Run a queue example
        yolo solutions queue show=True

        # Pass a source video
        yolo solutions queue source="path/to/video.mp4"

        # Pass queue coordinates
        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Define queue points
        queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # region points
        # queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]    # polygon points

        # Initialize queue manager object
        queuemanager = solutions.QueueManager(
            show=True,  # display the output
            model="yolo26n.pt",  # path to the YOLO26 model file
            region=queue_region,  # pass queue region points
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break
            results = queuemanager(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `QueueManager` Arguments

Here's a table with the `QueueManager` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

The `QueueManagement` solution also support some `track` arguments:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the following visualization parameters are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## Implementation Strategies

When implementing queue management with YOLO26, consider these best practices:

1. **Strategic Camera Placement:** Position cameras to capture the entire queue area without obstructions.
2. **Define Appropriate Queue Regions:** Carefully set queue boundaries based on the physical layout of your space.
3. **Adjust Detection Confidence:** Fine-tune the confidence threshold based on lighting conditions and crowd density.
4. **Integrate with Existing Systems:** Connect your queue management solution with digital signage or staff notification systems for automated responses.

## FAQ

### How can I use Ultralytics YOLO26 for real-time queue management?

To use Ultralytics YOLO26 for real-time queue management, you can follow these steps:

1. Load the YOLO26 model with `YOLO("yolo26n.pt")`.
2. Capture the video feed using `cv2.VideoCapture`.
3. Define the region of interest (ROI) for queue management.
4. Process frames to detect objects and manage queues.

Here's a minimal example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

queuemanager = solutions.QueueManager(
    model="yolo26n.pt",
    region=queue_region,
    line_width=3,
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        results = queuemanager(im0)

cap.release()
cv2.destroyAllWindows()
```

Leveraging [Ultralytics Platform](https://docs.ultralytics.com/platform/) can streamline this process by providing a user-friendly platform for deploying and managing your queue management solution.

### What are the key advantages of using Ultralytics YOLO26 for queue management?

Using Ultralytics YOLO26 for queue management offers several benefits:

- **Plummeting Waiting Times:** Efficiently organizes queues, reducing customer wait times and boosting satisfaction.
- **Enhancing Efficiency:** Analyzes queue data to optimize staff deployment and operations, thereby reducing costs.
- **Real-time Alerts:** Provides real-time notifications for long queues, enabling quick intervention.
- **Scalability:** Easily scalable across different environments like retail, airports, and healthcare.

For more details, explore our [Queue Management](https://docs.ultralytics.com/reference/solutions/queue_management/) solutions.

### Why should I choose Ultralytics YOLO26 over competitors like [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) or Detectron2 for queue management?

Ultralytics YOLO26 has several advantages over TensorFlow and Detectron2 for queue management:

- **Real-time Performance:** YOLO26 is known for its real-time detection capabilities, offering faster processing speeds.
- **Ease of Use:** Ultralytics provides a user-friendly experience, from training to deployment, via [Ultralytics Platform](https://docs.ultralytics.com/platform/).
- **Pretrained Models:** Access to a range of pretrained models, minimizing the time needed for setup.
- **Community Support:** Extensive documentation and active community support make problem-solving easier.

Learn how to get started with [Ultralytics YOLO](https://docs.ultralytics.com/quickstart/).

### Can Ultralytics YOLO26 handle multiple types of queues, such as in airports and retail?

Yes, Ultralytics YOLO26 can manage various types of queues, including those in airports and retail environments. By configuring the QueueManager with specific regions and settings, YOLO26 can adapt to different queue layouts and densities.

Example for airports:

```python
queue_region_airport = [(50, 600), (1200, 600), (1200, 550), (50, 550)]
queue_airport = solutions.QueueManager(
    model="yolo26n.pt",
    region=queue_region_airport,
    line_width=3,
)
```

For more information on diverse applications, check out our [Real World Applications](#real-world-applications) section.

### What are some real-world applications of Ultralytics YOLO26 in queue management?

Ultralytics YOLO26 is used in various real-world applications for queue management:

- **Retail:** Monitors checkout lines to reduce wait times and improve customer satisfaction.
- **Airports:** Manages queues at ticket counters and security checkpoints for a smoother passenger experience.
- **Healthcare:** Optimizes patient flow in clinics and hospitals.
- **Banks:** Enhances customer service by managing queues efficiently in banks.

Check our [blog on real-world queue management](https://www.ultralytics.com/blog/a-look-at-real-time-queue-monitoring-enabled-by-computer-vision) to learn more about how computer vision is transforming queue monitoring across industries.
