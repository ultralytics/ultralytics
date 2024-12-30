---
comments: true
description: Transform complex data into insightful heatmaps using Ultralytics YOLO11. Calculate Presence time in zone, the movement from one zone to another zone.
keywords: Ultralytics, YOLO11, Dwell Time, Avrage Dwell Time, Funnel Stages, Counting, Zone Visualization,
---

# Advanced Dwell Time Analyzer: Count Time using Ultralytics YOLO11 🚀

## Introduction to Dwell Time Analyzer

Understanding user engagement and behavioral patterns with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) is critical for optimizing systems, workflows, or user interfaces. One of the key metrics for evaluating these patterns is dwell time, which refers to the duration a objects spends in a specific place, such as a racks, counter, or queue. etc

### Why Choose Dwell Time Analyzer for Data Analysis?

In today’s data-driven world, understanding how users interact with systems is essential for making informed decisions. The **Dwell Time Analyzer** stands out as a powerful tool for analyzing engagement and user behavior, offering several compelling advantages:

1. **Comprehensive Insights**:  
   The analyzer provides detailed metrics on how long users engage with specific elements, enabling a deeper understanding of user preferences and intent.

2. **Real-Time Monitoring**:  
   With real-time data tracking, the Dwell Time Analyzer allows immediate identification of engagement patterns, helping organizations react quickly to emerging trends.

3. **Enhanced Decision-Making**:  
   By identifying what captures user attention, businesses can make data-backed decisions to improve content, design, or workflows.

4. **Customizable Features**:  
   Tailored to meet diverse industry needs, the analyzer can be configured to monitor dwell time across various platforms and environments.

5. **Scalable for Any Application**:  
   Whether analyzing website interactions, retail environments, or digital marketing campaigns, the Dwell Time Analyzer adapts to projects of any scale.

6. **Improved User Experience**:  
   By highlighting areas where users spend more or less time, the tool helps in refining products and interfaces, enhancing overall satisfaction and engagement.

7. **Seamless Integration**:  
   The Dwell Time Analyzer integrates easily with existing analytics platforms, making it a hassle-free addition to your data analysis toolkit.

Choosing the Dwell Time Analyzer equips organizations with a precise and reliable means of uncovering actionable insights, empowering them to optimize performance and deliver exceptional user experiences.

!!! example "DwellTimeAnalyzer using Ultralytics YOLO11 Example"

    === "Python"

        ```python
        import cv2

        from ultralytics.solutions import DwellTimeAnalyzer

        # Open the input video
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"

        # Get video properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Prepare the output video writer
        video_writer = cv2.VideoWriter("dwell_time.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Define zones for dwell time analysis
        zones = {
            "Entrance": [(15, 600), (470, 600), (500, 400), (180, 400)],
            "Aisle": [(230, 300), (500, 300), (490, 200), (300, 200)],
        }


        analyzer = DwellTimeAnalyzer(
            show=True,  # If True, will attempt to display frames (requires UI support)
            model="yolo11n.pt",  # YOLO model path
            source="path/to/video/file.mp4",  # Video source path
            fps=fps,  # Frames per second for converting dwell time to seconds
            # zones=zones,           # Defined zones (optional) if not provided, zones are selected interactively)
            # classes=[],            # List of class indices to track (optional)
        )


        # Read frames and analyze
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or processing has completed.")
                break

            # Process the current frame
            im0 = analyzer.count(im0)

            # Write the annotated frame to the output file
            video_writer.write(im0)

        # Release resources
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### Arguments `DwellTimeAnalyzer()`

| Name               | Type    | Default      | Description                                                |
| ------------------ | ------- | ------------ | ---------------------------------------------------------- |
| `model`            | `str`   | `None`       | Path to Ultralytics YOLO Model File                        |
| `Source`           | `file`  | `None`       | Path to source file.                                       |
| `show`             | `bool`  | `True`       | Whether to display the image with the dwell time analysis. |
| `fps`              | `int`   | `None`       | FPS for counting dwell time.                               |
| `zones`            | `list`  | `{}`         | Regions of interst.                                        |
| `classes`          | `list`  | `None`       | Specific classes filter.                                   |
| `enable_funnel`    | `bool`  | `False`      | Enable funnel analysis.                                    |
| `funnel_stages`    | `tuple` | `None`       | Define funnel stages explicitly if regions are define.     |
| `enable_avg_dwell` | `bool`  | `False`      | Average dwell time computation.                            |
| `detect_mode`      | `str`   | `all_frames` | Choose "all_frames" or "enter_zones.                       |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## FAQ

### How does Ultralytics YOLO11 Calculate dwell time?

Ultralytics YOLO11, the latest advancement in the YOLO (You Only Look Once) series, is a state-of-the-art deep learning model tailored for real-time object detection and behavior analysis. It powers the Dwell Time Analyzer by leveraging its cutting-edge capabilities in detecting and tracking objects or users in dynamic environments.

### Can I use Ultralytics YOLO11 to perform object tracking and counting time for objects simultaneously?

Yes, Ultralytics YOLO11 can certainly be used to perform object tracking and calculate dwell time. `DwellTimeAnalyzer` solution integrated with object tracking models. To do so, you need to initialize the dwelltimeanalyzer object and use YOLO11's tracking capabilities. Here's a simple example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

analyzer = solutions.DwellTimeAnalyzer(show=True, model="yolo11n.pt", fps=fps, classes=[1, 2])

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    im0 = analyzer.count(im0)
    cv2.imshow("analyzer", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

For further guidance, check the [Tracking Mode](../modes/track.md) page.

### What makes Ultralytics YOLO11 Dwell Time Analyzer different from other Techniques?

Ultralytics YOLO11 dwell time analyzer is specifically designed for integration with its [object detection](https://www.ultralytics.com/glossary/object-detection) and tracking models, providing an end-to-end solution for real-time data analysis. This offers several distinct advantages that set it apart from traditional object detection and tracking techniques. It provides a highly accurate, scalable, and efficient solution for calculating dwell time and analyzing user or object behavior. For more information on YOLO11's unique features, visit the [Ultralytics YOLO11 Introduction](https://www.ultralytics.com/blog/introducing-ultralytics-yolov8).

### How can I Count Dwell Time only for specific object classes using Ultralytics YOLO11?

You can count dwell time for specific object classes by specifying the desired classes at the initialization of the `DwellTimeAnalyzer()` class of the YOLO model. For instance, if you only want to visualize cars and persons (assuming their class indices are 0 and 2), you can set the `classes` parameter accordingly.

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

analyzer = solutions.DwellTimeAnalyzer(show=True, model="yolo11n.pt", fps=fps)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    im0 = analyzer.count(im0)
    cv2.imshow("analyzer", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### Why should businesses choose Ultralytics YOLO11 for Dwell Time Analyzer in data analysis?

Ultralytics YOLO11 offers seamless integration of advanced object detection and real-time object presence time counting, making it an ideal choice for businesses looking to count time more effectively. The key advantages include accurate, actionable insights into user or object interactions, efficient pattern detection, and enhanced spatial analysis for better decision-making. Additionally, YOLO11's cutting-edge features such as persistent tracking, customizable colormaps, and support for various export formats make it superior to other tools like [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and OpenCV for comprehensive data analysis. Learn more about business applications at [Ultralytics Plans](https://www.ultralytics.com/plans).
