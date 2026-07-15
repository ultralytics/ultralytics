---
title: YOLO with ROS Quickstart
comments: true
description: Integrate Ultralytics YOLO with ROS1 (Noetic, rospy) and ROS2 (rclpy) to run object detection and segmentation on RGB images, depth images, and point clouds for robotic perception.
keywords: Ultralytics, YOLO, object detection, deep learning, machine learning, guide, ROS, ROS2, Robot Operating System, robotics, rclpy, rospy, ROS Noetic, Python, Ubuntu, simulation, visualization, communication, middleware, hardware abstraction, tools, utilities, ecosystem, Noetic Ninjemys, autonomous vehicle, AMV
---

# Use Ultralytics YOLO with ROS for Robot Perception

This guide shows you how to integrate [Ultralytics YOLO](../models/yolo26.md) with a robot running [ROS (Robot Operating System)](https://www.ros.org/), either ROS1 (Noetic, via `rospy`) or ROS2 (via `rclpy`), to run real-time [object detection](../tasks/detect.md) and [segmentation](../tasks/segment.md) on RGB images, depth images, and point clouds.

Jump to [setting up YOLO with ROS](#setting-up-ultralytics-yolo-with-ros), then work with [RGB images](#use-ultralytics-with-ros-sensor_msgsimage), [depth images](#use-ultralytics-with-ros-depth-images), or [point clouds](#use-ultralytics-with-ros-sensor_msgspointcloud2) — or start with the [background on ROS](#background-about-ros) if you're new to the framework.

<p align="center"> <iframe src="https://player.vimeo.com/video/639236696?h=740f412ce5" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe></p>
<p align="center"><a href="https://vimeo.com/639236696">ROS Introduction (captioned)</a> from <a href="https://vimeo.com/osrfoundation">Open Robotics</a> on <a href="https://vimeo.com/">Vimeo</a>.</p>

## Why Use Ultralytics YOLO with ROS?

Integrating YOLO into a ROS pipeline turns raw camera and point-cloud topics into structured detections, segmentation masks, and 3D object positions that other ROS nodes — navigation, manipulation, monitoring — can consume directly, without a custom perception stack. The same YOLO models work across all three sensor modalities covered in this guide (RGB images, depth images, point clouds), and every example below ships both a ROS1 (`rospy`) and a ROS2 (`rclpy`) tab. If you're new to ROS itself, see the [background section](#background-about-ros) at the end of this guide.

## Setting Up Ultralytics YOLO with ROS

This guide has been tested using [this ROS environment](https://github.com/ambitious-octopus/rosbot_ros/tree/noetic), which is a fork of the [ROSbot ROS repository](https://github.com/husarion/rosbot_ros). This environment includes the Ultralytics YOLO package, a Docker container for easy setup, comprehensive ROS packages, and Gazebo worlds for rapid testing. It is designed to work with the [Husarion ROSbot 2 PRO](https://husarion.com/manuals/rosbot/). The code examples provided will work in any ROS Noetic/Melodic environment, including both simulation and real-world.

<p align="center">
  <img width="50%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/husarion-rosbot-2-pro.avif" alt="Husarion ROSbot 2 PRO autonomous robot platform">
</p>

### Dependencies Installation

Apart from the ROS environment, you will need to install the following dependencies:

!!! example "Installing dependencies"

    === "ROS1"

        - **[ROS NumPy package](https://github.com/eric-wieser/ros_numpy)**: This is required for fast conversion between ROS Image messages and NumPy arrays. It ships as a ROS package rather than a PyPI package, so install it through the ROS package manager instead of `pip`:

            ```bash
            sudo apt install ros-noetic-ros-numpy
            ```

        - **Ultralytics package**:

            ```bash
            pip install ultralytics
            ```

    === "ROS2"

        - **[cv_bridge](https://github.com/ros-perception/vision_opencv)**: Converts between ROS `sensor_msgs/Image` messages and OpenCV/NumPy arrays. It ships with a standard ROS2 desktop install; if it's missing, install it through your distro's package manager (`sudo apt install ros-$ROS_DISTRO-cv-bridge` on Ubuntu).

        - **Ultralytics package**:

            ```bash
            pip install ultralytics
            ```

        Point cloud conversion uses `sensor_msgs_py`, which ships with the core ROS2 install, so no extra package is required for the [`sensor_msgs/PointCloud2`](#use-ultralytics-with-ros-sensor_msgspointcloud2) example below.

## Use Ultralytics with ROS `sensor_msgs/Image`

The `sensor_msgs/Image` [message type](https://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html) is commonly used in ROS for representing image data. It contains fields for encoding, height, width, and pixel data, making it suitable for transmitting images captured by cameras or other sensors. Image messages are widely used in robotic applications for tasks such as visual perception, [object detection](https://www.ultralytics.com/glossary/object-detection), and navigation.

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/detection-segmentation-ros-gazebo.avif" alt="Detection and Segmentation in ROS Gazebo">
</p>

### Image Step-by-Step Usage

The following code snippet demonstrates how to use the Ultralytics YOLO package with ROS. In this example, we subscribe to a camera topic, process the incoming image using YOLO, and publish the detected objects to new topics for [detection](../tasks/detect.md) and [segmentation](../tasks/segment.md). ROS1 (`rospy`) uses a linear script with a global node, while ROS2 (`rclpy`) wraps the same logic in a `Node` subclass.

First, import the necessary libraries and instantiate two models: one for [segmentation](../tasks/segment.md) and one for [detection](../tasks/detect.md).

!!! example "Imports and model setup"

    === "ROS1"

        Initialize a ROS node (with the name `ultralytics`) to enable communication with the ROS master. To ensure a stable connection, we include a brief pause, giving the node sufficient time to establish the connection before proceeding.

        ```python
        import time

        import rospy

        from ultralytics import YOLO

        detection_model = YOLO("yolo26m.pt")
        segmentation_model = YOLO("yolo26m-seg.pt")
        rospy.init_node("ultralytics")
        time.sleep(1)
        ```

    === "ROS2"

        A ROS2 node is a class that inherits from `rclpy.node.Node`; models are instantiated once in `__init__` and reused across every callback.

        ```python
        import cv_bridge
        from rclpy.node import Node

        from ultralytics import YOLO


        class UltralyticsNode(Node):
            """ROS2 node that runs Ultralytics YOLO detection and segmentation on incoming images."""

            def __init__(self):
                super().__init__("ultralytics")
                self.bridge = cv_bridge.CvBridge()
                self.detection_model = YOLO("yolo26m.pt")
                self.segmentation_model = YOLO("yolo26m-seg.pt")
        ```

Next, create two topics: one for [detection](../tasks/detect.md) and one for [segmentation](../tasks/segment.md). These topics will be used to publish the annotated images, making them accessible for further processing. The communication is facilitated using `sensor_msgs/Image` messages.

!!! example "Publishers"

    === "ROS1"

        ```python
        from sensor_msgs.msg import Image

        det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
        seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)
        ```

    === "ROS2"

        Publishers are created inside `__init__` with `self.create_publisher(msg_type, topic, queue_size)`, rather than as free-standing objects. Add `from sensor_msgs.msg import Image` to the imports at the top of the file.

        ```python
                self.det_image_pub = self.create_publisher(Image, "/ultralytics/detection/image", 5)
                self.seg_image_pub = self.create_publisher(Image, "/ultralytics/segmentation/image", 5)
        ```

Finally, subscribe to the `/camera/color/image_raw` topic and process every incoming frame with the previously instantiated YOLO models, then publish the annotated results back to the respective topics: `/ultralytics/detection/image` for detection and `/ultralytics/segmentation/image` for segmentation.

!!! example "Subscriber and callback"

    === "ROS1"

        This callback receives messages of type `sensor_msgs/Image`, converts them into a NumPy array using `ros_numpy`, processes the images with the previously instantiated YOLO models, annotates the images, and then publishes them back to the respective topics. Unlike `cv_bridge`, `ros_numpy.numpify()` has no `desired_encoding` argument — it preserves whatever encoding the camera driver publishes, so the callback normalizes an `rgb8` frame to `bgr8` before inference, matching the channel order YOLO expects and what `plot()` returns for the published annotated image.

        ```python
        import ros_numpy


        def callback(data):
            """Callback function to process image and publish annotated images."""
            array = ros_numpy.numpify(data)
            if data.encoding == "rgb8":
                array = array[..., ::-1]  # normalize to bgr8, the channel order YOLO expects
            if det_image_pub.get_num_connections():
                det_result = detection_model(array)
                det_annotated = det_result[0].plot(show=False)
                det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="bgr8"))

            if seg_image_pub.get_num_connections():
                seg_result = segmentation_model(array)
                seg_annotated = seg_result[0].plot(show=False)
                seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="bgr8"))


        rospy.Subscriber("/camera/color/image_raw", Image, callback)

        rospy.spin()
        ```

    === "ROS2"

        The callback converts the incoming `sensor_msgs/Image` message with `cv_bridge` instead of `ros_numpy`. `rclpy.spin(node)` replaces the `rospy.spin()` loop, and the node must be created and torn down explicitly in a `main()` entry point. Camera drivers typically publish with "best effort" reliability, so the subscription uses `qos_profile_sensor_data` instead of the default "reliable" queue depth. Add `import rclpy` and `from rclpy.qos import qos_profile_sensor_data` to the imports at the top of the file.

        `cv_bridge` is requested to decode into `bgr8`, matching the channel order YOLO's NumPy preprocessing expects; requesting `rgb8` here would feed the model a channel-swapped image.

        ```python
                self.create_subscription(Image, "/camera/color/image_raw", self.callback, qos_profile_sensor_data)

            def callback(self, data):
                """Callback function to process image and publish annotated images."""
                array = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
                if self.det_image_pub.get_subscription_count():
                    det_result = self.detection_model(array)
                    det_annotated = det_result[0].plot(show=False)
                    self.det_image_pub.publish(self.bridge.cv2_to_imgmsg(det_annotated, encoding="bgr8"))

                if self.seg_image_pub.get_subscription_count():
                    seg_result = self.segmentation_model(array)
                    seg_annotated = seg_result[0].plot(show=False)
                    self.seg_image_pub.publish(self.bridge.cv2_to_imgmsg(seg_annotated, encoding="bgr8"))


        def main(args=None):
            """Entry point for the ultralytics ROS2 node."""
            rclpy.init(args=args)
            node = UltralyticsNode()
            rclpy.spin(node)
            node.destroy_node()
            rclpy.shutdown()


        if __name__ == "__main__":
            main()
        ```

??? example "Complete code"

    === "ROS1"

        ```python
        import time

        import ros_numpy
        import rospy
        from sensor_msgs.msg import Image

        from ultralytics import YOLO

        detection_model = YOLO("yolo26m.pt")
        segmentation_model = YOLO("yolo26m-seg.pt")
        rospy.init_node("ultralytics")
        time.sleep(1)

        det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
        seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)


        def callback(data):
            """Callback function to process image and publish annotated images."""
            array = ros_numpy.numpify(data)
            if data.encoding == "rgb8":
                array = array[..., ::-1]  # normalize to bgr8, the channel order YOLO expects
            if det_image_pub.get_num_connections():
                det_result = detection_model(array)
                det_annotated = det_result[0].plot(show=False)
                det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="bgr8"))

            if seg_image_pub.get_num_connections():
                seg_result = segmentation_model(array)
                seg_annotated = seg_result[0].plot(show=False)
                seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="bgr8"))


        rospy.Subscriber("/camera/color/image_raw", Image, callback)

        rospy.spin()
        ```

    === "ROS2"

        ```python
        import cv_bridge
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        from ultralytics import YOLO


        class UltralyticsNode(Node):
            """ROS2 node that runs Ultralytics YOLO detection and segmentation on incoming images."""

            def __init__(self):
                super().__init__("ultralytics")
                self.bridge = cv_bridge.CvBridge()
                self.detection_model = YOLO("yolo26m.pt")
                self.segmentation_model = YOLO("yolo26m-seg.pt")
                self.det_image_pub = self.create_publisher(Image, "/ultralytics/detection/image", 5)
                self.seg_image_pub = self.create_publisher(Image, "/ultralytics/segmentation/image", 5)
                self.create_subscription(Image, "/camera/color/image_raw", self.callback, qos_profile_sensor_data)

            def callback(self, data):
                """Callback function to process image and publish annotated images."""
                array = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
                if self.det_image_pub.get_subscription_count():
                    det_result = self.detection_model(array)
                    det_annotated = det_result[0].plot(show=False)
                    self.det_image_pub.publish(self.bridge.cv2_to_imgmsg(det_annotated, encoding="bgr8"))

                if self.seg_image_pub.get_subscription_count():
                    seg_result = self.segmentation_model(array)
                    seg_annotated = seg_result[0].plot(show=False)
                    self.seg_image_pub.publish(self.bridge.cv2_to_imgmsg(seg_annotated, encoding="bgr8"))


        def main(args=None):
            """Entry point for the ultralytics ROS2 node."""
            rclpy.init(args=args)
            node = UltralyticsNode()
            rclpy.spin(node)
            node.destroy_node()
            rclpy.shutdown()


        if __name__ == "__main__":
            main()
        ```

!!! tip "Debugging"

    Debugging ROS (Robot Operating System) nodes can be challenging due to the system's distributed nature. Several tools can assist with this process:

    1. `rostopic echo <TOPIC-NAME>` : This command allows you to view messages published on a specific topic, helping you inspect the data flow.
    2. `rostopic list`: Use this command to list all available topics in the ROS system, giving you an overview of the active data streams.
    3. `rqt_graph`: This visualization tool displays the communication graph between nodes, providing insights into how nodes are interconnected and how they interact.
    4. For more complex visualizations, such as 3D representations, you can use [RViz](https://wiki.ros.org/rviz). RViz (ROS Visualization) is a powerful 3D visualization tool for ROS. It allows you to visualize the state of your robot and its environment in real-time. With RViz, you can view sensor data (e.g., `sensor_msgs/Image`), robot model states, and various other types of information, making it easier to debug and understand the behavior of your robotic system.

### Publish Detected Classes with `std_msgs/String`

Standard ROS messages also include `std_msgs/String` messages. In many applications, it is not necessary to republish the entire annotated image; instead, only the classes present in the robot's view are needed. The following example demonstrates how to use `std_msgs/String` [messages](https://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html) to republish the detected classes on the `/ultralytics/detection/classes` topic. These messages are more lightweight and provide essential information, making them valuable for various applications.

#### Example Use Case

Consider a warehouse robot equipped with a camera and object [detection model](../tasks/detect.md). Instead of sending large annotated images over the network, the robot can publish a list of detected classes as `std_msgs/String` messages. For instance, when the robot detects objects like "box", "pallet" and "forklift" it publishes these classes to the `/ultralytics/detection/classes` topic. This information can then be used by a central monitoring system to track the inventory in real-time, optimize the robot's path planning to avoid obstacles, or trigger specific actions such as picking up a detected box. This approach reduces the bandwidth required for communication and focuses on transmitting critical data. To detect warehouse-specific classes like these, train a custom YOLO model — for example with [Ultralytics Platform](https://platform.ultralytics.com) handling dataset management and cloud training — then swap the resulting weights in place of `yolo26m.pt` above.

### String Step-by-Step Usage

This example demonstrates how to use the Ultralytics YOLO package with ROS. In this example, we subscribe to a camera topic, process the incoming image using YOLO, and publish the detected objects to new topic `/ultralytics/detection/classes` using `std_msgs/String` messages. The `ros_numpy` package is used to convert the ROS Image message to a NumPy array for processing with YOLO.

```python
import time

import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ultralytics import YOLO

detection_model = YOLO("yolo26m.pt")
rospy.init_node("ultralytics")
time.sleep(1)
classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)


def callback(data):
    """Callback function to process image and publish detected classes."""
    array = ros_numpy.numpify(data)
    if data.encoding == "rgb8":
        array = array[..., ::-1]  # normalize to bgr8, the channel order YOLO expects
    if classes_pub.get_num_connections():
        det_result = detection_model(array)
        classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
        names = [det_result[0].names[i] for i in classes]
        classes_pub.publish(String(data=str(names)))


rospy.Subscriber("/camera/color/image_raw", Image, callback)
rospy.spin()
```

## Use Ultralytics with ROS Depth Images

In addition to RGB images, ROS supports [depth images](https://en.wikipedia.org/wiki/Depth_map), which provide information about the distance of objects from the camera. Depth images are crucial for robotic applications such as obstacle avoidance, 3D mapping, and localization.

A depth image is an image where each pixel represents the distance from the camera to an object. Unlike RGB images that capture color, depth images capture spatial information, enabling robots to perceive the 3D structure of their environment.

!!! tip "Obtaining Depth Images"

    Depth images can be obtained using various sensors:

    1. [Stereo Cameras](https://en.wikipedia.org/wiki/Stereo_camera): Use two cameras to calculate depth based on image disparity.
    2. [Time-of-Flight (ToF) Cameras](https://en.wikipedia.org/wiki/Time-of-flight_camera): Measure the time light takes to return from an object.
    3. [Structured Light Sensors](https://en.wikipedia.org/wiki/Structured-light_3D_scanner): Project a pattern and measure its deformation on surfaces.

### Using YOLO with Depth Images

In ROS, depth images are represented by the `sensor_msgs/Image` message type, which includes fields for encoding, height, width, and pixel data. The encoding field for depth images often uses a format like "16UC1", indicating a 16-bit unsigned integer per pixel, where each value represents the distance to the object. Depth images are commonly used in conjunction with RGB images to provide a more comprehensive view of the environment.

Using YOLO, it is possible to extract and combine information from both RGB and depth images. For instance, YOLO can detect objects within an RGB image, and this detection can be used to pinpoint corresponding regions in the depth image. This allows for the extraction of precise depth information for detected objects, enhancing the robot's ability to understand its environment in three dimensions.

!!! warning "RGB-D Cameras"

    When working with depth images, it is essential to ensure that the RGB and depth images are correctly aligned. RGB-D cameras, such as the [Intel RealSense](https://www.realsenseai.com/) series, provide synchronized RGB and depth images, making it easier to combine information from both sources. If using separate RGB and depth cameras, it is crucial to calibrate them to ensure accurate alignment.

    The examples below subscribe to `/camera/aligned_depth_to_color/image_raw`, a depth stream already registered to the color frame (on RealSense, enable it with the `align_depth.enable:=true` launch argument) — not the raw `/camera/depth/image_rect_raw` stream. A raw depth topic can differ from the color image in both resolution and viewpoint, so a mask computed from the color image would index the wrong depth pixels even when the array shapes happen to match.

    In the ROS1 tab below, `rospy.wait_for_message()` also has no timeout by default — if the named topic stops publishing, the callback blocks indefinitely. Pass a `timeout` argument (e.g., `rospy.wait_for_message(topic, Image, timeout=1.0)`) and handle the resulting `rospy.ROSException` to fail fast instead.

    Depth pixel values are only in meters when the topic publishes `32FC1`. A `16UC1` topic (common on RealSense and similar drivers) reports millimeters as integers, with `0` marking invalid pixels instead of `NaN` — the code below converts `16UC1` to meters and remaps `0` to `NaN` before averaging.

#### Depth Step-by-Step Usage

In this example, we use YOLO to segment an image and apply the extracted mask to segment the object in the depth image. This allows us to determine the distance of each pixel of the object of interest from the camera's focal center. By obtaining this distance information, we can calculate the distance between the camera and the specific object in the scene. Begin by importing the necessary libraries, creating a ROS node, and instantiating a segmentation model and a ROS topic.

!!! example "Node setup"

    === "ROS1"

        ```python
        import time

        import rospy
        from std_msgs.msg import String

        from ultralytics import YOLO

        rospy.init_node("ultralytics")
        time.sleep(1)

        segmentation_model = YOLO("yolo26m-seg.pt")

        classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
        ```

    === "ROS2"

        ```python
        import cv_bridge
        from rclpy.node import Node
        from std_msgs.msg import String

        from ultralytics import YOLO


        class UltralyticsNode(Node):
            """ROS2 node that pairs depth images with the latest RGB frame to estimate object distance."""

            def __init__(self):
                super().__init__("ultralytics")
                self.bridge = cv_bridge.CvBridge()
                self.segmentation_model = YOLO("yolo26m-seg.pt")
                self.latest_image = None
                self.classes_pub = self.create_publisher(String, "/ultralytics/detection/distance", 5)
        ```

Next, define the callbacks that process the incoming RGB and depth messages. Sensors report out-of-range pixels as either `NaN` or `inf` depending on the driver; the code filters both out with `np.isfinite` before averaging, and falls back to `np.inf` — rather than a misleading `0` — if every pixel under the mask turns out to be invalid. Pass `retina_masks=True` to the model so the returned mask matches the color image's full resolution instead of the smaller, letterboxed size used for inference; because the depth topic is aligned to the color frame (see the warning above), this makes the mask line up with `depth` pixel-for-pixel. Without `retina_masks=True`, indexing `depth` with the mask raises a shape mismatch on most camera resolutions.

!!! example "Callbacks"

    === "ROS1"

        The callback waits for the next available RGB image message — `rospy.wait_for_message()` returns whatever frame arrives next on the topic, not one matched to the depth frame's timestamp, so this is an approximate pairing like the ROS2 flow below (for hard timestamp-based synchronization use `message_filters.ApproximateTimeSynchronizer`). It converts both images into NumPy arrays and applies the segmentation model to the RGB image, then extracts the segmentation mask for each detected object and calculates the average distance of the object from the camera using the depth image, finally publishing the detected objects along with their average distances.

        ```python
        import numpy as np
        import ros_numpy
        from sensor_msgs.msg import Image


        def callback(data):
            """Callback function to process depth image and RGB image."""
            try:
                image_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=1.0)
            except rospy.ROSException:
                return
            image = ros_numpy.numpify(image_msg)
            if image_msg.encoding == "rgb8":
                image = image[..., ::-1]  # normalize to bgr8, the channel order YOLO expects
            depth = ros_numpy.numpify(data)
            if data.encoding == "16UC1":  # millimeters with 0 = invalid; convert to meters with NaN = invalid
                depth = np.where(depth == 0, np.nan, depth.astype(np.float32) / 1000)
            result = segmentation_model(image, retina_masks=True)

            all_objects = []
            for index, cls in enumerate(result[0].boxes.cls):
                class_index = int(cls.cpu().numpy())
                name = result[0].names[class_index]
                mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
                obj = depth[mask == 1]
                obj = obj[np.isfinite(obj)]
                avg_distance = np.mean(obj) if len(obj) else np.inf
                all_objects.append(f"{name}: {avg_distance:.2f}m")

            classes_pub.publish(String(data=str(all_objects)))


        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, callback)

        rospy.spin()
        ```

    === "ROS2"

        Instead of blocking on a matching RGB message, the ROS2 node subscribes to both topics independently and caches the latest RGB frame; the depth callback reuses whatever frame is most recently available. This is an approximate pairing, adequate for a quickstart — for hard timestamp-based synchronization use [`message_filters.ApproximateTimeSynchronizer`](https://github.com/ros2/message_filters) instead. Add `import numpy as np`, `import rclpy`, `from sensor_msgs.msg import Image`, and `from rclpy.qos import qos_profile_sensor_data` to the imports at the top of the file.

        ```python
                self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, qos_profile_sensor_data)
                self.create_subscription(
                    Image, "/camera/aligned_depth_to_color/image_raw", self.depth_callback, qos_profile_sensor_data
                )

            def image_callback(self, data):
                """Cache the latest RGB frame for pairing with the next depth callback."""
                self.latest_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

            def depth_callback(self, data):
                """Callback function to process depth image using the latest cached RGB image."""
                if self.latest_image is None:
                    return
                depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                if data.encoding == "16UC1":  # millimeters with 0 = invalid; convert to meters with NaN = invalid
                    depth = np.where(depth == 0, np.nan, depth.astype(np.float32) / 1000)
                result = self.segmentation_model(self.latest_image, retina_masks=True)

                all_objects = []
                for index, cls in enumerate(result[0].boxes.cls):
                    class_index = int(cls.cpu().numpy())
                    name = result[0].names[class_index]
                    mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
                    obj = depth[mask == 1]
                    obj = obj[np.isfinite(obj)]
                    avg_distance = np.mean(obj) if len(obj) else np.inf
                    all_objects.append(f"{name}: {avg_distance:.2f}m")

                self.classes_pub.publish(String(data=str(all_objects)))


        def main(args=None):
            """Entry point for the ultralytics ROS2 node."""
            rclpy.init(args=args)
            node = UltralyticsNode()
            rclpy.spin(node)
            node.destroy_node()
            rclpy.shutdown()


        if __name__ == "__main__":
            main()
        ```

??? example "Complete code"

    === "ROS1"

        ```python
        import time

        import numpy as np
        import ros_numpy
        import rospy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        from ultralytics import YOLO

        rospy.init_node("ultralytics")
        time.sleep(1)

        segmentation_model = YOLO("yolo26m-seg.pt")

        classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)


        def callback(data):
            """Callback function to process depth image and RGB image."""
            try:
                image_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=1.0)
            except rospy.ROSException:
                return
            image = ros_numpy.numpify(image_msg)
            if image_msg.encoding == "rgb8":
                image = image[..., ::-1]  # normalize to bgr8, the channel order YOLO expects
            depth = ros_numpy.numpify(data)
            if data.encoding == "16UC1":  # millimeters with 0 = invalid; convert to meters with NaN = invalid
                depth = np.where(depth == 0, np.nan, depth.astype(np.float32) / 1000)
            result = segmentation_model(image, retina_masks=True)

            all_objects = []
            for index, cls in enumerate(result[0].boxes.cls):
                class_index = int(cls.cpu().numpy())
                name = result[0].names[class_index]
                mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
                obj = depth[mask == 1]
                obj = obj[np.isfinite(obj)]
                avg_distance = np.mean(obj) if len(obj) else np.inf
                all_objects.append(f"{name}: {avg_distance:.2f}m")

            classes_pub.publish(String(data=str(all_objects)))


        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, callback)

        rospy.spin()
        ```

    === "ROS2"

        ```python
        import cv_bridge
        import numpy as np
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        from ultralytics import YOLO


        class UltralyticsNode(Node):
            """ROS2 node that pairs depth images with the latest RGB frame to estimate object distance."""

            def __init__(self):
                super().__init__("ultralytics")
                self.bridge = cv_bridge.CvBridge()
                self.segmentation_model = YOLO("yolo26m-seg.pt")
                self.latest_image = None
                self.classes_pub = self.create_publisher(String, "/ultralytics/detection/distance", 5)
                self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, qos_profile_sensor_data)
                self.create_subscription(
                    Image, "/camera/aligned_depth_to_color/image_raw", self.depth_callback, qos_profile_sensor_data
                )

            def image_callback(self, data):
                """Cache the latest RGB frame for pairing with the next depth callback."""
                self.latest_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

            def depth_callback(self, data):
                """Callback function to process depth image using the latest cached RGB image."""
                if self.latest_image is None:
                    return
                depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                if data.encoding == "16UC1":  # millimeters with 0 = invalid; convert to meters with NaN = invalid
                    depth = np.where(depth == 0, np.nan, depth.astype(np.float32) / 1000)
                result = self.segmentation_model(self.latest_image, retina_masks=True)

                all_objects = []
                for index, cls in enumerate(result[0].boxes.cls):
                    class_index = int(cls.cpu().numpy())
                    name = result[0].names[class_index]
                    mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
                    obj = depth[mask == 1]
                    obj = obj[np.isfinite(obj)]
                    avg_distance = np.mean(obj) if len(obj) else np.inf
                    all_objects.append(f"{name}: {avg_distance:.2f}m")

                self.classes_pub.publish(String(data=str(all_objects)))


        def main(args=None):
            """Entry point for the ultralytics ROS2 node."""
            rclpy.init(args=args)
            node = UltralyticsNode()
            rclpy.spin(node)
            node.destroy_node()
            rclpy.shutdown()


        if __name__ == "__main__":
            main()
        ```

## Use Ultralytics with ROS `sensor_msgs/PointCloud2`

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/detection-segmentation-ros-gazebo-1.avif" alt="Detection and Segmentation in ROS Gazebo">
</p>

The `sensor_msgs/PointCloud2` [message type](https://docs.ros.org/en/api/sensor_msgs/html/msg/PointCloud2.html) is a data structure used in ROS to represent 3D point cloud data. This message type is integral to robotic applications, enabling tasks such as 3D mapping, object recognition, and localization.

A point cloud is a collection of data points defined within a three-dimensional coordinate system. These data points represent the external surface of an object or a scene, captured via 3D scanning technologies. Each point in the cloud has `X`, `Y`, and `Z` coordinates, which correspond to its position in space, and may also include additional information such as color and intensity.

!!! warning "Reference frame"

    When working with `sensor_msgs/PointCloud2`, it's essential to consider the reference frame of the sensor from which the point cloud data was acquired. The point cloud is initially captured in the sensor's reference frame. You can determine this reference frame by listening to the `/tf_static` topic. However, depending on your specific application requirements, you might need to convert the point cloud into another reference frame. This transformation can be achieved using the `tf2_ros` package, which provides tools for managing coordinate frames and transforming data between them.

!!! tip "Obtaining Point clouds"

    Point Clouds can be obtained using various sensors:

    1. **LIDAR (Light Detection and Ranging)**: Uses laser pulses to measure distances to objects and create high-[precision](https://www.ultralytics.com/glossary/precision) 3D maps.
    2. **Depth Cameras**: Capture depth information for each pixel, allowing for 3D reconstruction of the scene.
    3. **Stereo Cameras**: Utilize two or more cameras to obtain depth information through triangulation.
    4. **Structured Light Scanners**: Project a known pattern onto a surface and measure the deformation to calculate depth.

### Using YOLO with Point Clouds

To integrate YOLO with `sensor_msgs/PointCloud2` type messages, extract a 2D image from the color information embedded in the point cloud, perform segmentation on this image using YOLO, and then apply the resulting mask to the three-dimensional points to isolate the 3D object of interest. This workflow needs an organized, colorized point cloud — the `height x width` grid with a packed `rgb` field that RGB-D cameras like the Intel RealSense publish. A raw LIDAR cloud is normally unorganized (`height == 1`) and reports intensity instead of color, so it needs a separate projection and colorization step before it fits this pipeline.

For handling point clouds, we recommend using Open3D, a user-friendly Python library that provides robust tools for managing point cloud data structures, visualizing them, and executing complex operations seamlessly. This library can significantly simplify the process and enhance our ability to manipulate and analyze point clouds in conjunction with YOLO-based segmentation.

```bash
pip install open3d
```

#### Point Clouds Step-by-Step Usage

Import the necessary libraries and instantiate the YOLO model for segmentation.

!!! example "Node setup"

    === "ROS1"

        ```python
        import time

        import rospy

        from ultralytics import YOLO

        rospy.init_node("ultralytics")
        time.sleep(1)
        segmentation_model = YOLO("yolo26m-seg.pt")
        ```

    === "ROS2"

        This example is a one-shot script that waits for a single point cloud, so the subscriber callback just stores the message on the node for the main function to pick up.

        ```python
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2

        from ultralytics import YOLO


        class UltralyticsNode(Node):
            """ROS2 node that segments a single incoming PointCloud2 message with YOLO."""

            def __init__(self):
                super().__init__("ultralytics")
                self.segmentation_model = YOLO("yolo26m-seg.pt")
                self.cloud = None
                self.create_subscription(PointCloud2, "/camera/depth/points", self.callback, qos_profile_sensor_data)

            def callback(self, data):
                """Store the incoming point cloud for the main loop to process."""
                self.cloud = data
        ```

Create a function `pointcloud2_to_array`, which transforms a `sensor_msgs/PointCloud2` message into two NumPy arrays. The `sensor_msgs/PointCloud2` messages contain `n` points based on the `width` and `height` of the acquired image. For instance, a `480 x 640` image will have `307,200` points. Each point includes three spatial coordinates (`xyz`) and the corresponding color in `RGB` format. These can be considered as two separate channels of information.

The function returns the `xyz` coordinates and `RGB` values in the format of the original camera resolution (`width x height`). Most sensors report out-of-range points as `NaN`; the function zeroes out both the coordinates and color of these invalid points so they don't distort downstream processing.

!!! example "Point cloud conversion"

    === "ROS1"

        ```python
        import numpy as np
        import ros_numpy
        from sensor_msgs.msg import PointCloud2


        def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
            """Convert a ROS PointCloud2 message to a numpy array.

            Args:
                pointcloud2 (PointCloud2): the PointCloud2 message

            Returns:
                (tuple): tuple containing (xyz, rgb)
            """
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
            split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
            rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
            xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
            xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
            nan_rows = np.isnan(xyz).all(axis=2)
            xyz[nan_rows] = [0, 0, 0]
            rgb[nan_rows] = [0, 0, 0]
            return xyz, rgb
        ```

    === "ROS2"

        ROS2 ships `sensor_msgs_py.point_cloud2` as part of the core install, so no third-party point cloud package is required. The packed `rgb` field is a single float32 that encodes three bytes; unpack it by viewing it as `uint32` and bit-shifting.

        ```python
        import numpy as np
        from sensor_msgs_py import point_cloud2


        def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
            """Convert a ROS2 PointCloud2 message to a numpy array.

            Args:
                pointcloud2 (PointCloud2): the PointCloud2 message

            Returns:
                (tuple): tuple containing (xyz, rgb)
            """
            points = point_cloud2.read_points_numpy(pointcloud2, field_names=("x", "y", "z", "rgb"))
            xyz = points[:, :3].reshape((pointcloud2.height, pointcloud2.width, 3))
            packed = points[:, 3].copy().view(np.uint32)
            b, g, r = packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF
            rgb = np.stack([b, g, r], axis=1).reshape((pointcloud2.height, pointcloud2.width, 3)).astype(np.uint8)
            nan_rows = np.isnan(xyz).all(axis=2)
            xyz[nan_rows] = [0, 0, 0]
            rgb[nan_rows] = [0, 0, 0]
            return xyz, rgb
        ```

Next, wait for a point cloud message and convert it into NumPy arrays containing the XYZ coordinates and RGB values (using the `pointcloud2_to_array` function). Process the RGB image using the YOLO model to extract segmented objects, passing `retina_masks=True` so each mask matches the cloud's native resolution rather than the smaller size used for inference. For each detected object, extract the segmentation mask and apply it to both the RGB image and the XYZ coordinates to isolate the object in 3D space.

The mask is a binary array, with `1` indicating the presence of the object and `0` indicating the absence. Boolean-index `xyz` and `rgb` with `mask == 1` to keep only the object's points — multiplying by the mask instead would leave every background point in the cloud at coordinate `(0, 0, 0)`, rendering as a dense artificial cluster at the origin rather than an isolated object. Finally, create an Open3D point cloud object and visualize the segmented object in 3D space with associated colors — `pointcloud2_to_array` packs `rgb` in the BGR channel order YOLO expects, so reverse the channel axis when assigning `pcd.colors`, since Open3D expects colors in RGB order.

!!! example "Segment and visualize"

    === "ROS1"

        ```python
        import sys

        import open3d as o3d

        ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
        xyz, rgb = pointcloud2_to_array(ros_cloud)
        result = segmentation_model(rgb, retina_masks=True)

        if not len(result[0].boxes.cls):
            print("No objects detected")
            sys.exit()

        classes = result[0].boxes.cls.cpu().numpy().astype(int)
        for index, class_id in enumerate(classes):
            mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz[mask == 1])
            pcd.colors = o3d.utility.Vector3dVector(rgb[mask == 1][:, ::-1] / 255)
            o3d.visualization.draw_geometries([pcd])
        ```

    === "ROS2"

        The node spins with `rclpy.spin_once` until the subscriber callback stores a message, then the rest of the pipeline is identical to ROS1. Add `import rclpy` to the imports at the top of the file.

        ```python
        import sys

        import open3d as o3d


        def main(args=None):
            """Entry point for the ultralytics ROS2 node."""
            rclpy.init(args=args)
            node = UltralyticsNode()
            while node.cloud is None:
                rclpy.spin_once(node)
            ros_cloud = node.cloud

            xyz, rgb = pointcloud2_to_array(ros_cloud)
            result = node.segmentation_model(rgb, retina_masks=True)

            if not len(result[0].boxes.cls):
                print("No objects detected")
                node.destroy_node()
                rclpy.shutdown()
                sys.exit()

            classes = result[0].boxes.cls.cpu().numpy().astype(int)
            for index, class_id in enumerate(classes):
                mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz[mask == 1])
                pcd.colors = o3d.utility.Vector3dVector(rgb[mask == 1][:, ::-1] / 255)
                o3d.visualization.draw_geometries([pcd])

            node.destroy_node()
            rclpy.shutdown()


        if __name__ == "__main__":
            main()
        ```

??? example "Complete code"

    === "ROS1"

        ```python
        import sys
        import time

        import numpy as np
        import open3d as o3d
        import ros_numpy
        import rospy
        from sensor_msgs.msg import PointCloud2

        from ultralytics import YOLO

        rospy.init_node("ultralytics")
        time.sleep(1)
        segmentation_model = YOLO("yolo26m-seg.pt")


        def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
            """Convert a ROS PointCloud2 message to a numpy array.

            Args:
                pointcloud2 (PointCloud2): the PointCloud2 message

            Returns:
                (tuple): tuple containing (xyz, rgb)
            """
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
            split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
            rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
            xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
            xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
            nan_rows = np.isnan(xyz).all(axis=2)
            xyz[nan_rows] = [0, 0, 0]
            rgb[nan_rows] = [0, 0, 0]
            return xyz, rgb


        ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
        xyz, rgb = pointcloud2_to_array(ros_cloud)
        result = segmentation_model(rgb, retina_masks=True)

        if not len(result[0].boxes.cls):
            print("No objects detected")
            sys.exit()

        classes = result[0].boxes.cls.cpu().numpy().astype(int)
        for index, class_id in enumerate(classes):
            mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz[mask == 1])
            pcd.colors = o3d.utility.Vector3dVector(rgb[mask == 1][:, ::-1] / 255)
            o3d.visualization.draw_geometries([pcd])
        ```

    === "ROS2"

        ```python
        import sys

        import numpy as np
        import open3d as o3d
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py import point_cloud2

        from ultralytics import YOLO


        class UltralyticsNode(Node):
            """ROS2 node that segments a single incoming PointCloud2 message with YOLO."""

            def __init__(self):
                super().__init__("ultralytics")
                self.segmentation_model = YOLO("yolo26m-seg.pt")
                self.cloud = None
                self.create_subscription(PointCloud2, "/camera/depth/points", self.callback, qos_profile_sensor_data)

            def callback(self, data):
                """Store the incoming point cloud for the main loop to process."""
                self.cloud = data


        def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
            """Convert a ROS2 PointCloud2 message to a numpy array.

            Args:
                pointcloud2 (PointCloud2): the PointCloud2 message

            Returns:
                (tuple): tuple containing (xyz, rgb)
            """
            points = point_cloud2.read_points_numpy(pointcloud2, field_names=("x", "y", "z", "rgb"))
            xyz = points[:, :3].reshape((pointcloud2.height, pointcloud2.width, 3))
            packed = points[:, 3].copy().view(np.uint32)
            b, g, r = packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF
            rgb = np.stack([b, g, r], axis=1).reshape((pointcloud2.height, pointcloud2.width, 3)).astype(np.uint8)
            nan_rows = np.isnan(xyz).all(axis=2)
            xyz[nan_rows] = [0, 0, 0]
            rgb[nan_rows] = [0, 0, 0]
            return xyz, rgb


        def main(args=None):
            """Entry point for the ultralytics ROS2 node."""
            rclpy.init(args=args)
            node = UltralyticsNode()
            while node.cloud is None:
                rclpy.spin_once(node)
            ros_cloud = node.cloud

            xyz, rgb = pointcloud2_to_array(ros_cloud)
            result = node.segmentation_model(rgb, retina_masks=True)

            if not len(result[0].boxes.cls):
                print("No objects detected")
                node.destroy_node()
                rclpy.shutdown()
                sys.exit()

            classes = result[0].boxes.cls.cpu().numpy().astype(int)
            for index, class_id in enumerate(classes):
                mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz[mask == 1])
                pcd.colors = o3d.utility.Vector3dVector(rgb[mask == 1][:, ::-1] / 255)
                o3d.visualization.draw_geometries([pcd])

            node.destroy_node()
            rclpy.shutdown()


        if __name__ == "__main__":
            main()
        ```

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/point-cloud-segmentation-ultralytics.avif" alt="Point Cloud Segmentation with Ultralytics ">
</p>

## Background: About ROS

The [Robot Operating System (ROS)](https://www.ros.org/) is an open-source framework widely used in robotics research and industry. ROS provides a collection of [libraries and tools](https://www.ros.org/blog/ecosystem/) to help developers create robot applications, and is designed to work with various [robotic platforms](https://robots.ros.org/).

| Feature                      | Description                                                                                                                                                                                                                                                                                                       |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Modular Architecture**     | Complex systems are built by combining smaller, reusable components called [nodes](https://wiki.ros.org/ROS/Tutorials/UnderstandingNodes), which communicate over [topics](https://wiki.ros.org/ROS/Tutorials/UnderstandingTopics) or [services](https://wiki.ros.org/ROS/Tutorials/UnderstandingServicesParams). |
| **Communication Middleware** | A publish-subscribe model handles data streams (topics) and a request-reply model handles service calls, supporting inter-process communication and distributed computing.                                                                                                                                        |
| **Hardware Abstraction**     | A layer of abstraction over the hardware lets the same code run across different hardware setups.                                                                                                                                                                                                                 |
| **Tools and Utilities**      | RViz visualizes sensor data and robot state; Gazebo provides a simulation environment for testing algorithms and robot designs.                                                                                                                                                                                   |
| **Extensive Ecosystem**      | Community-maintained packages cover navigation, manipulation, perception, and more.                                                                                                                                                                                                                               |

Since its development in 2007, ROS has evolved through [multiple versions](https://wiki.ros.org/Distributions), split into two main series: ROS 1 and ROS 2. This guide covers both — the ROS1 tabs target the Long Term Support release ROS Noetic Ninjemys (the code should also work with earlier ROS 1 versions), and the ROS2 tabs target current LTS releases such as Humble and Jazzy.

| Aspect                     | ROS 1                                       | ROS 2                                                              |
| -------------------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| **Real-time Performance**  | Limited support                             | Improved support for real-time systems and deterministic behavior  |
| **Security**               | Minimal built-in security                   | Enhanced security features for safe, reliable operation            |
| **Scalability**            | Single ROS Master limits multi-robot setups | Better support for multi-robot systems and large-scale deployments |
| **Cross-platform Support** | Primarily Linux                             | Expanded support for Linux, Windows, and macOS                     |
| **Communication**          | Custom TCPROS/UDPROS middleware             | DDS for more flexible and efficient inter-process communication    |

Communication between nodes is built on [messages](https://wiki.ros.org/Messages) and [topics](https://wiki.ros.org/Topics): a message defines the data exchanged between nodes, and a topic is the named channel over which messages are published and subscribed, enabling asynchronous, decoupled communication. Each sensor or actuator in a robotic system typically publishes to a topic that other nodes consume for processing or control. This guide focuses on Image, Depth, and PointCloud2 messages carried over camera topics.

## Conclusion

With Ultralytics YOLO integrated into ROS, your robot can run [object detection](../tasks/detect.md) and [segmentation](../tasks/segment.md) across RGB images, depth images, and point clouds, turning raw sensor streams into actionable perception. From here, explore the [Predict mode](../modes/predict.md) for more inference options, or follow the [steps of a computer vision project](steps-of-a-cv-project.md) to take your robotics application from prototype to production.

## FAQ

### Should I use ROS 1 or ROS 2 with Ultralytics YOLO?

Use whichever your robot or simulation environment already runs — every code example in this guide ships a ROS1 (`rospy`, tested on Noetic) tab and a ROS2 (`rclpy`, targeting current LTS releases like Humble and Jazzy) tab. If you have a free choice, prefer ROS2: it is the actively developed line, with better real-time performance, security, and multi-robot support (see the [ROS 1 vs. ROS 2 comparison](#background-about-ros)).

### How do I integrate Ultralytics YOLO with ROS for real-time object detection?

Integrating Ultralytics YOLO with ROS involves setting up a ROS environment and using YOLO for processing sensor data. Begin by installing the required dependencies — `ros_numpy` through the ROS package manager, and Ultralytics YOLO through pip:

```bash
sudo apt install ros-noetic-ros-numpy
pip install ultralytics
```

Next, create a ROS node and subscribe to an image topic to process the incoming data for [object detection](../tasks/detect.md). Here is a minimal example:

```python
import ros_numpy
import rospy
from sensor_msgs.msg import Image

from ultralytics import YOLO

detection_model = YOLO("yolo26m.pt")
rospy.init_node("ultralytics")
det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)


def callback(data):
    array = ros_numpy.numpify(data)
    if data.encoding == "rgb8":
        array = array[..., ::-1]  # normalize to bgr8, the channel order YOLO expects
    det_result = detection_model(array)
    det_annotated = det_result[0].plot(show=False)
    det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="bgr8"))


rospy.Subscriber("/camera/color/image_raw", Image, callback)
rospy.spin()
```

### What are ROS topics and how are they used in Ultralytics YOLO?

ROS topics facilitate communication between nodes in a ROS network by using a publish-subscribe model. A topic is a named channel that nodes use to send and receive messages asynchronously. In the context of Ultralytics YOLO, you can make a node subscribe to an image topic, process the images using YOLO for tasks like [detection](../tasks/detect.md) or [segmentation](../tasks/segment.md), and publish outcomes to new topics.

For example, subscribe to a camera topic and process the incoming image for detection:

```python
rospy.Subscriber("/camera/color/image_raw", Image, callback)
```

### Why use depth images with Ultralytics YOLO in ROS?

Depth images in ROS, represented by `sensor_msgs/Image`, provide the distance of objects from the camera, crucial for tasks like obstacle avoidance, 3D mapping, and localization. By [using depth information](https://en.wikipedia.org/wiki/Depth_map) along with RGB images, robots can better understand their 3D environment.

With YOLO, you can extract [segmentation masks](https://www.ultralytics.com/glossary/image-segmentation) from RGB images and apply these masks to depth images to obtain precise 3D object information, improving the robot's ability to navigate and interact with its surroundings.

### What hardware do I need to run Ultralytics YOLO with ROS?

Any machine capable of running [Ultralytics YOLO inference](../modes/predict.md) works — a GPU speeds up inference but isn't required for smaller models like `yolo26n.pt`. On the ROS side, you need any RGB camera for [image detection](#use-ultralytics-with-ros-sensor_msgsimage), a depth-capable sensor such as an [Intel RealSense](https://www.realsenseai.com/) for [depth workflows](#use-ultralytics-with-ros-depth-images), or an RGB-D camera publishing an organized, colorized `sensor_msgs/PointCloud2` for [point cloud workflows](#use-ultralytics-with-ros-sensor_msgspointcloud2) — a raw LIDAR cloud lacks the `rgb` field and organized grid this workflow assumes. This guide was tested on the [Husarion ROSbot 2 PRO](https://husarion.com/manuals/rosbot/), but the code works with any ROS Noetic- or ROS2-compatible robot or simulation.
