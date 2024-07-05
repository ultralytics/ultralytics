---
comments: true
description: todo
keywords: Ultralytics, YOLO, object detection, deep learning, machine learning, guide, ROS, Robot Operating System, robotics, ROS Noetic, Python, Ubuntu, simulation, visualization, communication, middleware, hardware abstraction, tools, utilities, ecosystem, Noetic Ninjemys, autonomous vehicle, AMV
---

# ROS (Robot Operating System) quickstart guide

<p align="center"> <iframe src="https://player.vimeo.com/video/639236696?h=740f412ce5" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe></p>
<p align="center"><a href="https://vimeo.com/639236696">ROS Introduction (captioned)</a> from <a href="https://vimeo.com/osrfoundation">Open Robotics</a> on <a href="https://vimeo.com">Vimeo</a>.</p>

## What is ROS?

The [Robot Operating System (ROS)](https://www.ros.org/) is an open-source framework widely used in robotics research and industry. ROS provides a collection of [libraries and tools](https://www.ros.org/blog/ecosystem/) to help developers create robot applications. ROS is designed to work with various [robotic platforms](https://robots.ros.org/), making it a flexible and powerful tool for roboticists.

### Key Features of ROS

1. **Modular Architecture**: ROS has a modular architecture, allowing developers to build complex systems by combining smaller, reusable components called [nodes](https://wiki.ros.org/ROS/Tutorials/UnderstandingNodes). Each node typically performs a specific function, and nodes communicate with each other using messages over [topics](https://wiki.ros.org/ROS/Tutorials/UnderstandingTopics) or [services](https://wiki.ros.org/ROS/Tutorials/UnderstandingServicesParams).

2. **Communication Middleware**: ROS offers a robust communication infrastructure that supports inter-process communication and distributed computing. This is achieved through a publish-subscribe model for data streams (topics) and a request-reply model for service calls.

3. **Hardware Abstraction**: ROS provides a layer of abstraction over the hardware, enabling developers to write device-agnostic code. This allows the same code to be used with different hardware setups, facilitating easier integration and experimentation.

4. **Tools and Utilities**: ROS comes with a rich set of tools and utilities for visualization, debugging, and simulation. For instance, RViz is used for visualizing sensor data and robot state information, while Gazebo provides a powerful simulation environment for testing algorithms and robot designs.

5. **Extensive Ecosystem**: The ROS ecosystem is vast and continually growing, with numerous packages available for different robotic applications, including navigation, manipulation, perception, and more. The community actively contributes to the development and maintenance of these packages.

???+ note "Evolution of ROS Versions"

    Since its development in 2007, ROS has evolved through [multiple versions](https://wiki.ros.org/Distributions), each introducing new features and improvements to meet the growing needs of the robotics community. The development of ROS can be categorized into two main series: ROS 1 and ROS 2. This guide focuses on the Long Term Support (LTS) version of ROS 1, known as ROS Noetic Ninjemys, the code should also work with earlier versions.

    ### ROS 1 vs. ROS 2

    While ROS 1 provided a solid foundation for robotic development, ROS 2 addresses its shortcomings by offering:

    - **Real-time Performance**: Improved support for real-time systems and deterministic behavior.
    - **Security**: Enhanced security features for safe and reliable operation in various environments.
    - **Scalability**: Better support for multi-robot systems and large-scale deployments.
    - **Cross-platform Support**: Expanded compatibility with various operating systems beyond Linux, including Windows and macOS.
    - **Flexible Communication**: Use of DDS for more flexible and efficient inter-process communication.

### ROS Messages and Topics

In ROS, communication between nodes is facilitated through [messages](https://wiki.ros.org/Messages) and [topics](https://wiki.ros.org/Topics). A message is a data structure that defines the information exchanged between nodes, while a topic is a named channel over which messages are sent and received. Nodes can publish messages to a topic or subscribe to messages from a topic, enabling them to communicate with each other. This publish-subscribe model allows for asynchronous communication and decoupling between nodes. Each sensor or actuator in a robotic system typically publishes data to a topic, which can then be consumed by other nodes for processing or control. For the purpose of this guide, we will focus on Image, Depth and PointCloud messages and camera topics.

## Setting Up Ultralytics YOLO with ROS

This guide has been tested using [this ROS environment](https://github.com/ambitious-octopus/rosbot_ros/tree/noetic), which is a fork of the [ROSbot ROS repository](https://github.com/husarion/rosbot_ros). This environment includes the Ultralytics YOLO package, a Docker container for easy setup, comprehensive ROS packages, and Gazebo worlds for rapid testing. It is designed to work with the [Husarion ROSbot 2 PRO](https://husarion.com/manuals/rosbot/). The code examples provided will work in any ROS Noetic/Melodic environment, including both simulation and real-world.

<p align="center">
  <img width="50%" src="https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/242b431d-6ea2-4dad-81d6-e31be69141af" alt="Husarion ROSbot 2 PRO">
</p>

### Dependencies Installation

Apart from the ROS environment, you will need to install the following dependencies:

- **[ROS Numpy package](https://github.com/eric-wieser/ros_numpy)**: This is required for fast conversion between ROS Image messages and numpy arrays.

    ```bash
    pip install ros_numpy
    ```

- **Ultralytics package**:

    ```bash
    pip install ultralytics
    ```

## Use Ultralytics with ROS `sensor_msgs/Image`

The `sensor_msgs/Image` [message type](https://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html) is commonly used in ROS for representing image data. It contains fields for encoding, height, width, and pixel data, making it suitable for transmitting images captured by cameras or other sensors. Image messages are widely used in robotic applications for tasks such as visual perception, object detection, and navigation.

<p align="center">
  <img width="100%" src="https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/652cb3e8-ecb0-45cf-9ce1-a514dc06c605" alt="Detection and Segmentation in ROS Gazebo">
</p>

### Image Step-by-Step Usage

The following code snippet demonstrates how to use the Ultralytics YOLO package with ROS. In this example, we subscribe to a camera topic, process the incoming image using YOLO, and publish the detected objects to new topics for [detection](../tasks/detect.md) and [segmentation](../tasks/segment.md).

First, import the necessary libraries and instantiate two models: one for [segmentation](../tasks/segment.md) and one for [detection](../tasks/detect.md). Initialize a ROS node (with the name `ultralytics`) to enable communication with the ROS master. To ensure a stable connection, we include a brief pause, giving the node sufficient time to establish the connection before proceeding.

```python
import time

import rospy

from ultralytics import YOLO

detection_model = YOLO("yolov8m.pt")
segmentation_model = YOLO("yolov8m-seg.pt")
rospy.init_node("ultralytics")
time.sleep(1)
```

Initialize two ROS topics: one for [detection](../tasks/detect.md) and one for [segmentation](../tasks/segment.md). These topics will be used to publish the annotated images, making them accessible for further processing. The communication between nodes is facilitated using `sensor_msgs/Image` messages.

```python
from sensor_msgs.msg import Image

det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)
```

Finally, create a subscriber that listens to messages on the `/camera/color/image_raw` topic and calls a callback function for each new message. This callback function receives messages of type `sensor_msgs/Image`, converts them into a numpy array using `ros_numpy`, processes the images with the previously instantiated YOLO models, annotates the images, and then publishes them back to the respective topics: `/ultralytics/detection/image` for detection and `/ultralytics/segmentation/image` for segmentation.

```python
import ros_numpy


def callback(data):
    """Callback function to process image and publish annotated images."""
    array = ros_numpy.numpify(data)
    if det_image_pub.get_num_connections():
        det_result = detection_model(array)
        det_annotated = det_result[0].plot(show=False)
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

    if seg_image_pub.get_num_connections():
        seg_result = segmentation_model(array)
        seg_annotated = seg_result[0].plot(show=False)
        seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))


rospy.Subscriber("/camera/color/image_raw", Image, callback)

while True:
    rospy.spin()
```

??? Example "Complete code"

    ```python
    import time

    import ros_numpy
    import rospy
    from sensor_msgs.msg import Image

    from ultralytics import YOLO

    detection_model = YOLO("yolov8m.pt")
    segmentation_model = YOLO("yolov8m-seg.pt")
    rospy.init_node("ultralytics")
    time.sleep(1)

    det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
    seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)


    def callback(data):
        """Callback function to process image and publish annotated images."""
        array = ros_numpy.numpify(data)
        if det_image_pub.get_num_connections():
            det_result = detection_model(array)
            det_annotated = det_result[0].plot(show=False)
            det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

        if seg_image_pub.get_num_connections():
            seg_result = segmentation_model(array)
            seg_annotated = seg_result[0].plot(show=False)
            seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))


    rospy.Subscriber("/camera/color/image_raw", Image, callback)

    while True:
        rospy.spin()
    ```

???+ tip "Debugging"

    Debugging ROS (Robot Operating System) nodes can be challenging due to the system's distributed nature. Several tools can assist with this process:

    1. `rostopic echo <TOPIC-NAME>` : This command allows you to view messages published on a specific topic, helping you inspect the data flow.
    2. `rostopic list`: Use this command to list all available topics in the ROS system, giving you an overview of the active data streams.
    3. `rqt_graph`: This visualization tool displays the communication graph between nodes, providing insights into how nodes are interconnected and how they interact.
    4. For more complex visualizations, such as 3D representations, you can use [RViz](https://wiki.ros.org/rviz). RViz (ROS Visualization) is a powerful 3D visualization tool for ROS. It allows you to visualize the state of your robot and its environment in real-time. With RViz, you can view sensor data (e.g. `sensors_msgs/Image`), robot model states, and various other types of information, making it easier to debug and understand the behavior of your robotic system.

### Publish Detected Classes with `std_msgs/String`

Standard ROS messages also include `std_msgs/String` messages. In many applications, it is not necessary to republish the entire annotated image; instead, only the classes present in the robot's view are needed. The following example demonstrates how to use `std_msgs/String` [messages](https://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html) to republish the detected classes on the `/ultralytics/detection/classes` topic. These messages are more lightweight and provide essential information, making them valuable for various applications.

#### Example Use Case

Consider a warehouse robot equipped with a camera and object [detection model](../tasks/detect.md). Instead of sending large annotated images over the network, the robot can publish a list of detected classes as `std_msgs/String` messages. For instance, when the robot detects objects like "box", "pallet" and "forklift" it publishes these classes to the `/ultralytics/detection/classes` topic. This information can then be used by a central monitoring system to track the inventory in real-time, optimize the robot's path planning to avoid obstacles, or trigger specific actions such as picking up a detected box. This approach reduces the bandwidth required for communication and focuses on transmitting critical data.

### String Step-by-Step Usage

This example demonstrates how to use the Ultralytics YOLO package with ROS. In this example, we subscribe to a camera topic, process the incoming image using YOLO, and publish the detected objects to new topic `/ultralytics/detection/classes` using `std_msgs/String` messages. The `ros_numpy` package is used to convert the ROS Image message to a numpy array for processing with YOLO.

```python
import time

import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ultralytics import YOLO

detection_model = YOLO("yolov8m.pt")
rospy.init_node("ultralytics")
time.sleep(1)
classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)


def callback(data):
    """Callback function to process image and publish detected classes."""
    array = ros_numpy.numpify(data)
    if classes_pub.get_num_connections():
        det_result = detection_model(array)
        classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
        names = [det_result[0].names[i] for i in classes]
        classes_pub.publish(String(data=str(names)))


rospy.Subscriber("/camera/color/image_raw", Image, callback)
while True:
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

    When working with depth images, it is essential to ensure that the RGB and depth images are correctly aligned. RGB-D cameras, such as the [Intel RealSense](https://www.intelrealsense.com/) series, provide synchronized RGB and depth images, making it easier to combine information from both sources. If using separate RGB and depth cameras, it is crucial to calibrate them to ensure accurate alignment.

#### Depth Step-by-Step Usage

In this example, we use YOLO to segment an image and apply the extracted mask to segment the object in the depth image. This allows us to determine the distance of each pixel of the object of interest from the camera's focal center. By obtaining this distance information, we can calculate the distance between the camera and the specific object in the scene. Begin by importing the necessary libraries, creating a ROS node, and instantiating a segmentation model and a ROS topic.

```python
import time

import rospy
from std_msgs.msg import String

from ultralytics import YOLO

rospy.init_node("ultralytics")
time.sleep(1)

segmentation_model = YOLO("yolov8m-seg.pt")

classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
```

Next, define a callback function that processes the incoming depth image message. The function waits for the depth image and RGB image messages, converts them into numpy arrays, and applies the segmentation model to the RGB image. It then extracts the segmentation mask for each detected object and calculates the average distance of the object from the camera using the depth image. Most sensors have a maximum distance, known as the clip distance, beyond which values are represented as inf (`np.inf`). Before processing, it is important to filter out these null values and assign them a value of `0`. Finally, it publishes the detected objects along with their average distances to the `/ultralytics/detection/distance` topic.

```python
import numpy as np
import ros_numpy
from sensor_msgs.msg import Image


def callback(data):
    """Callback function to process depth image and RGB image."""
    image = rospy.wait_for_message("/camera/color/image_raw", Image)
    image = ros_numpy.numpify(image)
    depth = ros_numpy.numpify(data)
    result = segmentation_model(image)

    for index, cls in enumerate(result[0].boxes.cls):
        class_index = int(cls.cpu().numpy())
        name = result[0].names[class_index]
        mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
        obj = depth[mask == 1]
        obj = obj[~np.isnan(obj)]
        avg_distance = np.mean(obj) if len(obj) else np.inf

    classes_pub.publish(String(data=str(all_objects)))


rospy.Subscriber("/camera/depth/image_raw", Image, callback)

while True:
    rospy.spin()
```

??? Example "Complete code"

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

    segmentation_model = YOLO("yolov8m-seg.pt")

    classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)


    def callback(data):
        """Callback function to process depth image and RGB image."""
        image = rospy.wait_for_message("/camera/color/image_raw", Image)
        image = ros_numpy.numpify(image)
        depth = ros_numpy.numpify(data)
        result = segmentation_model(image)

        for index, cls in enumerate(result[0].boxes.cls):
            class_index = int(cls.cpu().numpy())
            name = result[0].names[class_index]
            mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
            obj = depth[mask == 1]
            obj = obj[~np.isnan(obj)]
            avg_distance = np.mean(obj) if len(obj) else np.inf

        classes_pub.publish(String(data=str(all_objects)))


    rospy.Subscriber("/camera/depth/image_raw", Image, callback)

    while True:
        rospy.spin()
    ```

## Use Ultralytics with ROS `sensor_msgs/PointCloud2`

<p align="center">
  <img width="100%" src="https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/ef2e1ed9-a840-499a-b324-574bd26c3bc7" alt="Detection and Segmentation in ROS Gazebo">
</p>

The `sensor_msgs/PointCloud2` [message type](https://docs.ros.org/en/api/sensor_msgs/html/msg/PointCloud2.html) is a data structure used in ROS to represent 3D point cloud data. This message type is integral to robotic applications, enabling tasks such as 3D mapping, object recognition, and localization.

A point cloud is a collection of data points defined within a three-dimensional coordinate system. These data points represent the external surface of an object or a scene, captured via 3D scanning technologies. Each point in the cloud has `X`, `Y`, and `Z` coordinates, which correspond to its position in space, and may also include additional information such as color and intensity.

!!! warning "Reference frame"

    When working with `sensor_msgs/PointCloud2`, it's essential to consider the reference frame of the sensor from which the point cloud data was acquired. The point cloud is initially captured in the sensor's reference frame. You can determine this reference frame by listening to the `/tf_static` topic. However, depending on your specific application requirements, you might need to convert the point cloud into another reference frame. This transformation can be achieved using the `tf2_ros` package, which provides tools for managing coordinate frames and transforming data between them.

!!! tip "Obtaining Point clouds"

    Point Clouds can be obtained using various sensors:

    1. **LIDAR (Light Detection and Ranging)**: Uses laser pulses to measure distances to objects and create high-precision 3D maps.
    2. **Depth Cameras**: Capture depth information for each pixel, allowing for 3D reconstruction of the scene.
    3. **Stereo Cameras**: Utilize two or more cameras to obtain depth information through triangulation.
    4. **Structured Light Scanners**: Project a known pattern onto a surface and measure the deformation to calculate depth.

### Using YOLO with Point Clouds

To integrate YOLO with `sensor_msgs/PointCloud2` type messages, we can employ a method similar to the one used for depth maps. By leveraging the color information embedded in the point cloud, we can extract a 2D image, perform segmentation on this image using YOLO, and then apply the resulting mask to the three-dimensional points to isolate the 3D object of interest.

For handling point clouds, we recommend using Open3D (`pip install open3d`), a user-friendly Python library. Open3D provides robust tools for managing point cloud data structures, visualizing them, and executing complex operations seamlessly. This library can significantly simplify the process and enhance our ability to manipulate and analyze point clouds in conjunction with YOLO-based segmentation.

#### Point Clouds Step-by-Step Usage

Import the necessary libraries and instantiate the YOLO model for segmentation.

```python
import time

import rospy

from ultralytics import YOLO

rospy.init_node("ultralytics")
time.sleep(1)
segmentation_model = YOLO("yolov8m-seg.pt")
```

Create a function `pointcloud2_to_array`, which transforms a `sensor_msgs/PointCloud2` message into two numpy arrays. The `sensor_msgs/PointCloud2` messages contain `n` points based on the `width` and `height` of the acquired image. For instance, a `480 x 640` image will have `307,200` points. Each point includes three spatial coordinates (`xyz`) and the corresponding color in `RGB` format. These can be considered as two separate channels of information.

The function returns the `xyz` coordinates and `RGB` values in the format of the original camera resolution (`width x height`). Most sensors have a maximum distance, known as the clip distance, beyond which values are represented as inf (`np.inf`). Before processing, it is important to filter out these null values and assign them a value of `0`.

```python
import numpy as np
import ros_numpy


def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
    """
    Convert a ROS PointCloud2 message to a numpy array.

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

Next, subscribe to the `/camera/depth/points` topic to receive the point cloud message and convert the `sensor_msgs/PointCloud2` message into numpy arrays containing the XYZ coordinates and RGB values (using the `pointcloud2_to_array` function). Process the RGB image using the YOLO model to extract segmented objects. For each detected object, extract the segmentation mask and apply it to both the RGB image and the XYZ coordinates to isolate the object in 3D space.

Processing the mask is straightforward since it consists of binary values, with `1` indicating the presence of the object and `0` indicating the absence. To apply the mask, simply multiply the original channels by the mask. This operation effectively isolates the object of interest within the image. Finally, create an Open3D point cloud object and visualize the segmented object in 3D space with associated colors.

```python
import sys

import open3d as o3d

ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
xyz, rgb = pointcloud2_to_array(ros_cloud)
result = segmentation_model(rgb)

if not len(result[0].boxes.cls):
    print("No objects detected")
    sys.exit()

classes = result[0].boxes.cls.cpu().numpy().astype(int)
for index, class_id in enumerate(classes):
    mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
    mask_expanded = np.stack([mask, mask, mask], axis=2)

    obj_rgb = rgb * mask_expanded
    obj_xyz = xyz * mask_expanded

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_xyz.reshape((ros_cloud.height * ros_cloud.width, 3)))
    pcd.colors = o3d.utility.Vector3dVector(obj_rgb.reshape((ros_cloud.height * ros_cloud.width, 3)) / 255)
    o3d.visualization.draw_geometries([pcd])
```

??? Example "Complete code"

    ```python
    import sys
    import time

    import numpy as np
    import open3d as o3d
    import ros_numpy
    import rospy

    from ultralytics import YOLO

    rospy.init_node("ultralytics")
    time.sleep(1)
    segmentation_model = YOLO("yolov8m-seg.pt")


    def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
        """
        Convert a ROS PointCloud2 message to a numpy array.

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
    result = segmentation_model(rgb)

    if not len(result[0].boxes.cls):
        print("No objects detected")
        sys.exit()

    classes = result[0].boxes.cls.cpu().numpy().astype(int)
    for index, class_id in enumerate(classes):
        mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
        mask_expanded = np.stack([mask, mask, mask], axis=2)

        obj_rgb = rgb * mask_expanded
        obj_xyz = xyz * mask_expanded

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_xyz.reshape((ros_cloud.height * ros_cloud.width, 3)))
        pcd.colors = o3d.utility.Vector3dVector(obj_rgb.reshape((ros_cloud.height * ros_cloud.width, 3)) / 255)
        o3d.visualization.draw_geometries([pcd])
    ```

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/ultralytics/assets/3855193/3caafc4a-0edd-4e5f-8dd1-37e30be70123" alt="Point Cloud Segmentation with Ultralytics ">
</p>

## FAQ

### What is the Robot Operating System (ROS)?

The [Robot Operating System (ROS)](https://www.ros.org/) is an open-source framework commonly used in robotics to help developers create robust robot applications. It provides a collection of [libraries and tools](https://www.ros.org/blog/ecosystem/) for building and interfacing with robotic systems, enabling easier development of complex applications. ROS supports communication between nodes using messages over [topics](https://wiki.ros.org/ROS/Tutorials/UnderstandingTopics) or [services](https://wiki.ros.org/ROS/Tutorials/UnderstandingServicesParams).

### How do I integrate Ultralytics YOLO with ROS for real-time object detection?

Integrating Ultralytics YOLO with ROS involves setting up a ROS environment and using YOLO for processing sensor data. Begin by installing the required dependencies like `ros_numpy` and Ultralytics YOLO:

```bash
pip install ros_numpy ultralytics
```

Next, create a ROS node and subscribe to an [image topic](../tasks/detect.md) to process the incoming data. Here is a minimal example:

```python
import ros_numpy
import rospy
from sensor_msgs.msg import Image

from ultralytics import YOLO

detection_model = YOLO("yolov8m.pt")
rospy.init_node("ultralytics")
det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)


def callback(data):
    array = ros_numpy.numpify(data)
    det_result = detection_model(array)
    det_annotated = det_result[0].plot(show=False)
    det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))


rospy.Subscriber("/camera/color/image_raw", Image, callback)
rospy.spin()
```

### What are ROS topics and how are they used in Ultralytics YOLO?

ROS topics facilitate communication between nodes in a ROS network by using a publish-subscribe model. A topic is a named channel that nodes use to send and receive messages asynchronously. In the context of Ultralytics YOLO, you can make a node subscribe to an image topic, process the images using YOLO for tasks like detection or segmentation, and publish outcomes to new topics.

For example, subscribe to a camera topic and process the incoming image for detection:

```python
rospy.Subscriber("/camera/color/image_raw", Image, callback)
```

### Why use depth images with Ultralytics YOLO in ROS?

Depth images in ROS, represented by `sensor_msgs/Image`, provide the distance of objects from the camera, crucial for tasks like obstacle avoidance, 3D mapping, and localization. By [using depth information](https://en.wikipedia.org/wiki/Depth_map) along with RGB images, robots can better understand their 3D environment.

With YOLO, you can extract segmentation masks from RGB images and apply these masks to depth images to obtain precise 3D object information, improving the robot's ability to navigate and interact with its surroundings.

### How can I visualize 3D point clouds with YOLO in ROS?

To visualize 3D point clouds in ROS with YOLO:

1. Convert `sensor_msgs/PointCloud2` messages to numpy arrays.
2. Use YOLO to segment RGB images.
3. Apply the segmentation mask to the point cloud.

Here's an example using Open3D for visualization:

```python
import sys

import open3d as o3d
import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2

from ultralytics import YOLO

rospy.init_node("ultralytics")
segmentation_model = YOLO("yolov8m-seg.pt")


def pointcloud2_to_array(pointcloud2):
    pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
    split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
    rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
    xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
    xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
    return xyz, rgb


ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2)
xyz, rgb = pointcloud2_to_array(ros_cloud)
result = segmentation_model(rgb)

if not len(result[0].boxes.cls):
    print("No objects detected")
    sys.exit()

classes = result[0].boxes.cls.cpu().numpy().astype(int)
for index, class_id in enumerate(classes):
    mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
    mask_expanded = np.stack([mask, mask, mask], axis=2)

    obj_rgb = rgb * mask_expanded
    obj_xyz = xyz * mask_expanded

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_xyz.reshape((-1, 3)))
    pcd.colors = o3d.utility.Vector3dVector(obj_rgb.reshape((-1, 3)) / 255)
    o3d.visualization.draw_geometries([pcd])
```

This approach provides a 3D visualization of segmented objects, useful for tasks like navigation and manipulation.
