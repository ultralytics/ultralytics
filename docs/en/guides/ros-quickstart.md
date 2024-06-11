---
comments: true
description: todo
keywords: Ultralytics, YOLO, object detection, deep learning, machine learning, guide, ROS, Robot Operating System, robotics, ROS Noetic, Python, Ubuntu, simulation, visualization, communication, middleware, hardware abstraction, tools, utilities, ecosystem, Noetic Ninjemys,compatibility, performance, stability
---

# ROS (Robot Operating System) quickstart guide

<p align="center"> <iframe src="https://player.vimeo.com/video/639236696?h=740f412ce5" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe></p>
<p align="center"><a href="https://vimeo.com/639236696">ROS Introduction (captioned)</a> from <a href="https://vimeo.com/osrfoundation">Open Robotics</a> on <a href="https://vimeo.com">Vimeo</a>.</p>

### What is ROS?

The [Robot Operating System (ROS)](https://www.ros.org/) is an open-source framework widely used in robotics research and industry. ROS provides a collection of [libraries and tools](https://www.ros.org/blog/ecosystem/) to help developers create robot applications. ROS is designed to work with various [robotic platforms](https://robots.ros.org/), making it a flexible and powerful tool for roboticists.

### Key Features of ROS

1. **Modular Architecture**: ROS has a modular architecture, allowing developers to build complex systems by combining smaller, reusable components called nodes. Each node typically performs a specific function, and nodes communicate with each other using messages over topics or services.

2. **Communication Middleware**: ROS offers a robust communication infrastructure that supports inter-process communication and distributed computing. This is achieved through a publish-subscribe model for data streams (topics) and a request-reply model for service calls.

3. **Hardware Abstraction**: ROS provides a layer of abstraction over the hardware, enabling developers to write device-agnostic code. This allows the same code to be used with different hardware setups, facilitating easier integration and experimentation.

4. **Tools and Utilities**: ROS comes with a rich set of tools and utilities for visualization, debugging, and simulation. For instance, RViz is used for visualizing sensor data and robot state information, while Gazebo provides a powerful simulation environment for testing algorithms and robot designs.

5. **Extensive Ecosystem**: The ROS ecosystem is vast and continually growing, with numerous packages available for different robotic applications, including navigation, manipulation, perception, and more. The community actively contributes to the development and maintenance of these packages.

???+ note "Evolution of ROS Versions"

    Since its inception in 2007, ROS has evolved through multiple versions, each introducing new features and improvements to meet the growing needs of the robotics community. The development of ROS can be categorized into two main series: ROS 1 and ROS 2. This guide focuses on the Long Term Support (LTS) version of ROS 1, known as ROS Noetic Ninjemys and the LTS of ROS 2, known as Galactic Geochelone.

    ### ROS 1 vs. ROS 2

    While ROS 1 provided a solid foundation for robotic development, ROS 2 addresses its shortcomings by offering:

    - **Real-time Performance**: Improved support for real-time systems and deterministic behavior.
    - **Security**: Enhanced security features for safe and reliable operation in various environments.
    - **Scalability**: Better support for multi-robot systems and large-scale deployments.
    - **Cross-platform Support**: Expanded compatibility with various operating systems beyond Linux, including Windows and macOS.
    - **Flexible Communication**: Use of DDS for more flexible and efficient inter-process communication.

### ROS Messages and Topics

In ROS, communication between nodes is facilitated through [messages](https://wiki.ros.org/Messages) and [topics](https://wiki.ros.org/Topics). A message is a data structure that defines the information exchanged between nodes, while a topic is a named channel over which messages are sent and received. Nodes can publish messages to a topic or subscribe to messages from a topic, enabling them to communicate with each other. This publish-subscribe model allows for asynchronous communication and decoupling between nodes. Each sensor or actuator in a robotic system typically publishes data to a topic, which can then be consumed by other nodes for processing or control. For the purpose of this guide, we will focus on Image messages and camera topics.

## Setting Up Ultralytics YOLO with ROS

This guide has been tested using [this ROS environment](https://github.com/ambitious-octopus/rosbot_ros.git), which is a fork of the Rosbot ROS repository. This environment includes the Ultralytics YOLO package, a Docker container for easy setup, comprehensive ROS packages, and Gazebo worlds for rapid testing. It is designed to work with the Husarion ROSbot robot. The code examples provided should work in any ROS environment, including both simulation and real-world applications.


### Dependencies Installation

Apart from the ROS environment, you will need to install the following dependencies:

- **ROS Numpy package**: This is required for fast conversion between ROS Image messages and numpy arrays.
    ``` bash
    pip install ros_numpy
    ```


- **Ultralytics package**:
  
    ``` bash
    pip install ultralytics
    ``` 

## Use Ultralytics with ROS `sensor_msgs/Image`

The `sensor_msgs/Image` [message type](https://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html) is commonly used in ROS for representing image data. It contains fields for encoding, height, width, and pixel data, making it suitable for transmitting images captured by cameras or other sensors. Image messages are widely used in robotic applications for tasks such as visual perception, object detection, and navigation.

!!! Example "Usage"

    === "ROS Noetic"

        ``` py
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        from sensor_msgs.msg import Image
        detection_model = YOLO("yolov8m.pt") # (1) 
        segmentation_model = YOLO("yolov8m-seg.pt")
        rospy.init_node('ultralytics') # (2)
        time.sleep(1) # (3)

        # (4)
        det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
        seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)


        def callback(data):
            """
            Callback function to process incoming image data
            """
            array = ros_numpy.numpify(data) # (5) 
            if det_image_pub.get_num_connections(): # (6)
                det_result = detection_model(array) # (7)
                det_annotated = det_result[0].plot(show=False) # (8)
                det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding='rgb8'))
            
            if seg_image_pub.get_num_connections():
                seg_result = segmentation_model(array)
                seg_annotated = seg_result[0].plot(show=False)
                seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding='rgb8'))
            
        rospy.Subscriber("/camera/color/image_raw", Image, callback) # (9)

        while True: # (10)
            rospy.spin()
        ```
        
        1. Load the YOLO models for detection and segmentation.
        2. Initialize ROS node with the name `ultralytics`.
        3. Wait for the node to initialize before starting the main loop.
        4. Create 2 publishers topic for detection and segmentation
        5. Convert the ROS Image message to a numpy array for processing with YOLO.
        6. Check if there are any subscribers to the publishers before publishing the annotated images, this is to avoid unnecessary processing.
        7. Process the incoming image using YOLO.
        8. Annotate the detected objects on the image and publish it back as an Image message for visualization or further processing.
        9. Create a subscriber to listen to the camera topic and call the callback function to process the incoming image data.
        10. Keep the node running to continue processing incoming messages.
   

        This code snippet demonstrates how to use the Ultralytics YOLO package with ROS. In this example, we subscribe to a camera topic, process the incoming image using YOLO, and publish the detected objects to new topics for detection and segmentation. The `ros_numpy` package is used to convert the ROS Image message to a numpy array for processing with YOLO. The detected objects are then annotated on the image and published back as Image messages for visualization or further processing.

    === "ROS 2 Humble"

        ```python
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        from sensor_msgs.msg import Image
        detection_model = YOLO("yolov8m.pt")
        segmentation_model = YOLO("yolov8m-seg.pt")
        rospy.init_node('ultralytics')
        time.sleep(1)

        det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
        seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)


        def callback(data):
            array = ros_numpy.numpify(data)
            if det_image_pub.get_num_connections():
                det_result = detection_model(array)
                det_annotated = det_result[0].plot(show=False)
                det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding='rgb8'))
            
            if seg_image_pub.get_num_connections():
                seg_result = segmentation_model(array)
                seg_annotated = seg_result[0].plot(show=False)
                seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding='rgb8'))
            

        rospy.Subscriber("/camera/color/image_raw", Image, callback)


        while True:
            rospy.spin()
        ```

???+ tip "Debugging"

    Debugging ROS (Robot Operating System) nodes can be challenging due to the system's distributed nature. Several tools can assist with this process:

    1. `rostopic echo <TOPIC-NAME>` : This command allows you to view messages published on a specific topic, helping you inspect the data flow.
    2. `rostopic list`: Use this command to list all available topics in the ROS system, giving you an overview of the active data streams.
    3. `rqt_graph`: This visualization tool displays the communication graph between nodes, providing insights into how nodes are interconnected and how they interact.
    4. For more complex visualizations, such as 3D representations, you can use RViz. RViz (ROS Visualization) is a powerful 3D visualization tool for ROS. It allows you to visualize the state of your robot and its environment in real-time. With RViz, you can view sensor data (e.g. `sensors_msgs/Image`), robot model states, and various other types of information, making it easier to debug and understand the behavior of your robotic system.

### Publish Detected Classes with `std_msgs/String`
Standard ROS messages also include `std_msgs/String` messages. In many applications, it is not necessary to republish the entire annotated image; instead, only the classes present in the robot's view are needed. The following example demonstrates how to use `std_msgs/String` messages to republish the detected classes on the `/ultralytics/detection/classes` topic. These messages are more lightweight and provide essential information, making them valuable for various applications.

#### Example Use Case
Consider a warehouse robot equipped with a camera and object detection model. Instead of sending large annotated images over the network, the robot can publish a list of detected classes as `std_msgs/String` messages. For instance, when the robot detects objects like "box" "pallet" and "forklift" it publishes these classes to the `/ultralytics/detection/classes` topic. This information can then be used by a central monitoring system to track the inventory in real-time, optimize the robot's path planning to avoid obstacles, or trigger specific actions such as picking up a detected box. This approach reduces the bandwidth required for communication and focuses on transmitting critical data.

!!! Example "Usage"

    === "ROS Noetic"

        ``` py
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        detection_model = YOLO("yolov8m.pt") # (1)
        rospy.init_node('ultralytics') # (2)
        time.sleep(1)

        classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5) # (3)

        def callback(data):
            """
            Callback function to process incoming image data
            """
            array = ros_numpy.numpify(data) # (4)
            if classes_pub.get_num_connections():
                det_result = detection_model(array) # (5)
                classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
                names = [det_result[0].names[i] for i in classes] # (6)
                classes_pub.publish(String(data=str(names))) # (7)

        rospy.Subscriber("/camera/color/image_raw", Image, callback)

        while True:
            rospy.spin()
        ```
        
        1. Load the YOLO model for detection.
        2. Initialize ROS node with the name `ultralytics`.
        3. Create a publisher topic for detected classes.
        4. Use `ros_numpy` to convert the ROS Image message to a numpy array for processing with YOLO.
        5. Process the incoming image using YOLO.
        6. Extract the detected classes from the YOLO result.
        7. Create a `std_msgs/String` message containing the detected classes and publish it to the `/ultralytics/detection/classes` topic.

        This example demonstrates how to use the Ultralytics YOLO package with ROS. In this example, we subscribe to a camera topic, process the incoming image using YOLO, and publish the detected objects to new topic `/ultralytics/detection/classes` using `std_msgs/String` messages. The `ros_numpy` package is used to convert the ROS Image message to a numpy array for processing with YOLO. 

    === "ROS 2 Humble"

        ``` py
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        detection_model = YOLO("yolov8m.pt")
        rospy.init_node('integration')
        time.sleep(1)

        classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)

        def callback(data):
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

        This code snippet demonstrates how to use the Ultralytics YOLO package with ROS. In this example, we subscribe to a camera topic, process the incoming image using YOLO, and publish the detected objects to new topics for detection and segmentation. The `ros_numpy` package is used to convert the ROS Image message to a numpy array for processing with YOLO. The detected objects are then annotated on the image and published back as Image messages for visualization or further processing.

## Use Ultralytics with ROS Depth Images

In addition to RGB images, ROS supports depth images, which provide information about the distance of objects from the camera. Depth images are crucial for robotic applications such as obstacle avoidance, 3D mapping, and localization.

A depth image is an image where each pixel represents the distance from the camera to an object. Unlike RGB images that capture color, depth images capture spatial information, enabling robots to perceive the 3D structure of their environment.

#### Obtaining Depth Images

Depth images can be obtained using various sensors:

1. Stereo Cameras: Use two cameras to calculate depth based on image disparity.
2. Time-of-Flight (ToF) Cameras: Measure the time light takes to return from an object.
3. Structured Light Sensors: Project a pattern and measure its deformation on surfaces.

#### Using YOLO with Depth Images
In ROS, depth images are represented by the `sensor_msgs/Image` message type, which includes fields for encoding, height, width, and pixel data. The encoding field for depth images often uses a format like "16UC1", indicating a 16-bit unsigned integer per pixel, where each value represents the distance to the object. Depth images are commonly used in conjunction with RGB images to provide a more comprehensive view of the environment.

Using YOLO, it is possible to extract and combine information from both RGB and depth images. For instance, YOLO can detect objects within an RGB image, and this detection can be used to pinpoint corresponding regions in the depth image. This allows for the extraction of precise depth information for detected objects, enhancing the robot's ability to understand its environment in three dimensions.

!!! warning "RGB-D Cameras"
    When working with depth images, it is essential to ensure that the RGB and depth images are correctly aligned. RGB-D cameras, such as the Intel RealSense series, provide synchronized RGB and depth images, making it easier to combine information from both sources. If using separate RGB and depth cameras, it is crucial to calibrate them to ensure accurate alignment. 


!!! Example "Usage"

    === "ROS Noetic"

        ``` py
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        import numpy as np
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        rospy.init_node('ultralytics') # (1)
        time.sleep(1)

        segmentation_model = YOLO("yolov8m-seg.pt") # (2)

        classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5) # (3)

        def callback(data):
            """
            Callback function to process incoming depth image data
            """
            image = rospy.wait_for_message("/camera/color/image_raw", Image) # (4)
            image = ros_numpy.numpify(image)
            depth = ros_numpy.numpify(data)
            result = segmentation_model(image) # (5)
            
            all_objects = [] # (6)
            for index, cls in enumerate(result[0].boxes.cls):
                class_index = int(cls.cpu().numpy())
                name = result[0].names[class_index]
                mask = result[0].masks.data.cpu().numpy()[index,:,:].astype(int)
                obj = depth[mask == 1]
                obj = obj[~np.isnan(obj)]
                avg_distance = np.mean(obj) if len(obj) else np.inf
                all_objects.append((name, avg_distance))
            classes_pub.publish(String(data = str(all_objects))) # (7)

        rospy.Subscriber("/camera/depth/image_raw", Image, callback)

        while True:
            rospy.spin()
        ```

        1. Initialize ROS node with the name `ultralytics`.
        2. Initialize the YOLO model for segmentation.
        3. Create a publisher topic for detected classes and distances.
        4. Every time a depth image is received, wait for the corresponding RGB image and process them together.
        5. Process the incoming RGB image using YOLO and extract the detected objects.
        6. For each detected object, apply the segmentation mask to the depth image to calculate the average distance to the object. 
        7. Publish the detected classes and distances as a `std_msgs/String` message.

        This script initializes a ROS node named 'ultralytics' and sets up a YOLO segmentation model to process images. It subscribes to a depth image topic, processes incoming data to detect objects and calculate their average distance, and publishes these results to another topic. The segmentation mask extracted from the RGB image is applied to the depth image to compute the average distance between the camera's optical center and the object for the frame of the camera.
        

    === "ROS 2 Humble"

        ``` py
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        import numpy as np
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        rospy.init_node('integration')
        time.sleep(1)

        segmentation_model = YOLO("yolov8m-seg.pt")

        classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)

        def callback(data):
            image = rospy.wait_for_message("/camera/color/image_raw", Image)
            image = ros_numpy.numpify(image)
            depth = ros_numpy.numpify(data)
            result = segmentation_model(image)
            
            all_objects = []
            for index, cls in enumerate(result[0].boxes.cls):
                class_index = int(cls.cpu().numpy())
                name = result[0].names[class_index]
                mask = result[0].masks.data.cpu().numpy()[index,:,:].astype(int)
                obj = depth[mask == 1]
                obj = obj[~np.isnan(obj)]
                avg_distance = np.mean(obj) if len(obj) else np.inf
                all_objects.append((name, avg_distance))
                
            msg = String()
            msg.data = str(all_objects)
            classes_pub.publish(msg)

        rospy.Subscriber("/camera/depth/image_raw", Image, callback)

        while True:
            rospy.spin()
        ```

        This code snippet demonstrates how to use the Ultralytics YOLO package with ROS. In this example, we subscribe to a camera topic, process the incoming image using YOLO, and publish the detected objects to new topics for detection and segmentation. The `ros_numpy` package is used to convert the ROS Image message to a numpy array for processing with YOLO. The detected objects are then annotated on the image and published back as Image messages for visualization or further processing.

## Use Ultralytics with ROS `sensor_msgs/PointCloud2`

The `sensor_msgs/PointCloud2` [message type](https://docs.ros.org/en/api/sensor_msgs/html/msg/PointCloud2.html) is a data structure used in ROS to represent 3D point cloud data. This message type is integral to robotic applications, enabling tasks such as 3D mapping, object recognition, and localization.

A point cloud is a collection of data points defined within a three-dimensional coordinate system. These data points represent the external surface of an object or a scene, captured via 3D scanning technologies. Each point in the cloud has `X`, `Y`, and `Z` coordinates, which correspond to its position in space, and may also include additional information such as color and intensity.

!!! warning "RGB-D Cameras"
    When working with `sensor_msgs/PointCloud2`, it's essential to consider the reference frame of the sensor from which the point cloud data was acquired. The point cloud is initially captured in the sensor's reference frame. You can determine this reference frame by listening to the `/tf_static` topic. However, depending on your specific application requirements, you might need to convert the point cloud into another reference frame. This transformation can be achieved using the `tf2_ros` package, which provides tools for managing coordinate frames and transforming data between them.

#### Obtaining Point clouds 
Point Clouds can be obtained using various sensors:

1. **LIDAR (Light Detection and Ranging)**: Uses laser pulses to measure distances to objects and create high-precision 3D maps.
2. **Depth Cameras**: Capture depth information for each pixel, allowing for 3D reconstruction of the scene.
3. **Stereo Cameras**: Utilize two or more cameras to obtain depth information through triangulation.
4. **Structured Light Scanners**: Project a known pattern onto a surface and measure the deformation to calculate depth.

#### Using YOLO with Point Clouds

To integrate YOLO with `sensor_msgs/PointCloud2` type messages, we can employ a method similar to the one used for depth maps. By leveraging the color information embedded in the point cloud, we can extract a 2D image, perform segmentation on this image using YOLO, and then apply the resulting mask to the three-dimensional points to isolate the 3D object of interest.

For handling point clouds, we recommend using Open3D (`pip install open3d`), a user-friendly Python library. Open3D provides robust tools for managing point cloud data structures, visualizing them, and executing complex operations seamlessly. This library can significantly simplify the process and enhance our ability to manipulate and analyze point clouds in conjunction with YOLO-based segmentation.


!!! Example "Usage"

    === "ROS Noetic"

        ``` py
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        from sensor_msgs.msg import Image, PointCloud2
        import numpy as np
        import open3d as o3d
        import cv2
        import sys
        rospy.init_node('ultralytics') # (1)
        time.sleep(1)

        def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
            """
            Convert a ROS PointCloud2 message to a numpy array
            Args:
                pointcloud2 (PointCloud2): the PointCloud2 message
            Returns:
                tuple: tuple containing: (xyz, rgb)
            """
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
            split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
            rgb = np.stack([split['b'], split['g'], split['r']], axis=2)
            xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
            xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
            nan_rows = np.isnan(xyz).all(axis=2)
            xyz[nan_rows] = [0, 0, 0]
            rgb[nan_rows] = [0, 0, 0]
            return xyz, rgb


        segmentation_model = YOLO("yolov8m-seg.pt") # (2)
        ros_cloud = rospy.wait_for_message("/camera/depth/points", PointCloud2) # (3)
        xyz, rgb = pointcloud2_to_array(ros_cloud) # (4)
        result = segmentation_model(rgb) # (5)

        if not len(result[0].boxes.cls):
            print("No objects detected")
            sys.exit()

        classes = result[0].boxes.cls.cpu().numpy().astype(int)
        for index, class_id in enumerate(classes):
            mask = result[0].masks.data.cpu().numpy()[index,:,:].astype(int) # (6)
            mask_expanded = np.stack([mask, mask, mask], axis=2)
            
            rgb = rgb * mask_expanded # (7)
            xyz = xyz * mask_expanded # (8)
            
            pcd = o3d.geometry.PointCloud() # (9)
            pcd.points = o3d.utility.Vector3dVector(xyz.reshape((ros_cloud.height* ros_cloud.width, 3)))
            pcd.colors = o3d.utility.Vector3dVector(rgb.reshape((ros_cloud.height* ros_cloud.width, 3)) / 255)
            o3d.visualization.draw_geometries([pcd])
        ```

        1. Initialize ROS node with the name `ultralytics`.
        2. Initialize the YOLO model for segmentation.
        3. Take the first point cloud message from the `/camera/depth/points` topic.
        4. Convert the point cloud message to a numpy array containing the XYZ coordinates and RGB values.
        5. Process the RGB image using YOLO and extract the segmented objects.
        6. Extract the segmentation mask for each segmented object.
        7. Apply the segmentation mask to the RGB image to isolate the object.
        8. Apply the segmentation mask to the XYZ coordinates to isolate the object in 3D space.
        9. Create an Open3D point cloud object and visualize the segmented object in 3D space with colors.

        This code initializes a ROS node named `ultralytics` and a YOLO model for segmentation. It then listens for a point cloud message from the `/camera/depth/points` topic, converts this message into a numpy array with XYZ coordinates and RGB values, and processes the RGB image using the YOLO model to extract segmented objects. For each detected object, the code extracts the segmentation mask and applies it to both the RGB image and the XYZ coordinates to isolate the object in 3D space. Finally, it creates an Open3D point cloud object and visualizes the segmented object in 3D with associated colors.
        