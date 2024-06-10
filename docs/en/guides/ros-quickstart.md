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
        detection_model = YOLO("yolov8n.pt")
        segmentation_model = YOLO("yolov8n-seg.pt")
        rospy.init_node('ultralytics') # Initialize ROS node
        time.sleep(1) # Wait for node to initialize

        # create 2 publishers for detection and segmentation
        det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
        seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)


        def callback(data):
            """
            Callback function to process incoming image data
            """
            array = ros_numpy.numpify(data)
            if det_image_pub.get_num_connections():
                det_result = detection_model(array)
                det_annotated = det_result[0].plot(show=False)
                det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding='rgb8'))
            
            if seg_image_pub.get_num_connections():
                seg_result = segmentation_model(array)
                seg_annotated = seg_result[0].plot(show=False)
                seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding='rgb8'))
            
        # create a subscriber to listen to the camera topic
        rospy.Subscriber("/camera/color/image_raw", Image, callback)

        # keep the node running
        while True:
            rospy.spin()
        ```

        This code snippet demonstrates how to use the Ultralytics YOLO package with ROS. In this example, we subscribe to a camera topic, process the incoming image using YOLO, and publish the detected objects to new topics for detection and segmentation. The `ros_numpy` package is used to convert the ROS Image message to a numpy array for processing with YOLO. The detected objects are then annotated on the image and published back as Image messages for visualization or further processing.

    === "ROS 2 Humble"

        ```python
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        from sensor_msgs.msg import Image
        detection_model = YOLO("yolov8n.pt")
        segmentation_model = YOLO("yolov8n-seg.pt")
        rospy.init_node('integration')
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

If instead of republishing the image with the detected objects, you want to publish the detected classes you can use the following code:

!!! Example "Usage"

    === "ROS Noetic"

        ``` py
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        detection_model = YOLO("yolov8n.pt")
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

    === "ROS 2 Humble"

        ``` py
        import rospy
        import time
        from ultralytics import YOLO
        import ros_numpy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        detection_model = YOLO("yolov8n.pt")
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

In addition to RGB images, ROS also supports depth images, which provide information about the distance of objects from the camera. Depth images are commonly used in robotic applications for tasks such as obstacle avoidance, 3D mapping, and localization. The `sensor_msgs/Image` message type can be used to represent depth images in ROS, with additional fields for encoding, height, width, and pixel data.


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

