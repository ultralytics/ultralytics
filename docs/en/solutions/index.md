---
comments: true
description: Explore Ultralytics Solutions using YOLO11 for object counting, blurring, security, and more. Enhance efficiency and solve real-world problems with cutting-edge AI.
keywords: Ultralytics, YOLO11, object counting, object blurring, security systems, AI solutions, real-time analysis, computer vision applications
---

# Ultralytics Solutions: Harness YOLO11 to Solve Real-World Problems

Ultralytics Solutions provide cutting-edge applications of YOLO models, offering real-world solutions like object counting, blurring, and security systems, enhancing efficiency and [accuracy](https://www.ultralytics.com/glossary/accuracy) in diverse industries. Discover the power of YOLO11 for practical, impactful implementations.

![Ultralytics Solutions Thumbnail](https://github.com/ultralytics/docs/releases/download/0/ultralytics-solutions-thumbnail.avif)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/bjkt5OE_ANA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Ultralytics Solutions from the Command Line (CLI) | Ultralytics YOLO11 🚀
</p>

## Solutions

Here's our curated list of Ultralytics solutions that can be used to create awesome [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) projects.

- [Object Counting](../guides/object-counting.md) 🚀: Learn to perform real-time object counting with YOLO11. Gain the expertise to accurately count objects in live video streams.
- [Object Cropping](../guides/object-cropping.md) 🚀: Master object cropping with YOLO11 for precise extraction of objects from images and videos.
- [Object Blurring](../guides/object-blurring.md) 🚀: Apply object blurring using YOLO11 to protect privacy in image and video processing.
- [Workouts Monitoring](../guides/workouts-monitoring.md) 🚀: Discover how to monitor workouts using YOLO11. Learn to track and analyze various fitness routines in real time.
- [Objects Counting in Regions](../guides/region-counting.md) 🚀: Count objects in specific regions using YOLO11 for accurate detection in varied areas.
- [Security Alarm System](../guides/security-alarm-system.md) 🚀: Create a security alarm system with YOLO11 that triggers alerts upon detecting new objects. Customize the system to fit your specific needs.
- [Heatmaps](../guides/heatmaps.md) 🚀: Utilize detection heatmaps to visualize data intensity across a matrix, providing clear insights in computer vision tasks.
- [Instance Segmentation with Object Tracking](../guides/instance-segmentation-and-tracking.md) 🚀 NEW: Implement [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) and object tracking with YOLO11 to achieve precise object boundaries and continuous monitoring.
- [VisionEye View Objects Mapping](../guides/vision-eye.md) 🚀: Develop systems that mimic human eye focus on specific objects, enhancing the computer's ability to discern and prioritize details.
- [Speed Estimation](../guides/speed-estimation.md) 🚀: Estimate object speed using YOLO11 and object tracking techniques, crucial for applications like autonomous vehicles and traffic monitoring.
- [Distance Calculation](../guides/distance-calculation.md) 🚀: Calculate distances between objects using [bounding box](https://www.ultralytics.com/glossary/bounding-box) centroids in YOLO11, essential for spatial analysis.
- [Queue Management](../guides/queue-management.md) 🚀: Implement efficient queue management systems to minimize wait times and improve productivity using YOLO11.
- [Parking Management](../guides/parking-management.md) 🚀: Organize and direct vehicle flow in parking areas with YOLO11, optimizing space utilization and user experience.
- [Analytics](../guides/analytics.md) 📊: Conduct comprehensive data analysis to discover patterns and make informed decisions, leveraging YOLO11 for descriptive, predictive, and prescriptive analytics.
- [Live Inference with Streamlit](../guides/streamlit-live-inference.md) 🚀: Leverage the power of YOLO11 for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) directly through your web browser with a user-friendly Streamlit interface.
- [Track Objects in Zone](../guides/trackzone.md) 🎯 NEW: Learn how to track objects within specific zones of video frames using YOLO11 for precise and efficient monitoring.

### Solutions Arguments

!!! note "Predict args"

     Solutions also support some of the arguments from `predict`, including parameters such as `conf`, `line_width`, `tracker`, `model`, `show`, `verbose` and `classes`.

{% include "macros/solutions-args.md" %}

### Usage of SolutionAnnotator

All Ultralytics Solutions use the separate class [`SolutionAnnotator`](https://docs.ultralytics.com/reference/solutions/solutions/#ultralytics.solutions.solutions.SolutionAnnotator), that extends the main [`Annotator`](https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Annotator) class, and have the following methods:

| Method                             | Return Type | Description                                                            |
| ---------------------------------- | ----------- | ---------------------------------------------------------------------- |
| `draw_region()`                    | `None`      | Draws a region using specified points, colors, and thickness.          |
| `queue_counts_display()`           | `None`      | Displays queue counts in the specified region.                         |
| `display_analytics()`              | `None`      | Displays overall statistics for parking lot management.                |
| `estimate_pose_angle()`            | `float`     | Calculates the angle between three points in an object pose.           |
| `draw_specific_points()`           | `None`      | Draws specific keypoints on the image.                                 |
| `plot_workout_information()`       | `None`      | Draws a labeled text box on the image.                                 |
| `plot_angle_and_count_and_stage()` | `None`      | Visualizes angle, step count, and stage for workout monitoring.        |
| `plot_distance_and_line()`         | `None`      | Displays the distance between centroids and connects them with a line. |
| `display_objects_labels()`         | `None`      | Annotates bounding boxes with object class labels.                     |
| `seg_bbox()`                       | `None`      | Draws contours for segmented objects and optionally labels them.       |
| `sweep_annotator()`                | `None`      | Visualizes a vertical sweep line and optional label.                   |
| `visioneye()`                      | `None`      | Maps and connects object centroids to a visual "eye" point.            |
| `circle_label()`                   | `None`      | Draws a circular label in the place of a bounding box.                 |
| `text_label()`                     | `None`      | Draws a rectangular label in the place of a bounding box.              |

### Working with SolutionResults

All Solutions calls will return a list of `SolutionResults` objects:

!!! example

    === "Count"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # region_points = [(20, 400), (1080, 400)]                                      # line counting
        region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangle region
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init ObjectCounter
        counter = solutions.ObjectCounter(
            show=True,  # display the output
            region=region_points,  # pass region points
            model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
            # classes=[0, 2],           # count specific classes i.e. person and car with COCO pretrained model.
            # tracker="botsort.yaml"    # Choose trackers i.e "bytetrack.yaml"
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = counter.count(im0)

            # Access the output
            # print(f"In count: , {results.in_count}")
            # print(f"Out count: , {results.out_count}")
            # print(f"Class wise count: , {results.classwise_count}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "Crop"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Init ObjectCropper
        cropper = solutions.ObjectCropper(
            show=True,  # display the output
            model="yolo11n.pt",  # model for object cropping i.e yolo11x.pt.
            classes=[0, 2],  # crop specific classes i.e. person and car with COCO pretrained model.
            # conf=0.5  # adjust confidence threshold for the objects.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = cropper.crop(im0)

            # Access the output
            # print(f"Total cropped objects: , {results.total_crop_objects}")

        cap.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "Blur"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("object_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init ObjectBlurrer
        blurrer = solutions.ObjectBlurrer(
            show=True,  # display the output
            model="yolo11n.pt",  # model for object blurring i.e. yolo11m.pt
            # line_width=2,   # width of bounding box.
            # classes=[0, 2], # count specific classes i.e, person and car with COCO pretrained model.
        )

        # Adjust percentage of blur intensity
        blurrer.set_blur_ratio(0.6)  # the value in range 0.1 - 1.0

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = blurrer.blur(im0)

            # Access the output
            # print(f"Total tracks: , {results.total_tracks}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "Workout"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("workouts_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init AIGym
        gym = solutions.AIGym(
            show=True,  # display the frame
            kpts=[6, 8, 10],  # keypoints for monitoring specific exercise, by default it's for pushup
            model="yolo11n-pose.pt",  # path to the YOLO11 pose estimation model file
            # line_width=2,             # adjust the line width for bounding boxes and text display
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = gym.monitor(im0)

            # Access the output
            # print(f"Workout count: , {results.workout_count}")
            # print(f"Workout angle: , {results.workout_angle}")
            # print(f"Workout stage: , {results.workout_stage}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "ZoneCounting"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Pass region as list
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

        # Pass region as dictionary
        region_points = {
            "region-01": [(50, 50), (250, 50), (250, 250), (50, 250)],
            "region-02": [(640, 640), (780, 640), (780, 720), (640, 720)],
        }

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("region_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init RegionCounter
        region = solutions.RegionCounter(
            show=True,  # display the frame
            region=region_points,  # pass region points
            model="yolo11n.pt",  # model for counting in regions i.e yolo11s.pt
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = region.count(im0)

            # Access the output
            # print(f"Region counts: , {results.region_counts}")
            # print(f"Total tracks: , {results.total_tracks}")

            video_writer.write(results.plot_im)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "AlarmSystem"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        from_email = "abc@gmail.com"  # the sender email address
        password = "---- ---- ---- ----"  # 16-digits password generated via: https://myaccount.google.com/apppasswords
        to_email = "xyz@gmail.com"  # the receiver email address

        # Init SecurityAlarm
        security = solutions.SecurityAlarm(
            show=True,  # display the output
            model="yolo11n.pt",  # i.e. yolo11s.pt, yolo11m.pt
            records=1,  # Total detections count to send an email
        )

        security.authenticate(from_email, password, to_email)  # Authenticate the email server

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = security.monitor(im0)

            # Access the output
            # print(f"Total tracks: , {results.total_tracks}")
            # print(f"Email sent status: , {results.email_sent}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "Heatmap"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # For object counting with heatmap, you can pass region points.
        # region_points = [(20, 400), (1080, 400)]                                      # line points
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # rectangle region
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon points

        # Init heatmap
        heatmap = solutions.Heatmap(
            show=True,  # Display the output
            model="yolo11n.pt",  # Path to the YOLO11 model file
            colormap=cv2.COLORMAP_PARULA,  # Colormap of heatmap
            # region=region_points,         # object counting with heatmaps, you can pass region_points
            # classes=[0, 2],               # generate heatmap for specific classes i.e person and car.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = heatmap.generate_heatmap(im0)

            # Access the output
            # print(f"Total tracks: , {results.total_tracks}")
            # print(f"In count: , {results.in_count}")
            # print(f"Out count: , {results.out_count}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "ISegment"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("isegment_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init InstanceSegmentation
        isegment = solutions.InstanceSegmentation(
            show=True,  # display the output
            model="yolo11n-seg.pt",  # model="yolo11n-seg.pt" for object segmentation using YOLO11.
            # classes=[0, 2],                   # segment specific classes i.e, person and car with pretrained model.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = isegment.segment(im0)

            # Access the output
            # print(f"Total tracks: , {results.total_tracks}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "VisionEye"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init VisionEye
        visioneye = solutions.VisionEye(
            show=True,  # display the output
            model="yolo11n.pt",  # use any model that Ultralytics support, i.e, YOLOv10
            classes=[0, 2],  # generate visioneye view for specific classes
        )

        # Adjust visioneye monitoring point explicitly
        visioneye.set_vision_point((50, 50))

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = visioneye.mapping(im0)

            # Access the output
            print(f"Total tracks: , {results.total_tracks}")

            video_writer.write(results.plot_im)  # write the video file

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "Speed"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # speed region points
        speed_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

        speed = solutions.SpeedEstimator(
            show=True,  # display the output
            model="yolo11n.pt",  # path to the YOLO11 model file.
            region=speed_region,  # pass region points
            # classes=[0, 2],           # estimate speed of specific classes.
            # line_width=2,             # Adjust the line width for bounding boxes
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = speed.estimate_speed(im0)

            # Access the output
            # print(f"Total tracks: , {results.total_tracks}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "Distance"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init DistanceCalculation
        distance = solutions.DistanceCalculation(
            model="yolo11n.pt",  # path to the YOLO11 model file.
            show=True,  # display the output
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = distance.calculate(im0)

            # Access the output
            # print(f"Pexels distance: , {results.pixels_distance}")
            # print(f"Total tracks: , {results.total_tracks}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "Queue"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Define queue points
        queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # region points
        # queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]    # polygon points

        # Init QueueManager
        queue = solutions.QueueManager(
            show=True,  # display the output
            model="yolo11n.pt",  # path to the YOLO11 model file
            region=queue_region,  # pass queue region points
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break
            results = queue.process_queue(im0)

            # Access the output
            # print(f"Queue counts: , {results.queue_count}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "ParkingManagement"

        ```python
        import cv2

        from ultralytics import solutions

        # Video capture
        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("parking management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize ParkingManagement
        parking_manager = solutions.ParkingManagement(
            model="yolo11n.pt",  # path to model file
            json_file="bounding_boxes.json",  # path to parking annotations file
        )

        while cap.isOpened():
            ret, im0 = cap.read()
            if not ret:
                break

            results = parking_manager.process_data(im0)

            # Access the output
            # print(f"Available slots: , {results.available_slots}")
            # print(f"Filled slots: , {results.filled_slots}")

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "AnalyticalGraphs"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            "analytics_output.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (1280, 720),  # This is fixed
        )

        # Init Analytics
        analytics = solutions.Analytics(
            show=True,  # display the output
            analytics_type="line",  # pass the analytics type, could be "pie", "bar" or "area".
            model="yolo11n.pt",  # path to the YOLO11 model file
            # classes=[0, 2],           # display analytics for specific detection classes
        )

        # Process video
        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if success:
                frame_count += 1
                results = analytics.process_data(im0, frame_count)  # update analytics graph every frame

                # Access the output
                # print(f"Total tracks: , {results.total_tracks}")

                out.write(results.plot_im)  # write the video file
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

    === "Web Inference"

        ```python
        from ultralytics import solutions

        inf = solutions.Inference(
            model="yolo11n.pt",  # You can use any model that Ultralytics support, i.e. YOLO11, or custom trained model
        )

        inf.inference()

        ### Make sure to run the file using command `streamlit run <file-name.py>`
        ```

    === "TrackZone"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Define region points
        region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("trackzone_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init TrackZone (Object Tracking in Zones, not complete frame)
        trackzone = solutions.TrackZone(
            show=True,  # display the output
            region=region_points,  # pass region points
            model="yolo11n.pt",  # use any model that Ultralytics support, i.e. YOLOv9, YOLOv10
            # line_width=2,             # Adjust the line width for bounding boxes and text display
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = trackzone.trackzone(im0)

            # Access the output
            # print(f"Total tracks: , {results.total_tracks}")

            video_writer.write(results.plot_im)  # write the video file

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

`SolutionResults` objects have the following attributes:

| Attribute            | Type              | Description                                       |
| -------------------- | ----------------- | ------------------------------------------------- |
| `plot_im`            | `numpy.ndarray`   | Processed image with visualized results.          |
| `in_count`           | `int, optional`   | Total number of objects entering a region.        |
| `out_count`          | `int, optional`   | Total number of objects exiting a region.         |
| `classwise_count`    | `dict, optional`  | Object count categorized by class.                |
| `queue_count`        | `int, optional`   | Number of objects in a queue.                     |
| `workout_count`      | `int, optional`   | Number of completed workout reps.                 |
| `workout_angle`      | `float, optional` | Measured joint angle in workouts.                 |
| `workout_stage`      | `str, optional`   | Current stage of a workout motion.                |
| `pixels_distance`    | `float, optional` | Distance covered by an object in pixels.          |
| `available_slots`    | `int, optional`   | Number of vacant slots in a parking area.         |
| `filled_slots`       | `int, optional`   | Number of occupied slots in a parking area.       |
| `email_sent`         | `bool, optional`  | Indicates whether an email notification was sent. |
| `total_tracks`       | `int, optional`   | Total number of tracked objects.                  |
| `region_counts`      | `int, optional`   | Object count per defined region.                  |
| `speed_dict`         | `dict, optional`  | Dictionary storing object speeds.                 |
| `total_crop_objects` | `int, optional`   | Number of cropped detected objects.               |

For more details see the [`SolutionResults` class documentation](https://docs.ultralytics.com/reference/solutions/solutions/#ultralytics.solutions.solutions.SolutionAnnotator).

### Solutions Usage via CLI

Several Ultralytics Solutions can be accessed directly through the command-line interface, including:

- Object Counting
- Heatmap
- Queue Management
- Speed Estimation
- Workout Monitoring
- TrackZone
- Analytics

!!! tip "Command Info"

    `yolo SOLUTIONS SOLUTION_NAME ARGS`

    - **SOLUTIONS** is a required keyword.
    - **SOLUTION_NAME** (optional) is one of: `['count', 'heatmap', 'queue', 'speed', 'workout', 'analytics', 'trackzone']`.
    - **ARGS** (optional) are custom `arg=value` pairs, such as `show_in=True`, to override default settings.

```bash
yolo solutions count show=True  # for object counting

yolo solutions source="path/to/video/file.mp4"  # specify video file path
```

### Contribute to Our Solutions

We welcome contributions from the community! If you've mastered a particular aspect of Ultralytics YOLO that's not yet covered in our solutions, we encourage you to share your expertise. Writing a guide is a great way to give back to the community and help us make our documentation more comprehensive and user-friendly.

To get started, please read our [Contributing Guide](../help/contributing.md) for guidelines on how to open up a Pull Request (PR) 🛠️. We look forward to your contributions!

Let's work together to make the Ultralytics YOLO ecosystem more robust and versatile 🙏!

## FAQ

### How can I use Ultralytics YOLO for real-time object counting?

Ultralytics YOLO11 can be used for real-time object counting by leveraging its advanced object detection capabilities. You can follow our detailed guide on [Object Counting](../guides/object-counting.md) to set up YOLO11 for live video stream analysis. Simply install YOLO11, load your model, and process video frames to count objects dynamically.

### What are the benefits of using Ultralytics YOLO for security systems?

Ultralytics YOLO11 enhances security systems by offering real-time object detection and alert mechanisms. By employing YOLO11, you can create a security alarm system that triggers alerts when new objects are detected in the surveillance area. Learn how to set up a [Security Alarm System](../guides/security-alarm-system.md) with YOLO11 for robust security monitoring.

### How can Ultralytics YOLO improve queue management systems?

Ultralytics YOLO11 can significantly improve queue management systems by accurately counting and tracking people in queues, thus helping to reduce wait times and optimize service efficiency. Follow our detailed guide on [Queue Management](../guides/queue-management.md) to learn how to implement YOLO11 for effective queue monitoring and analysis.

### Can Ultralytics YOLO be used for workout monitoring?

Yes, Ultralytics YOLO11 can be effectively used for monitoring workouts by tracking and analyzing fitness routines in real-time. This allows for precise evaluation of exercise form and performance. Explore our guide on [Workouts Monitoring](../guides/workouts-monitoring.md) to learn how to set up an AI-powered workout monitoring system using YOLO11.

### How does Ultralytics YOLO help in creating heatmaps for [data visualization](https://www.ultralytics.com/glossary/data-visualization)?

Ultralytics YOLO11 can generate heatmaps to visualize data intensity across a given area, highlighting regions of high activity or interest. This feature is particularly useful in understanding patterns and trends in various computer vision tasks. Learn more about creating and using [Heatmaps](../guides/heatmaps.md) with YOLO11 for comprehensive data analysis and visualization.
