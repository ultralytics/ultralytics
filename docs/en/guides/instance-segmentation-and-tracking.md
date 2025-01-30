---
comments: true
description: Master instance segmentation and tracking with Ultralytics YOLO11. Learn techniques for precise object identification and tracking.
keywords: instance segmentation, tracking, YOLO11, Ultralytics, object detection, machine learning, computer vision, python
---

# Instance Segmentation and Tracking using Ultralytics YOLO11 🚀

## What is [Instance Segmentation](https://www.ultralytics.com/glossary/instance-segmentation)?

[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) instance segmentation involves identifying and outlining individual objects in an image, providing a detailed understanding of spatial distribution. Unlike [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation), it uniquely labels and precisely delineates each object, crucial for tasks like [object detection](https://www.ultralytics.com/glossary/object-detection) and medical imaging.

There are two types of instance segmentation tracking available in the Ultralytics package:

- **Instance Segmentation with Class Objects:** Each class object is assigned a unique color for clear visual separation.

- **Instance Segmentation with Object Tracks:** Every track is represented by a distinct color, facilitating easy identification and tracking.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/75G_S1Ngji8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Instance Segmentation with Object Tracking using Ultralytics YOLO11
</p>

## Samples

|                                                        Instance Segmentation                                                         |                                                                  Instance Segmentation + Object Tracking                                                                  |
| :----------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics Instance Segmentation](https://github.com/ultralytics/docs/releases/download/0/ultralytics-instance-segmentation.avif) | ![Ultralytics Instance Segmentation with Object Tracking](https://github.com/ultralytics/docs/releases/download/0/ultralytics-instance-segmentation-object-tracking.avif) |
|                                                 Ultralytics Instance Segmentation 😍                                                 |                                                         Ultralytics Instance Segmentation with Object Tracking 🔥                                                         |

!!! example "Instance Segmentation and Tracking"

    === "Instance Segmentation"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init InstanceSegmentation
        isegment = solutions.InstanceSegmentation(
            show=True,  # Display the output
            model="yolo11n-seg.pt",  # model="yolo11n-seg.pt" for object segmentation using YOLO11.
            line_width=4,  # Width of segmentation mask.
            # classes=[0, 2],  # If you want to segment specific classes i.e, person and car with COCO pretrained model.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            results = isegment.segment(im0)
            video_writer.write(results["im0"])

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### `seg_bbox` Arguments

| Name         | Type    | Default         | Description                                  |
| ------------ | ------- | --------------- | -------------------------------------------- |
| `mask`       | `array` | `None`          | Segmentation mask coordinates                |
| `mask_color` | `RGB`   | `(255, 0, 255)` | Mask color for every segmented box           |
| `label`      | `str`   | `None`          | Label for segmented object                   |
| `txt_color`  | `RGB`   | `None`          | Label color for segmented and tracked object |

## Note

For any inquiries, feel free to post your questions in the [Ultralytics Issue Section](https://github.com/ultralytics/ultralytics/issues/new/choose) or the discussion section mentioned below.

## FAQ

### How do I perform instance segmentation using Ultralytics YOLO11?

To perform instance segmentation using Ultralytics YOLO11, initialize the YOLO model with a segmentation version of YOLO11 and process video frames through it. Here's a simplified code example:

!!! example

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init InstanceSegmentation
        isegment = solutions.InstanceSegmentation(
            show=True,  # Display the output
            model="yolo11n-seg.pt",  # model="yolo11n-seg.pt" for object segmentation using YOLO11.
            line_width=4,  # Width of segmentation mask.
            # classes=[0, 2],  # If you want to segment specific classes i.e, person and car with COCO pretrained model.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            results = isegment.segment(im0)
            video_writer.write(results["im0"])

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

Learn more about instance segmentation in the [Ultralytics YOLO11 guide](#what-is-instance-segmentation).

### What is the difference between instance segmentation and object tracking in Ultralytics YOLO11?

Instance segmentation identifies and outlines individual objects within an image, giving each object a unique label and mask. Object tracking extends this by assigning consistent labels to objects across video frames, facilitating continuous tracking of the same objects over time. Learn more about the distinctions in the [Ultralytics YOLO11 documentation](#samples).

### Why should I use Ultralytics YOLO11 for instance segmentation and tracking over other models like Mask R-CNN or Faster R-CNN?

Ultralytics YOLO11 offers real-time performance, superior [accuracy](https://www.ultralytics.com/glossary/accuracy), and ease of use compared to other models like Mask R-CNN or Faster R-CNN. YOLO11 provides a seamless integration with Ultralytics HUB, allowing users to manage models, datasets, and training pipelines efficiently. Discover more about the benefits of YOLO11 in the [Ultralytics blog](https://www.ultralytics.com/blog/introducing-ultralytics-yolov8).

### Are there any datasets provided by Ultralytics suitable for training YOLO11 models for instance segmentation and tracking?

Yes, Ultralytics offers several datasets suitable for training YOLO11 models, including segmentation and tracking datasets. Dataset examples, structures, and instructions for use can be found in the [Ultralytics Datasets documentation](https://docs.ultralytics.com/datasets/).
