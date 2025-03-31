---
comments: true
description: Master instance segmentation and tracking with Ultralytics YOLO11. Learn techniques for precise object identification and tracking.
keywords: instance segmentation, tracking, YOLO11, Ultralytics, object detection, machine learning, computer vision, python
---

# Instance Segmentation and Tracking using Ultralytics YOLO11 🚀

## What is Instance Segmentation?

[Instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) is a computer vision task that involves identifying and outlining individual objects in an image at the pixel level. Unlike [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) which only classifies pixels by category, instance segmentation uniquely labels and precisely delineates each object instance, making it crucial for applications requiring detailed spatial understanding like medical imaging, autonomous driving, and industrial automation.

[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) provides powerful instance segmentation capabilities that enable precise object boundary detection while maintaining the speed and efficiency YOLO models are known for.

There are two types of instance segmentation tracking available in the Ultralytics package:

- **Instance Segmentation with Class Objects:** Each class object is assigned a unique color for clear visual separation.

- **Instance Segmentation with Object Tracks:** Every track is represented by a distinct color, facilitating easy identification and tracking across video frames.

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

!!! example "Instance segmentation using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Instance segmentation using Ultralytics YOLO11
        yolo solutions isegment show=True

        # Pass a source video
        yolo solutions isegment source="path/to/video.mp4"

        # Monitor the specific classes
        yolo solutions isegment classes="[0, 5]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("isegment_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize instance segmentation object
        isegment = solutions.InstanceSegmentation(
            show=True,  # display the output
            model="yolo11n-seg.pt",  # model="yolo11n-seg.pt" for object segmentation using YOLO11.
            # classes=[0, 2],  # segment specific classes i.e, person and car with pretrained model.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = isegment(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `InstanceSegmentation` Arguments

Here's a table with the `InstanceSegmentation` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

You can also take advantage of `track` arguments within the `InstanceSegmentation` solution:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization arguments are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## Applications of Instance Segmentation

Instance segmentation with YOLO11 has numerous real-world applications across various industries:

### Waste Management and Recycling

YOLO11 can be used in [waste management facilities](https://www.ultralytics.com/blog/simplifying-e-waste-management-with-ai-innovations) to identify and sort different types of materials. The model can segment plastic waste, cardboard, metal, and other recyclables with high precision, enabling automated sorting systems to process waste more efficiently. This is particularly valuable considering that only about 10% of the 7 billion tonnes of plastic waste generated globally gets recycled.

### Autonomous Vehicles

In [self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive), instance segmentation helps identify and track pedestrians, vehicles, traffic signs, and other road elements at the pixel level. This precise understanding of the environment is crucial for navigation and safety decisions. YOLO11's real-time performance makes it ideal for these time-sensitive applications.

### Medical Imaging

Instance segmentation can identify and outline tumors, organs, or cellular structures in medical scans. YOLO11's ability to precisely delineate object boundaries makes it valuable for [medical diagnostics](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency) and treatment planning.

### Construction Site Monitoring

At construction sites, instance segmentation can track heavy machinery, workers, and materials. This helps ensure safety by monitoring equipment positions and detecting when workers enter hazardous areas, while also optimizing workflow and resource allocation.

## Note

For any inquiries, feel free to post your questions in the [Ultralytics Issue Section](https://github.com/ultralytics/ultralytics/issues/new/choose) or the discussion section mentioned below.

## FAQ

### How do I perform instance segmentation using Ultralytics YOLO11?

To perform instance segmentation using Ultralytics YOLO11, initialize the YOLO model with a segmentation version of YOLO11 and process video frames through it. Here's a simplified code example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init InstanceSegmentation
isegment = solutions.InstanceSegmentation(
    show=True,  # display the output
    model="yolo11n-seg.pt",  # model="yolo11n-seg.pt" for object segmentation using YOLO11.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = isegment(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

Learn more about instance segmentation in the [Ultralytics YOLO11 guide](https://docs.ultralytics.com/tasks/segment/).

### What is the difference between instance segmentation and object tracking in Ultralytics YOLO11?

Instance segmentation identifies and outlines individual objects within an image, giving each object a unique label and mask. Object tracking extends this by assigning consistent IDs to objects across video frames, facilitating continuous tracking of the same objects over time. When combined, as in YOLO11's implementation, you get powerful capabilities for analyzing object movement and behavior in videos while maintaining precise boundary information.

### Why should I use Ultralytics YOLO11 for instance segmentation and tracking over other models like Mask R-CNN or Faster R-CNN?

Ultralytics YOLO11 offers real-time performance, superior [accuracy](https://www.ultralytics.com/glossary/accuracy), and ease of use compared to other models like Mask R-CNN or Faster R-CNN. YOLO11 processes images in a single pass (one-stage detection), making it significantly faster while maintaining high precision. It also provides seamless integration with [Ultralytics HUB](https://www.ultralytics.com/hub), allowing users to manage models, datasets, and training pipelines efficiently. For applications requiring both speed and accuracy, YOLO11 provides an optimal balance.

### Are there any datasets provided by Ultralytics suitable for training YOLO11 models for instance segmentation and tracking?

Yes, Ultralytics offers several datasets suitable for training YOLO11 models for instance segmentation, including [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/), [COCO8-Seg](https://docs.ultralytics.com/datasets/segment/coco8-seg/) (a smaller subset for quick testing), [Package-Seg](https://docs.ultralytics.com/datasets/segment/package-seg/), and [Crack-Seg](https://docs.ultralytics.com/datasets/segment/crack-seg/). These datasets come with pixel-level annotations needed for instance segmentation tasks. For more specialized applications, you can also create custom datasets following the Ultralytics format. Complete dataset information and usage instructions can be found in the [Ultralytics Datasets documentation](https://docs.ultralytics.com/datasets/).
