---
comments: true
description: Discover how to use YOLOv8 predict mode for various tasks. Learn about different inference sources like images, videos, and data formats.
keywords: Ultralytics, YOLOv8, predict mode, inference sources, prediction tasks, streaming mode, image processing, video processing, machine learning, AI
---

# Model Prediction with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

In the world of machine learning and computer vision, the process of making sense out of visual data is called 'inference' or 'prediction'. Ultralytics YOLOv8 offers a powerful feature known as **predict mode** that is tailored for high-performance, real-time inference on a wide range of data sources.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Extract the Outputs from Ultralytics YOLOv8 Model for Custom Projects.
</p>

## Real-world Applications

|                   Manufacturing                   |                        Sports                        |                   Safety                    |
|:-------------------------------------------------:|:----------------------------------------------------:|:-------------------------------------------:|
| ![Vehicle Spare Parts Detection][car spare parts] | ![Football Player Detection][football player detect] | ![People Fall Detection][human fall detect] |
| Vehicle Spare Parts Detection                     |  Football Player Detection                           | People Fall Detection                       |

## Why Use Ultralytics YOLO for Inference?

Here's why you should consider YOLOv8's predict mode for your various inference needs:

- **Versatility:** Capable of making inferences on images, videos, and even live streams.
- **Performance:** Engineered for real-time, high-speed processing without sacrificing accuracy.
- **Ease of Use:** Intuitive Python and CLI interfaces for rapid deployment and testing.
- **Highly Customizable:** Various settings and parameters to tune the model's inference behavior according to your specific requirements.

### Key Features of Predict Mode

YOLOv8's predict mode is designed to be robust and versatile, featuring:

- **Multiple Data Source Compatibility:** Whether your data is in the form of individual images, a collection of images, video files, or real-time video streams, predict mode has you covered.
- **Streaming Mode:** Use the streaming feature to generate a memory-efficient generator of `Results` objects. Enable this by setting `stream=True` in the predictor's call method.
- **Batch Processing:** The ability to process multiple images or video frames in a single batch, further speeding up inference time.
- **Integration Friendly:** Easily integrate with existing data pipelines and other software components, thanks to its flexible API.

Ultralytics YOLO models return either a Python list of `Results` objects, or a memory-efficient Python generator of `Results` objects when `stream=True` is passed to the model during inference:

!!! example "Predict"

    === "Return a list with `stream=False`"
        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

        # Run batched inference on a list of images
        results = model(['im1.jpg', 'im2.jpg'])  # return a list of Results objects

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
        ```

    === "Return a generator with `stream=True`"
        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

        # Run batched inference on a list of images
        results = model(['im1.jpg', 'im2.jpg'], stream=True)  # return a generator of Results objects

        # Process results generator
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
        ```

## Inference Sources

YOLOv8 can process different types of input sources for inference, as shown in the table below. The sources include static images, video streams, and various data formats. The table also indicates whether each source can be used in streaming mode with the argument `stream=True` ✅. Streaming mode is beneficial for processing videos or live streams as it creates a generator of results instead of loading all frames into memory.

!!! tip "Tip"

    Use `stream=True` for processing long videos or large datasets to efficiently manage memory. When `stream=False`, the results for all frames or data points are stored in memory, which can quickly add up and cause out-of-memory errors for large inputs. In contrast, `stream=True` utilizes a generator, which only keeps the results of the current frame or data point in memory, significantly reducing memory consumption and preventing out-of-memory issues.

| Source         | Argument                                   | Type            | Notes                                                                                       |
|----------------|--------------------------------------------|-----------------|---------------------------------------------------------------------------------------------|
| image          | `'image.jpg'`                              | `str` or `Path` | Single image file.                                                                          |
| URL            | `'https://ultralytics.com/images/bus.jpg'` | `str`           | URL to an image.                                                                            |
| screenshot     | `'screen'`                                 | `str`           | Capture a screenshot.                                                                       |
| PIL            | `Image.open('im.jpg')`                     | `PIL.Image`     | HWC format with RGB channels.                                                               |
| OpenCV         | `cv2.imread('im.jpg')`                     | `np.ndarray`    | HWC format with BGR channels `uint8 (0-255)`.                                               |
| numpy          | `np.zeros((640,1280,3))`                   | `np.ndarray`    | HWC format with BGR channels `uint8 (0-255)`.                                               |
| torch          | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | BCHW format with RGB channels `float32 (0.0-1.0)`.                                          |
| CSV            | `'sources.csv'`                            | `str` or `Path` | CSV file containing paths to images, videos, or directories.                                |
| video ✅        | `'video.mp4'`                              | `str` or `Path` | Video file in formats like MP4, AVI, etc.                                                   |
| directory ✅    | `'path/'`                                  | `str` or `Path` | Path to a directory containing images or videos.                                            |
| glob ✅         | `'path/*.jpg'`                             | `str`           | Glob pattern to match multiple files. Use the `*` character as a wildcard.                  |
| YouTube ✅      | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | URL to a YouTube video.                                                                     |
| stream ✅       | `'rtsp://example.com/media.mp4'`           | `str`           | URL for streaming protocols such as RTSP, RTMP, TCP, or an IP address.                      |
| multi-stream ✅ | `'list.streams'`                           | `str` or `Path` | `*.streams` text file with one stream URL per row, i.e. 8 streams will run at batch-size 8. |

Below are code examples for using each source type:

!!! example "Prediction sources"

    === "image"
        Run inference on an image file.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Define path to the image file
        source = 'path/to/image.jpg'

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "screenshot"
        Run inference on the current screen content as a screenshot.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Define current screenshot as source
        source = 'screen'

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "URL"
        Run inference on an image or video hosted remotely via URL.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Define remote image or video URL
        source = 'https://ultralytics.com/images/bus.jpg'

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "PIL"
        Run inference on an image opened with Python Imaging Library (PIL).
        ```python
        from PIL import Image
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Open an image using PIL
        source = Image.open('path/to/image.jpg')

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "OpenCV"
        Run inference on an image read with OpenCV.
        ```python
        import cv2
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Read an image using OpenCV
        source = cv2.imread('path/to/image.jpg')

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "numpy"
        Run inference on an image represented as a numpy array.
        ```python
        import numpy as np
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Create a random numpy array of HWC shape (640, 640, 3) with values in range [0, 255] and type uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

[car spare parts]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1
[football player detect]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442
[human fall detect]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43
