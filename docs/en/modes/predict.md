---
comments: true
description: Harness the power of Ultralytics YOLOv8 for real-time, high-speed inference on various data sources. Learn about predict mode, key features, and practical applications.
keywords: Ultralytics, YOLOv8, model prediction, inference, predict mode, real-time inference, computer vision, machine learning, streaming, high performance
---

# Model Prediction with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

In the world of [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv), the process of making sense out of visual data is called 'inference' or 'prediction'. Ultralytics YOLOv8 offers a powerful feature known as **predict mode** that is tailored for high-performance, real-time inference on a wide range of data sources.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Extract the Outputs from Ultralytics YOLOv8 Model for Custom Projects.
</p>

## Real-world Applications

|                   Manufacturing                   |                        Sports                        |                   Safety                    |
| :-----------------------------------------------: | :--------------------------------------------------: | :-----------------------------------------: |
| ![Vehicle Spare Parts Detection][car spare parts] | ![Football Player Detection][football player detect] | ![People Fall Detection][human fall detect] |
|           Vehicle Spare Parts Detection           |              Football Player Detection               |            People Fall Detection            |

## Why Use Ultralytics YOLO for Inference?

Here's why you should consider YOLOv8's predict mode for your various inference needs:

- **Versatility:** Capable of making inferences on images, videos, and even live streams.
- **Performance:** Engineered for real-time, high-speed processing without sacrificing [accuracy](https://www.ultralytics.com/glossary/accuracy).
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
        model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

        # Run batched inference on a list of images
        results = model(["image1.jpg", "image2.jpg"])  # return a list of Results objects

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk
        ```

    === "Return a generator with `stream=True`"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

        # Run batched inference on a list of images
        results = model(["image1.jpg", "image2.jpg"], stream=True)  # return a generator of Results objects

        # Process results generator
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk
        ```

## Inference Sources

YOLOv8 can process different types of input sources for inference, as shown in the table below. The sources include static images, video streams, and various data formats. The table also indicates whether each source can be used in streaming mode with the argument `stream=True` ✅. Streaming mode is beneficial for processing videos or live streams as it creates a generator of results instead of loading all frames into memory.

!!! tip

    Use `stream=True` for processing long videos or large datasets to efficiently manage memory. When `stream=False`, the results for all frames or data points are stored in memory, which can quickly add up and cause out-of-memory errors for large inputs. In contrast, `stream=True` utilizes a generator, which only keeps the results of the current frame or data point in memory, significantly reducing memory consumption and preventing out-of-memory issues.

| Source                                                | Example                                    | Type            | Notes                                                                                       |
| ----------------------------------------------------- | ------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------- |
| image                                                 | `'image.jpg'`                              | `str` or `Path` | Single image file.                                                                          |
| URL                                                   | `'https://ultralytics.com/images/bus.jpg'` | `str`           | URL to an image.                                                                            |
| screenshot                                            | `'screen'`                                 | `str`           | Capture a screenshot.                                                                       |
| PIL                                                   | `Image.open('image.jpg')`                  | `PIL.Image`     | HWC format with RGB channels.                                                               |
| [OpenCV](https://www.ultralytics.com/glossary/opencv) | `cv2.imread('image.jpg')`                  | `np.ndarray`    | HWC format with BGR channels `uint8 (0-255)`.                                               |
| numpy                                                 | `np.zeros((640,1280,3))`                   | `np.ndarray`    | HWC format with BGR channels `uint8 (0-255)`.                                               |
| torch                                                 | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | BCHW format with RGB channels `float32 (0.0-1.0)`.                                          |
| CSV                                                   | `'sources.csv'`                            | `str` or `Path` | CSV file containing paths to images, videos, or directories.                                |
| video ✅                                              | `'video.mp4'`                              | `str` or `Path` | Video file in formats like MP4, AVI, etc.                                                   |
| directory ✅                                          | `'path/'`                                  | `str` or `Path` | Path to a directory containing images or videos.                                            |
| glob ✅                                               | `'path/*.jpg'`                             | `str`           | Glob pattern to match multiple files. Use the `*` character as a wildcard.                  |
| YouTube ✅                                            | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | URL to a YouTube video.                                                                     |
| stream ✅                                             | `'rtsp://example.com/media.mp4'`           | `str`           | URL for streaming protocols such as RTSP, RTMP, TCP, or an IP address.                      |
| multi-stream ✅                                       | `'list.streams'`                           | `str` or `Path` | `*.streams` text file with one stream URL per row, i.e. 8 streams will run at batch-size 8. |

Below are code examples for using each source type:

!!! example "Prediction sources"

    === "image"

        Run inference on an image file.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Define path to the image file
        source = "path/to/image.jpg"

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "screenshot"

        Run inference on the current screen content as a screenshot.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Define current screenshot as source
        source = "screen"

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "URL"

        Run inference on an image or video hosted remotely via URL.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Define remote image or video URL
        source = "https://ultralytics.com/images/bus.jpg"

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "PIL"

        Run inference on an image opened with Python Imaging Library (PIL).
        ```python
        from PIL import Image

        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Open an image using PIL
        source = Image.open("path/to/image.jpg")

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "OpenCV"

        Run inference on an image read with OpenCV.
        ```python
        import cv2

        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Read an image using OpenCV
        source = cv2.imread("path/to/image.jpg")

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "numpy"

        Run inference on an image represented as a numpy array.
        ```python
        import numpy as np

        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Create a random numpy array of HWC shape (640, 640, 3) with values in range [0, 255] and type uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype="uint8")

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "torch"

        Run inference on an image represented as a [PyTorch](https://www.ultralytics.com/glossary/pytorch) tensor.
        ```python
        import torch

        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Create a random torch tensor of BCHW shape (1, 3, 640, 640) with values in range [0, 1] and type float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "CSV"

        Run inference on a collection of images, URLs, videos and directories listed in a CSV file.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Define a path to a CSV file with images, URLs, videos and directories
        source = "path/to/file.csv"

        # Run inference on the source
        results = model(source)  # list of Results objects
        ```

    === "video"

        Run inference on a video file. By using `stream=True`, you can create a generator of Results objects to reduce memory usage.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Define path to video file
        source = "path/to/video.mp4"

        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```

    === "directory"

        Run inference on all images and videos in a directory. To also capture images and videos in subdirectories use a glob pattern, i.e. `path/to/dir/**/*`.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Define path to directory containing images and videos for inference
        source = "path/to/dir"

        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```

    === "glob"

        Run inference on all images and videos that match a glob expression with `*` characters.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Define a glob search for all JPG files in a directory
        source = "path/to/dir/*.jpg"

        # OR define a recursive glob search for all JPG files including subdirectories
        source = "path/to/dir/**/*.jpg"

        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```

    === "YouTube"

        Run inference on a YouTube video. By using `stream=True`, you can create a generator of Results objects to reduce memory usage for long videos.
        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Define source as YouTube video URL
        source = "https://youtu.be/LNwODJXcvt4"

        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```

    === "Stream"

        Use the stream mode to run inference on live video streams using RTSP, RTMP, TCP, or IP address protocols. If a single stream is provided, the model runs inference with a [batch size](https://www.ultralytics.com/glossary/batch-size) of 1. For multiple streams, a `.streams` text file can be used to perform batched inference, where the batch size is determined by the number of streams provided (e.g., batch-size 8 for 8 streams).

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Single stream with batch-size 1 inference
        source = "rtsp://example.com/media.mp4"  # RTSP, RTMP, TCP, or IP streaming address

        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```

        For single stream usage, the batch size is set to 1 by default, allowing efficient real-time processing of the video feed.

    === "Multi-Stream"

        To handle multiple video streams simultaneously, use a `.streams` text file containing the streaming sources. The model will run batched inference where the batch size equals the number of streams. This setup enables efficient processing of multiple feeds concurrently.

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Multiple streams with batched inference (e.g., batch-size 8 for 8 streams)
        source = "path/to/list.streams"  # *.streams text file with one streaming address per line

        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```

        Example `.streams` text file:

        ```txt
        rtsp://example.com/media1.mp4
        rtsp://example.com/media2.mp4
        rtmp://example2.com/live
        tcp://192.168.1.100:554
        ...
        ```

        Each row in the file represents a streaming source, allowing you to monitor and perform inference on several video streams at once.

## Inference Arguments

`model.predict()` accepts multiple arguments that can be passed at inference time to override defaults:

!!! example

    ```python
    from ultralytics import YOLO

    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Run inference on 'bus.jpg' with arguments
    model.predict("bus.jpg", save=True, imgsz=320, conf=0.5)
    ```

Inference arguments:

{% include "macros/predict-args.md" %}

Visualization arguments:

{% include "macros/visualization-args.md" %}

## Image and Video Formats

YOLOv8 supports various image and video formats, as specified in [ultralytics/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py). See the tables below for the valid suffixes and example predict commands.

### Images

The below table contains valid Ultralytics image formats.

| Image Suffixes | Example Predict Command          | Reference                                                                  |
| -------------- | -------------------------------- | -------------------------------------------------------------------------- |
| `.bmp`         | `yolo predict source=image.bmp`  | [Microsoft BMP File Format](https://en.wikipedia.org/wiki/BMP_file_format) |
| `.dng`         | `yolo predict source=image.dng`  | [Adobe DNG](https://en.wikipedia.org/wiki/Digital_Negative)                |
| `.jpeg`        | `yolo predict source=image.jpeg` | [JPEG](https://en.wikipedia.org/wiki/JPEG)                                 |
| `.jpg`         | `yolo predict source=image.jpg`  | [JPEG](https://en.wikipedia.org/wiki/JPEG)                                 |
| `.mpo`         | `yolo predict source=image.mpo`  | [Multi Picture Object](https://fileinfo.com/extension/mpo)                 |
| `.png`         | `yolo predict source=image.png`  | [Portable Network Graphics](https://en.wikipedia.org/wiki/PNG)             |
| `.tif`         | `yolo predict source=image.tif`  | [Tag Image File Format](https://en.wikipedia.org/wiki/TIFF)                |
| `.tiff`        | `yolo predict source=image.tiff` | [Tag Image File Format](https://en.wikipedia.org/wiki/TIFF)                |
| `.webp`        | `yolo predict source=image.webp` | [WebP](https://en.wikipedia.org/wiki/WebP)                                 |
| `.pfm`         | `yolo predict source=image.pfm`  | [Portable FloatMap](https://en.wikipedia.org/wiki/Netpbm#File_formats)     |

### Videos

The below table contains valid Ultralytics video formats.

| Video Suffixes | Example Predict Command          | Reference                                                                        |
| -------------- | -------------------------------- | -------------------------------------------------------------------------------- |
| `.asf`         | `yolo predict source=video.asf`  | [Advanced Systems Format](https://en.wikipedia.org/wiki/Advanced_Systems_Format) |
| `.avi`         | `yolo predict source=video.avi`  | [Audio Video Interleave](https://en.wikipedia.org/wiki/Audio_Video_Interleave)   |
| `.gif`         | `yolo predict source=video.gif`  | [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF)                 |
| `.m4v`         | `yolo predict source=video.m4v`  | [MPEG-4 Part 14](https://en.wikipedia.org/wiki/M4V)                              |
| `.mkv`         | `yolo predict source=video.mkv`  | [Matroska](https://en.wikipedia.org/wiki/Matroska)                               |
| `.mov`         | `yolo predict source=video.mov`  | [QuickTime File Format](https://en.wikipedia.org/wiki/QuickTime_File_Format)     |
| `.mp4`         | `yolo predict source=video.mp4`  | [MPEG-4 Part 14 - Wikipedia](https://en.wikipedia.org/wiki/MPEG-4_Part_14)       |
| `.mpeg`        | `yolo predict source=video.mpeg` | [MPEG-1 Part 2](https://en.wikipedia.org/wiki/MPEG-1)                            |
| `.mpg`         | `yolo predict source=video.mpg`  | [MPEG-1 Part 2](https://en.wikipedia.org/wiki/MPEG-1)                            |
| `.ts`          | `yolo predict source=video.ts`   | [MPEG Transport Stream](https://en.wikipedia.org/wiki/MPEG_transport_stream)     |
| `.wmv`         | `yolo predict source=video.wmv`  | [Windows Media Video](https://en.wikipedia.org/wiki/Windows_Media_Video)         |
| `.webm`        | `yolo predict source=video.webm` | [WebM Project](https://en.wikipedia.org/wiki/WebM)                               |

## Working with Results

All Ultralytics `predict()` calls will return a list of `Results` objects:

!!! example "Results"

    ```python
    from ultralytics import YOLO

    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Run inference on an image
    results = model("bus.jpg")  # list of 1 Results object
    results = model(["bus.jpg", "zidane.jpg"])  # list of 2 Results objects
    ```

`Results` objects have the following attributes:

| Attribute    | Type                  | Description                                                                              |
| ------------ | --------------------- | ---------------------------------------------------------------------------------------- |
| `orig_img`   | `numpy.ndarray`       | The original image as a numpy array.                                                     |
| `orig_shape` | `tuple`               | The original image shape in (height, width) format.                                      |
| `boxes`      | `Boxes, optional`     | A Boxes object containing the detection bounding boxes.                                  |
| `masks`      | `Masks, optional`     | A Masks object containing the detection masks.                                           |
| `probs`      | `Probs, optional`     | A Probs object containing probabilities of each class for classification task.           |
| `keypoints`  | `Keypoints, optional` | A Keypoints object containing detected keypoints for each object.                        |
| `obb`        | `OBB, optional`       | An OBB object containing oriented bounding boxes.                                        |
| `speed`      | `dict`                | A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image. |
| `names`      | `dict`                | A dictionary of class names.                                                             |
| `path`       | `str`                 | The path to the image file.                                                              |

`Results` objects have the following methods:

| Method        | Return Type     | Description                                                                         |
| ------------- | --------------- | ----------------------------------------------------------------------------------- |
| `update()`    | `None`          | Update the boxes, masks, and probs attributes of the Results object.                |
| `cpu()`       | `Results`       | Return a copy of the Results object with all tensors on CPU memory.                 |
| `numpy()`     | `Results`       | Return a copy of the Results object with all tensors as numpy arrays.               |
| `cuda()`      | `Results`       | Return a copy of the Results object with all tensors on GPU memory.                 |
| `to()`        | `Results`       | Return a copy of the Results object with tensors on the specified device and dtype. |
| `new()`       | `Results`       | Return a new Results object with the same image, path, and names.                   |
| `plot()`      | `numpy.ndarray` | Plots the detection results. Returns a numpy array of the annotated image.          |
| `show()`      | `None`          | Show annotated results to screen.                                                   |
| `save()`      | `None`          | Save annotated results to file.                                                     |
| `verbose()`   | `str`           | Return log string for each task.                                                    |
| `save_txt()`  | `None`          | Save predictions into a txt file.                                                   |
| `save_crop()` | `None`          | Save cropped predictions to `save_dir/cls/file_name.jpg`.                           |
| `tojson()`    | `str`           | Convert the object to JSON format.                                                  |

For more details see the [`Results` class documentation](../reference/engine/results.md).

### Boxes

`Boxes` object can be used to index, manipulate, and convert bounding boxes to different formats.

!!! example "Boxes"

    ```python
    from ultralytics import YOLO

    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Run inference on an image
    results = model("bus.jpg")  # results list

    # View results
    for r in results:
        print(r.boxes)  # print the Boxes object containing the detection bounding boxes
    ```

Here is a table for the `Boxes` class methods and properties, including their name, type, and description:

| Name      | Type                      | Description                                                        |
| --------- | ------------------------- | ------------------------------------------------------------------ |
| `cpu()`   | Method                    | Move the object to CPU memory.                                     |
| `numpy()` | Method                    | Convert the object to a numpy array.                               |
| `cuda()`  | Method                    | Move the object to CUDA memory.                                    |
| `to()`    | Method                    | Move the object to the specified device.                           |
| `xyxy`    | Property (`torch.Tensor`) | Return the boxes in xyxy format.                                   |
| `conf`    | Property (`torch.Tensor`) | Return the confidence values of the boxes.                         |
| `cls`     | Property (`torch.Tensor`) | Return the class values of the boxes.                              |
| `id`      | Property (`torch.Tensor`) | Return the track IDs of the boxes (if available).                  |
| `xywh`    | Property (`torch.Tensor`) | Return the boxes in xywh format.                                   |
| `xyxyn`   | Property (`torch.Tensor`) | Return the boxes in xyxy format normalized by original image size. |
| `xywhn`   | Property (`torch.Tensor`) | Return the boxes in xywh format normalized by original image size. |

For more details see the [`Boxes` class documentation](../reference/engine/results.md#ultralytics.engine.results.Boxes).

### Masks

`Masks` object can be used index, manipulate and convert masks to segments.

!!! example "Masks"

    ```python
    from ultralytics import YOLO

    # Load a pretrained YOLOv8n-seg Segment model
    model = YOLO("yolov8n-seg.pt")

    # Run inference on an image
    results = model("bus.jpg")  # results list

    # View results
    for r in results:
        print(r.masks)  # print the Masks object containing the detected instance masks
    ```

Here is a table for the `Masks` class methods and properties, including their name, type, and description:

| Name      | Type                      | Description                                                     |
| --------- | ------------------------- | --------------------------------------------------------------- |
| `cpu()`   | Method                    | Returns the masks tensor on CPU memory.                         |
| `numpy()` | Method                    | Returns the masks tensor as a numpy array.                      |
| `cuda()`  | Method                    | Returns the masks tensor on GPU memory.                         |
| `to()`    | Method                    | Returns the masks tensor with the specified device and dtype.   |
| `xyn`     | Property (`torch.Tensor`) | A list of normalized segments represented as tensors.           |
| `xy`      | Property (`torch.Tensor`) | A list of segments in pixel coordinates represented as tensors. |

For more details see the [`Masks` class documentation](../reference/engine/results.md#ultralytics.engine.results.Masks).

### Keypoints

`Keypoints` object can be used index, manipulate and normalize coordinates.

!!! example "Keypoints"

    ```python
    from ultralytics import YOLO

    # Load a pretrained YOLOv8n-pose Pose model
    model = YOLO("yolov8n-pose.pt")

    # Run inference on an image
    results = model("bus.jpg")  # results list

    # View results
    for r in results:
        print(r.keypoints)  # print the Keypoints object containing the detected keypoints
    ```

Here is a table for the `Keypoints` class methods and properties, including their name, type, and description:

| Name      | Type                      | Description                                                       |
| --------- | ------------------------- | ----------------------------------------------------------------- |
| `cpu()`   | Method                    | Returns the keypoints tensor on CPU memory.                       |
| `numpy()` | Method                    | Returns the keypoints tensor as a numpy array.                    |
| `cuda()`  | Method                    | Returns the keypoints tensor on GPU memory.                       |
| `to()`    | Method                    | Returns the keypoints tensor with the specified device and dtype. |
| `xyn`     | Property (`torch.Tensor`) | A list of normalized keypoints represented as tensors.            |
| `xy`      | Property (`torch.Tensor`) | A list of keypoints in pixel coordinates represented as tensors.  |
| `conf`    | Property (`torch.Tensor`) | Returns confidence values of keypoints if available, else None.   |

For more details see the [`Keypoints` class documentation](../reference/engine/results.md#ultralytics.engine.results.Keypoints).

### Probs

`Probs` object can be used index, get `top1` and `top5` indices and scores of classification.

!!! example "Probs"

    ```python
    from ultralytics import YOLO

    # Load a pretrained YOLOv8n-cls Classify model
    model = YOLO("yolov8n-cls.pt")

    # Run inference on an image
    results = model("bus.jpg")  # results list

    # View results
    for r in results:
        print(r.probs)  # print the Probs object containing the detected class probabilities
    ```

Here's a table summarizing the methods and properties for the `Probs` class:

| Name       | Type                      | Description                                                             |
| ---------- | ------------------------- | ----------------------------------------------------------------------- |
| `cpu()`    | Method                    | Returns a copy of the probs tensor on CPU memory.                       |
| `numpy()`  | Method                    | Returns a copy of the probs tensor as a numpy array.                    |
| `cuda()`   | Method                    | Returns a copy of the probs tensor on GPU memory.                       |
| `to()`     | Method                    | Returns a copy of the probs tensor with the specified device and dtype. |
| `top1`     | Property (`int`)          | Index of the top 1 class.                                               |
| `top5`     | Property (`list[int]`)    | Indices of the top 5 classes.                                           |
| `top1conf` | Property (`torch.Tensor`) | Confidence of the top 1 class.                                          |
| `top5conf` | Property (`torch.Tensor`) | Confidences of the top 5 classes.                                       |

For more details see the [`Probs` class documentation](../reference/engine/results.md#ultralytics.engine.results.Probs).

### OBB

`OBB` object can be used to index, manipulate, and convert oriented bounding boxes to different formats.

!!! example "OBB"

    ```python
    from ultralytics import YOLO

    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n-obb.pt")

    # Run inference on an image
    results = model("bus.jpg")  # results list

    # View results
    for r in results:
        print(r.obb)  # print the OBB object containing the oriented detection bounding boxes
    ```

Here is a table for the `OBB` class methods and properties, including their name, type, and description:

| Name        | Type                      | Description                                                           |
| ----------- | ------------------------- | --------------------------------------------------------------------- |
| `cpu()`     | Method                    | Move the object to CPU memory.                                        |
| `numpy()`   | Method                    | Convert the object to a numpy array.                                  |
| `cuda()`    | Method                    | Move the object to CUDA memory.                                       |
| `to()`      | Method                    | Move the object to the specified device.                              |
| `conf`      | Property (`torch.Tensor`) | Return the confidence values of the boxes.                            |
| `cls`       | Property (`torch.Tensor`) | Return the class values of the boxes.                                 |
| `id`        | Property (`torch.Tensor`) | Return the track IDs of the boxes (if available).                     |
| `xyxy`      | Property (`torch.Tensor`) | Return the horizontal boxes in xyxy format.                           |
| `xywhr`     | Property (`torch.Tensor`) | Return the rotated boxes in xywhr format.                             |
| `xyxyxyxy`  | Property (`torch.Tensor`) | Return the rotated boxes in xyxyxyxy format.                          |
| `xyxyxyxyn` | Property (`torch.Tensor`) | Return the rotated boxes in xyxyxyxy format normalized by image size. |

For more details see the [`OBB` class documentation](../reference/engine/results.md#ultralytics.engine.results.OBB).

## Plotting Results

The `plot()` method in `Results` objects facilitates visualization of predictions by overlaying detected objects (such as bounding boxes, masks, keypoints, and probabilities) onto the original image. This method returns the annotated image as a NumPy array, allowing for easy display or saving.

!!! example "Plotting"

    ```python
    from PIL import Image

    from ultralytics import YOLO

    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Run inference on 'bus.jpg'
    results = model(["bus.jpg", "zidane.jpg"])  # results list

    # Visualize the results
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Show results to screen (in supported environments)
        r.show()

        # Save results to disk
        r.save(filename=f"results{i}.jpg")
    ```

### `plot()` Method Parameters

The `plot()` method supports various arguments to customize the output:

| Argument     | Type            | Description                                                                | Default       |
| ------------ | --------------- | -------------------------------------------------------------------------- | ------------- |
| `conf`       | `bool`          | Include detection confidence scores.                                       | `True`        |
| `line_width` | `float`         | Line width of bounding boxes. Scales with image size if `None`.            | `None`        |
| `font_size`  | `float`         | Text font size. Scales with image size if `None`.                          | `None`        |
| `font`       | `str`           | Font name for text annotations.                                            | `'Arial.ttf'` |
| `pil`        | `bool`          | Return image as a PIL Image object.                                        | `False`       |
| `img`        | `numpy.ndarray` | Alternative image for plotting. Uses the original image if `None`.         | `None`        |
| `im_gpu`     | `torch.Tensor`  | GPU-accelerated image for faster mask plotting. Shape: (1, 3, 640, 640).   | `None`        |
| `kpt_radius` | `int`           | Radius for drawn keypoints.                                                | `5`           |
| `kpt_line`   | `bool`          | Connect keypoints with lines.                                              | `True`        |
| `labels`     | `bool`          | Include class labels in annotations.                                       | `True`        |
| `boxes`      | `bool`          | Overlay bounding boxes on the image.                                       | `True`        |
| `masks`      | `bool`          | Overlay masks on the image.                                                | `True`        |
| `probs`      | `bool`          | Include classification probabilities.                                      | `True`        |
| `show`       | `bool`          | Display the annotated image directly using the default image viewer.       | `False`       |
| `save`       | `bool`          | Save the annotated image to a file specified by `filename`.                | `False`       |
| `filename`   | `str`           | Path and name of the file to save the annotated image if `save` is `True`. | `None`        |
| `color_mode` | `str`           | Specify the color mode, e.g., 'instance' or 'class'.                       | `'class'`     |

## Thread-Safe Inference

Ensuring thread safety during inference is crucial when you are running multiple YOLO models in parallel across different threads. Thread-safe inference guarantees that each thread's predictions are isolated and do not interfere with one another, avoiding race conditions and ensuring consistent and reliable outputs.

When using YOLO models in a multi-threaded application, it's important to instantiate separate model objects for each thread or employ thread-local storage to prevent conflicts:

!!! example "Thread-Safe Inference"

    Instantiate a single model inside each thread for thread-safe inference:
    ```python
    from threading import Thread

    from ultralytics import YOLO


    def thread_safe_predict(model, image_path):
        """Performs thread-safe prediction on an image using a locally instantiated YOLO model."""
        model = YOLO(model)
        results = model.predict(image_path)
        # Process results


    # Starting threads that each have their own model instance
    Thread(target=thread_safe_predict, args=("yolov8n.pt", "image1.jpg")).start()
    Thread(target=thread_safe_predict, args=("yolov8n.pt", "image2.jpg")).start()
    ```

For an in-depth look at thread-safe inference with YOLO models and step-by-step instructions, please refer to our [YOLO Thread-Safe Inference Guide](../guides/yolo-thread-safe-inference.md). This guide will provide you with all the necessary information to avoid common pitfalls and ensure that your multi-threaded inference runs smoothly.

## Streaming Source `for`-loop

Here's a Python script using OpenCV (`cv2`) and YOLOv8 to run inference on video frames. This script assumes you have already installed the necessary packages (`opencv-python` and `ultralytics`).

!!! example "Streaming for-loop"

    ```python
    import cv2

    from ultralytics import YOLO

    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Open the video file
    video_path = "path/to/your/video/file.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    ```

This script will run predictions on each frame of the video, visualize the results, and display them in a window. The loop can be exited by pressing 'q'.

[car spare parts]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1
[football player detect]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442
[human fall detect]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43

## FAQ

### What is Ultralytics YOLOv8 and its predict mode for real-time inference?

Ultralytics YOLOv8 is a state-of-the-art model for real-time [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and classification. Its **predict mode** allows users to perform high-speed inference on various data sources such as images, videos, and live streams. Designed for performance and versatility, it also offers batch processing and streaming modes. For more details on its features, check out the [Ultralytics YOLOv8 predict mode](#key-features-of-predict-mode).

### How can I run inference using Ultralytics YOLOv8 on different data sources?

Ultralytics YOLOv8 can process a wide range of data sources, including individual images, videos, directories, URLs, and streams. You can specify the data source in the `model.predict()` call. For example, use `'image.jpg'` for a local image or `'https://ultralytics.com/images/bus.jpg'` for a URL. Check out the detailed examples for various [inference sources](#inference-sources) in the documentation.

### How do I optimize YOLOv8 inference speed and memory usage?

To optimize inference speed and manage memory efficiently, you can use the streaming mode by setting `stream=True` in the predictor's call method. The streaming mode generates a memory-efficient generator of `Results` objects instead of loading all frames into memory. For processing long videos or large datasets, streaming mode is particularly useful. Learn more about [streaming mode](#key-features-of-predict-mode).

### What inference arguments does Ultralytics YOLOv8 support?

The `model.predict()` method in YOLOv8 supports various arguments such as `conf`, `iou`, `imgsz`, `device`, and more. These arguments allow you to customize the inference process, setting parameters like confidence thresholds, image size, and the device used for computation. Detailed descriptions of these arguments can be found in the [inference arguments](#inference-arguments) section.

### How can I visualize and save the results of YOLOv8 predictions?

After running inference with YOLOv8, the `Results` objects contain methods for displaying and saving annotated images. You can use methods like `result.show()` and `result.save(filename="result.jpg")` to visualize and save the results. For a comprehensive list of these methods, refer to the [working with results](#working-with-results) section.
