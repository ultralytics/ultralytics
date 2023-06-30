---
comments: true
description: Get started with YOLOv8 Predict mode and input sources. Accepts various input sources such as images, videos, and directories.
keywords: YOLOv8, predict mode, generator, streaming mode, input sources, video formats, arguments customization
---

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png">

YOLOv8 **predict mode** can generate predictions for various tasks, returning either a list of `Results` objects or a
memory-efficient generator of `Results` objects when using the streaming mode. Enable streaming mode by
passing `stream=True` in the predictor's call method.

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
            probs = result.probs  # Class probabilities for classification outputs
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
            probs = result.probs  # Class probabilities for classification outputs
        ```

## Inference Sources

YOLOv8 can process different types of input sources for inference, as shown in the table below. The sources include static images, video streams, and various data formats. The table also indicates whether each source can be used in streaming mode with the argument `stream=True` ✅. Streaming mode is beneficial for processing videos or live streams as it creates a generator of results instead of loading all frames into memory.

!!! tip "Tip"

    Use `stream=True` for processing long videos or large datasets to efficiently manage memory. When `stream=False`, the results for all frames or data points are stored in memory, which can quickly add up and cause out-of-memory errors for large inputs. In contrast, `stream=True` utilizes a generator, which only keeps the results of the current frame or data point in memory, significantly reducing memory consumption and preventing out-of-memory issues.

| Source      | Argument                                   | Type                                  | Notes                                                                      |
|-------------|--------------------------------------------|---------------------------------------|----------------------------------------------------------------------------|
| image       | `'image.jpg'`                              | `str` or `Path`                       | Single image file.                                                         |
| URL         | `'https://ultralytics.com/images/bus.jpg'` | `str`                                 | URL to an image.                                                           |
| screenshot  | `'screen'`                                 | `str`                                 | Capture a screenshot.                                                      |
| PIL         | `Image.open('im.jpg')`                     | `PIL.Image`                           | HWC format with RGB channels.                                              |
| OpenCV      | `cv2.imread('im.jpg')`                     | `np.ndarray` of `uint8 (0-255)`       | HWC format with BGR channels.                                              |
| numpy       | `np.zeros((640,1280,3))`                   | `np.ndarray` of `uint8 (0-255)`       | HWC format with BGR channels.                                              |
| torch       | `torch.zeros(16,3,320,640)`                | `torch.Tensor` of `float32 (0.0-1.0)` | BCHW format with RGB channels.                                             |
| CSV         | `'sources.csv'`                            | `str` or `Path`                       | CSV file containing paths to images, videos, or directories.               |       
| video ✅     | `'video.mp4'`                              | `str` or `Path`                       | Video file in formats like MP4, AVI, etc.                                  |
| directory ✅ | `'path/'`                                  | `str` or `Path`                       | Path to a directory containing images or videos.                           |
| glob ✅      | `'path/*.jpg'`                             | `str`                                 | Glob pattern to match multiple files. Use the `*` character as a wildcard. |
| YouTube ✅   | `'https://youtu.be/Zgi9g1ksQHc'`           | `str`                                 | URL to a YouTube video.                                                    |
| stream ✅    | `'rtsp://example.com/media.mp4'`           | `str`                                 | URL for streaming protocols such as RTSP, RTMP, or an IP address.          |

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
    
    === "torch"
        Run inference on an image represented as a PyTorch tensor.
        ```python
        import torch
        from ultralytics import YOLO
    
        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')
    
        # Create a random torch tensor of BCHW shape (1, 3, 640, 640) with values in range [0, 1] and type float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)
    
        # Run inference on the source
        results = model(source)  # list of Results objects
        ```
    
    === "CSV"
        Run inference on a collection of images, URLs, videos and directories listed in a CSV file.
        ```python
        import torch
        from ultralytics import YOLO
    
        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')
    
        # Define a path to a CSV file with images, URLs, videos and directories
        source = 'path/to/file.csv'
    
        # Run inference on the source
        results = model(source)  # list of Results objects
        ```
    
    === "video"
        Run inference on a video file. By using `stream=True`, you can create a generator of Results objects to reduce memory usage.
        ```python
        from ultralytics import YOLO
    
        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')
    
        # Define path to video file
        source = 'path/to/video.mp4'
    
        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```
    
    === "directory"
        Run inference on all images and videos in a directory. To also capture images and videos in subdirectories use a glob pattern, i.e. `path/to/dir/**/*`.
        ```python
        from ultralytics import YOLO
    
        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')
    
        # Define path to directory containing images and videos for inference
        source = 'path/to/dir'
    
        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```
    
    === "glob"
        Run inference on all images and videos that match a glob expression with `*` characters.
        ```python
        from ultralytics import YOLO
    
        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')
    
        # Define a glob search for all JPG files in a directory
        source = 'path/to/dir/*.jpg'
    
        # OR define a recursive glob search for all JPG files including subdirectories
        source = 'path/to/dir/**/*.jpg'
    
        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```
    
    === "YouTube"
        Run inference on a YouTube video. By using `stream=True`, you can create a generator of Results objects to reduce memory usage for long videos.
        ```python
        from ultralytics import YOLO
    
        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')
    
        # Define source as YouTube video URL
        source = 'https://youtu.be/Zgi9g1ksQHc'
    
        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```
    
    === "Stream"
        Run inference on remote streaming sources using RTSP, RTMP, and IP address protocols.
        ```python
        from ultralytics import YOLO
    
        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')
    
        # Define source as RTSP, RTMP or IP streaming address
        source = 'rtsp://example.com/media.mp4'
    
        # Run inference on the source
        results = model(source, stream=True)  # generator of Results objects
        ```

## Inference Arguments

`model.predict` accepts multiple arguments that control the prediction operation. These arguments can be passed directly to `model.predict`:
!!! example

    ```python
    model.predict(source, save=True, imgsz=320, conf=0.5)
    ```

All supported arguments:

| Key            | Value                  | Description                                                                    |
|----------------|------------------------|--------------------------------------------------------------------------------|
| `source`       | `'ultralytics/assets'` | source directory for images or videos                                          |
| `conf`         | `0.25`                 | object confidence threshold for detection                                      |
| `iou`          | `0.7`                  | intersection over union (IoU) threshold for NMS                                |
| `half`         | `False`                | use half precision (FP16)                                                      |
| `device`       | `None`                 | device to run on, i.e. cuda device=0/1/2/3 or device=cpu                       |
| `show`         | `False`                | show results if possible                                                       |
| `save`         | `False`                | save images with results                                                       |
| `save_txt`     | `False`                | save results as .txt file                                                      |
| `save_conf`    | `False`                | save results with confidence scores                                            |
| `save_crop`    | `False`                | save cropped images with results                                               |
| `hide_labels`  | `False`                | hide labels                                                                    |
| `hide_conf`    | `False`                | hide confidence scores                                                         |
| `max_det`      | `300`                  | maximum number of detections per image                                         |
| `vid_stride`   | `False`                | video frame-rate stride                                                        |
| `line_width`   | `None`                 | The line width of the bounding boxes. If None, it is scaled to the image size. |
| `visualize`    | `False`                | visualize model features                                                       |
| `augment`      | `False`                | apply image augmentation to prediction sources                                 |
| `agnostic_nms` | `False`                | class-agnostic NMS                                                             |
| `retina_masks` | `False`                | use high-resolution segmentation masks                                         |
| `classes`      | `None`                 | filter results by class, i.e. class=0, or class=[0,2,3]                        |
| `boxes`        | `True`                 | Show boxes in segmentation predictions                                         |

## Image and Video Formats

YOLOv8 supports various image and video formats, as specified in [yolo/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/data/utils.py). See the tables below for the valid suffixes and example predict commands.

### Image Suffixes

The below table contains valid Ultralytics image formats.

| Image Suffixes | Example Predict Command          | Reference                                                                     |
|----------------|----------------------------------|-------------------------------------------------------------------------------|
| .bmp           | `yolo predict source=image.bmp`  | [Microsoft BMP File Format](https://en.wikipedia.org/wiki/BMP_file_format)    |
| .dng           | `yolo predict source=image.dng`  | [Adobe DNG](https://www.adobe.com/products/photoshop/extend.displayTab2.html) |
| .jpeg          | `yolo predict source=image.jpeg` | [JPEG](https://en.wikipedia.org/wiki/JPEG)                                    |
| .jpg           | `yolo predict source=image.jpg`  | [JPEG](https://en.wikipedia.org/wiki/JPEG)                                    |
| .mpo           | `yolo predict source=image.mpo`  | [Multi Picture Object](https://fileinfo.com/extension/mpo)                    |
| .png           | `yolo predict source=image.png`  | [Portable Network Graphics](https://en.wikipedia.org/wiki/PNG)                |
| .tif           | `yolo predict source=image.tif`  | [Tag Image File Format](https://en.wikipedia.org/wiki/TIFF)                   |
| .tiff          | `yolo predict source=image.tiff` | [Tag Image File Format](https://en.wikipedia.org/wiki/TIFF)                   |
| .webp          | `yolo predict source=image.webp` | [WebP](https://en.wikipedia.org/wiki/WebP)                                    |
| .pfm           | `yolo predict source=image.pfm`  | [Portable FloatMap](https://en.wikipedia.org/wiki/Netpbm#File_formats)        |

### Video Suffixes

The below table contains valid Ultralytics video formats.

| Video Suffixes | Example Predict Command          | Reference                                                                        |
|----------------|----------------------------------|----------------------------------------------------------------------------------|
| .asf           | `yolo predict source=video.asf`  | [Advanced Systems Format](https://en.wikipedia.org/wiki/Advanced_Systems_Format) |
| .avi           | `yolo predict source=video.avi`  | [Audio Video Interleave](https://en.wikipedia.org/wiki/Audio_Video_Interleave)   |
| .gif           | `yolo predict source=video.gif`  | [Graphics Interchange Format](https://en.wikipedia.org/wiki/GIF)                 |
| .m4v           | `yolo predict source=video.m4v`  | [MPEG-4 Part 14](https://en.wikipedia.org/wiki/M4V)                              |
| .mkv           | `yolo predict source=video.mkv`  | [Matroska](https://en.wikipedia.org/wiki/Matroska)                               |
| .mov           | `yolo predict source=video.mov`  | [QuickTime File Format](https://en.wikipedia.org/wiki/QuickTime_File_Format)     |
| .mp4           | `yolo predict source=video.mp4`  | [MPEG-4 Part 14 - Wikipedia](https://en.wikipedia.org/wiki/MPEG-4_Part_14)       |
| .mpeg          | `yolo predict source=video.mpeg` | [MPEG-1 Part 2](https://en.wikipedia.org/wiki/MPEG-1)                            |
| .mpg           | `yolo predict source=video.mpg`  | [MPEG-1 Part 2](https://en.wikipedia.org/wiki/MPEG-1)                            |
| .ts            | `yolo predict source=video.ts`   | [MPEG Transport Stream](https://en.wikipedia.org/wiki/MPEG_transport_stream)     |
| .wmv           | `yolo predict source=video.wmv`  | [Windows Media Video](https://en.wikipedia.org/wiki/Windows_Media_Video)         |
| .webm          | `yolo predict source=video.webm` | [WebM Project](https://en.wikipedia.org/wiki/WebM)                               |

## Working with Results

The `Results` object contains the following components:

- `Results.boxes`: `Boxes` object with properties and methods for manipulating bounding boxes
- `Results.masks`: `Masks` object for indexing masks or getting segment coordinates
- `Results.keypoints`: `Keypoints` object for with properties and methods for manipulating predicted keypoints.
- `Results.probs`: `Probs` object for containing class probabilities.
- `Results.orig_img`: Original image loaded in memory
- `Results.path`: `Path` containing the path to the input image

Each result is composed of a `torch.Tensor` by default, which allows for easy manipulation:

!!! example "Results"

    ```python
    results = results.cuda()
    results = results.cpu()
    results = results.to('cpu')
    results = results.numpy()
    ```

### Boxes

`Boxes` object can be used to index, manipulate, and convert bounding boxes to different formats. Box format conversion
operations are cached, meaning they're only calculated once per object, and those values are reused for future calls.

- Indexing a `Boxes` object returns a `Boxes` object:

!!! example "Boxes"

    ```python
    results = model(img)
    boxes = results[0].boxes
    box = boxes[0]  # returns one box
    box.xyxy
    ```

- Properties and conversions

!!! example "Boxes Properties"

    ```python
    boxes.xyxy  # box with xyxy format, (N, 4)
    boxes.xywh  # box with xywh format, (N, 4)
    boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    boxes.xywhn  # box with xywh format but normalized, (N, 4)
    boxes.conf  # confidence score, (N, )
    boxes.cls  # cls, (N, )
    boxes.data  # raw bboxes tensor, (N, 6) or boxes.boxes
    ```

### Masks

`Masks` object can be used index, manipulate and convert masks to segments. The segment conversion operation is cached.

!!! example "Masks"

    ```python
    results = model(inputs)
    masks = results[0].masks  # Masks object
    masks.xy  # x, y segments (pixels), List[segment] * N
    masks.xyn  # x, y segments (normalized), List[segment] * N
    masks.data  # raw masks tensor, (N, H, W) or masks.masks 
    ```

### Keypoints

`Keypoints` object can be used index, manipulate and normalize coordinates. The keypoint conversion operation is cached.

!!! example "Keypoints"

    ```python
    results = model(inputs)
    keypoints = results[0].keypoints  # Masks object
    keypoints.xy  # x, y keypoints (pixels), (num_dets, num_kpts, 2/3), the last dimension can be 2 or 3, depends the model.
    keypoints.xyn  # x, y keypoints (normalized), (num_dets, num_kpts, 2/3)
    keypoints.conf  # confidence score(num_dets, num_kpts) of each keypoint if the last dimension is 3.
    keypoints.data  # raw keypoints tensor, (num_dets, num_kpts, 2/3) 
    ```

### probs

`Probs` object can be used index, get top1&top5 indices and scores of classification.

!!! example "Probs"

    ```python
    results = model(inputs)
    probs = results[0].probs  # cls prob, (num_class, )
    probs.top5    # The top5 indices of classification, List[Int] * 5.
    probs.top1    # The top1 indices of classification, a value with Int type.
    probs.top5conf  # The top5 scores of classification, a tensor with shape (5, ).
    probs.top1conf  # The top1 scores of classification. a value with torch.tensor type.
    keypoints.data  # raw probs tensor, (num_class, ) 
    ```

Class reference documentation for `Results` module and its components can be found [here](../reference/yolo/engine/results.md)

## Plotting results

You can use `plot()` function of `Result` object to plot results on in image object. It plots all components(boxes,
masks, classification probabilities, etc.) found in the results object

!!! example "Plotting"

    ```python
    res = model(img)
    res_plotted = res[0].plot()
    cv2.imshow("result", res_plotted)
    ```

| Argument                      | Description                                                                            |
|-------------------------------|----------------------------------------------------------------------------------------|
| `conf (bool)`                 | Whether to plot the detection confidence score.                                        |
| `line_width (int, optional)`  | The line width of the bounding boxes. If None, it is scaled to the image size.         |
| `font_size (float, optional)` | The font size of the text. If None, it is scaled to the image size.                    |
| `font (str)`                  | The font to use for the text.                                                          |
| `pil (bool)`                  | Whether to use PIL for image plotting.                                                 |
| `example (str)`               | An example string to display. Useful for indicating the expected format of the output. |
| `img (numpy.ndarray)`         | Plot to another image. if not, plot to original image.                                 |
| `labels (bool)`               | Whether to plot the label of bounding boxes.                                           |
| `boxes (bool)`                | Whether to plot the bounding boxes.                                                    |
| `masks (bool)`                | Whether to plot the masks.                                                             |
| `probs (bool)`                | Whether to plot classification probability.                                            |

## Streaming Source `for`-loop

Here's a Python script using OpenCV (cv2) and YOLOv8 to run inference on video frames. This script assumes you have already installed the necessary packages (opencv-python and ultralytics).

!!! example "Streaming for-loop"

    ```python
    import cv2
    from ultralytics import YOLO
    
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    
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