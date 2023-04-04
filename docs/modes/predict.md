<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png">

YOLOv8 **predict mode** can generate predictions for various tasks, returning either a list of `Results` objects or a
memory-efficient generator of `Results` objects when using the streaming mode. Enable streaming mode by
passing `stream=True` in the predictor's call method.

!!! example "Predict"

    === "Return a list with `Stream=False`"
        ```python
        inputs = [img, img]  # list of numpy arrays
        results = model(inputs)  # list of Results objects
        
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            probs = result.probs  # Class probabilities for classification outputs
        ```

    === "Return a generator with `Stream=True`"
        ```python
        inputs = [img, img]  # list of numpy arrays
        results = model(inputs, stream=True)  # generator of Results objects
        
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            probs = result.probs  # Class probabilities for classification outputs
        ```

!!! tip "Tip"

    Streaming mode with `stream=True` should be used for long videos or large predict sources, otherwise results will accumuate in memory and will eventually cause out-of-memory errors. 

## Sources

YOLOv8 can accept various input sources, as shown in the table below. This includes images, URLs, PIL images, OpenCV,
numpy arrays, torch tensors, CSV files, videos, directories, globs, YouTube videos, and streams. The table indicates
whether each source can be used in streaming mode with `stream=True` ✅ and an example argument for each source.

| source      | model(arg)                                 | type           | notes            |
|-------------|--------------------------------------------|----------------|------------------|
| image       | `'im.jpg'`                                 | `str`, `Path`  |                  |
| URL         | `'https://ultralytics.com/images/bus.jpg'` | `str`          |                  |
| screenshot  | `'screen'`                                 | `str`          |                  |
| PIL         | `Image.open('im.jpg')`                     | `PIL.Image`    | HWC, RGB         |
| OpenCV      | `cv2.imread('im.jpg')[:,:,::-1]`           | `np.ndarray`   | HWC, BGR to RGB  |
| numpy       | `np.zeros((640,1280,3))`                   | `np.ndarray`   | HWC              |
| torch       | `torch.zeros(16,3,320,640)`                | `torch.Tensor` | BCHW, RGB        |
| CSV         | `'sources.csv'`                            | `str`, `Path`  | RTSP, RTMP, HTTP |         
| video ✅     | `'vid.mp4'`                                | `str`, `Path`  |                  |
| directory ✅ | `'path/'`                                  | `str`, `Path`  |                  |
| glob ✅      | `'path/*.jpg'`                             | `str`          | Use `*` operator |
| YouTube ✅   | `'https://youtu.be/Zgi9g1ksQHc'`           | `str`          |                  |
| stream ✅    | `'rtsp://example.com/media.mp4'`           | `str`          | RTSP, RTMP, HTTP |


## Arguments
`model.predict` accepts multiple arguments that control the predction operation. These arguments can be passed directly to `model.predict`:
!!! example
    ```
    model.predict(source, save=True, imgsz=320, conf=0.5)
    ```

All supported arguments:

| Key              | Value                  | Description                                              |
|------------------|------------------------|----------------------------------------------------------|
| `source`         | `'ultralytics/assets'` | source directory for images or videos                    |
| `conf`           | `0.25`                 | object confidence threshold for detection                |
| `iou`            | `0.7`                  | intersection over union (IoU) threshold for NMS          |
| `half`           | `False`                | use half precision (FP16)                                |
| `device`         | `None`                 | device to run on, i.e. cuda device=0/1/2/3 or device=cpu |
| `show`           | `False`                | show results if possible                                 |
| `save`           | `False`                | save images with results                                 |
| `save_txt`       | `False`                | save results as .txt file                                |
| `save_conf`      | `False`                | save results with confidence scores                      |
| `save_crop`      | `False`                | save cropped images with results                         |
| `hide_labels`    | `False`                | hide labels                                              |
| `hide_conf`      | `False`                | hide confidence scores                                   |
| `max_det`        | `300`                  | maximum number of detections per image                   |
| `vid_stride`     | `False`                | video frame-rate stride                                  |
| `line_thickness` | `3`                    | bounding box thickness (pixels)                          |
| `visualize`      | `False`                | visualize model features                                 |
| `augment`        | `False`                | apply image augmentation to prediction sources           |
| `agnostic_nms`   | `False`                | class-agnostic NMS                                       |
| `retina_masks`   | `False`                | use high-resolution segmentation masks                   |
| `classes`        | `None`                 | filter results by class, i.e. class=0, or class=[0,2,3]  |
| `boxes`          | `True`                 | Show boxes in segmentation predictions                   |

## Image and Video Formats

YOLOv8 supports various image and video formats, as specified
in [yolo/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/data/utils.py). See the
tables below for the valid suffixes and example predict commands.

### Image Suffixes

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
- `Results.probs`: `torch.Tensor` containing class probabilities or logits
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
    boxes.conf  # confidence score, (N, 1)
    boxes.cls  # cls, (N, 1)
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

### probs

`probs` attribute of `Results` class is a `Tensor` containing class probabilities of a classification operation.

!!! example "Probs"

    ```python
    results = model(inputs)
    results[0].probs  # cls prob, (num_class, )
    ```

Class reference documentation for `Results` module and its components can be found [here](../reference/results.md)

## Plotting results

You can use `plot()` function of `Result` object to plot results on in image object. It plots all components(boxes,
masks, classification logits, etc.) found in the results object

!!! example "Plotting"

    ```python
    res = model(img)
    res_plotted = res[0].plot()
    cv2.imshow("result", res_plotted)
    ```

- `show_conf (bool)`: Show confidence
- `line_width (Float)`: The line width of boxes. Automatically scaled to img size if not provided
- `font_size (Float)`: The font size of . Automatically scaled to img size if not provided

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