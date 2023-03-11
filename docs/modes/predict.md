<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png">

Inference or prediction of a task returns a list of `Results` objects. Alternatively, in the streaming mode, it returns
a generator of `Results` objects which is memory efficient. Streaming mode can be enabled by passing `stream=True` in
predictor's call method.

!!! example "Predict"

    === "Return a List"

    ```python
    inputs = [img, img]  # list of np arrays
    results = model(inputs)  # List of Results objects
    
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmenation masks outputs
        probs = result.probs  # Class probabilities for classification outputs
    ```
    
    === "Return a Generator"

    ```python
    inputs = [img, img]  # list of numpy arrays
    results = model(inputs, stream=True)  # generator of Results objects
    
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segmenation masks outputs
        probs = r.probs  # Class probabilities for classification outputs
    ```

## Sources

YOLOv8 can run inference on a variety of sources. The table below lists the various sources that can be used as input
for YOLOv8, along with the required format and notes. Sources include images, URLs, PIL images, OpenCV, numpy arrays,
torch tensors, CSV files, videos, directories, globs, YouTube videos, and streams. The table also indicates whether each
source can be used as a stream and the model argument required for that source.

| source     | stream  | model(arg)                                 | type           | notes            |
|------------|---------|--------------------------------------------|----------------|------------------|
| image      |         | `'im.jpg'`                                 | `str`, `Path`  |                  |
| URL        |         | `'https://ultralytics.com/images/bus.jpg'` | `str`          |                  |
| screenshot |         | `'screen'`                                 | `str`          |                  |
| PIL        |         | `Image.open('im.jpg')`                     | `PIL.Image`    | HWC, RGB         |
| OpenCV     |         | `cv2.imread('im.jpg')[:,:,::-1]`           | `np.ndarray`   | HWC, BGR to RGB  |
| numpy      |         | `np.zeros((640,1280,3))`                   | `np.ndarray`   | HWC              |
| torch      |         | `torch.zeros(16,3,320,640)`                | `torch.Tensor` | BCHW, RGB        |
| CSV        |         | `'sources.csv'`                            | `str`, `Path`  | RTSP, RTMP, HTTP |         
| video      | &check; | `'vid.mp4'`                                | `str`, `Path`  |                  |
| directory  | &check; | `'path/'`                                  | `str`, `Path`  |                  |
| glob       | &check; | `'path/*.jpg'`                             | `str`          | Use `*` operator |
| YouTube    | &check; | `'https://youtu.be/Zgi9g1ksQHc'`           | `str`          |                  |
| stream     | &check; | `'rtsp://example.com/media.mp4'`           | `str`          | RTSP, RTMP, HTTP |

## Image Formats

For images, YOLOv8 supports a variety of image formats defined
in [yolo/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/data/utils.py). The
following suffixes are valid for images:

| Image Suffixes | Example Predict Command          | Reference                                                                            |
|----------------|----------------------------------|--------------------------------------------------------------------------------------|
| bmp            | `yolo predict source=image.bmp`  | [Microsoft](https://docs.microsoft.com/en-us/windows/win32/gdi/bitmap-file-format)   |
| dng            | `yolo predict source=image.dng`  | [Adobe](https://helpx.adobe.com/photoshop/using/digital-negative.html)               |
| jpeg           | `yolo predict source=image.jpeg` | [Joint Photographic Experts Group](https://jpeg.org/jpeg/)                           |
| jpg            | `yolo predict source=image.jpg`  | [Joint Photographic Experts Group](https://jpeg.org/jpeg/)                           |
| mpo            | `yolo predict source=image.mpo`  | [CIPA](https://www.cipa.jp/std/documents/e/DC-007-Translation-2018-E.pdf)            |
| png            | `yolo predict source=image.png`  | [Portable Network Graphics](https://www.w3.org/TR/PNG/)                              |
| tif            | `yolo predict source=image.tif`  | [Adobe](https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/tiff6.pdf) |
| tiff           | `yolo predict source=image.tiff` | [Adobe](https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/tiff6.pdf) |
| webp           | `yolo predict source=image.webp` | [Google Developers](https://developers.google.com/speed/webp)                        |
| pfm            | `yolo predict source=image.pfm`  | [HDR Labs](http://hdrlabs.com/tools/pfrenchy/)                                       |

## Video Formats

For videos, YOLOv8 also supports a variety of video formats defined
in [yolo/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/data/utils.py). The
following suffixes are valid for videos:

| Video Suffixes | Example Predict Command          | Reference                                                                                                      |
|----------------|----------------------------------|----------------------------------------------------------------------------------------------------------------|
| asf            | `yolo predict source=video.asf`  | [Microsoft](https://docs.microsoft.com/en-us/windows/win32/wmformat/asf-file-structure)                        |
| avi            | `yolo predict source=video.avi`  | [Microsoft](https://docs.microsoft.com/en-us/windows/win32/directshow/avi-riff-file-reference)                 |
| gif            | `yolo predict source=video.gif`  | [CompuServe](https://www.w3.org/Graphics/GIF/spec-gif89a.txt)                                                  |
| m4v            | `yolo predict source=video.m4v`  | [Apple](https://developer.apple.com/library/archive/documentation/QuickTime/QTFF/QTFFChap2/qtff2.html)         |
| mkv            | `yolo predict source=video.mkv`  | [Matroska](https://matroska.org/technical/specs/index.html)                                                    |
| mov            | `yolo predict source=video.mov`  | [Apple](https://developer.apple.com/library/archive/documentation/QuickTime/QTFF/QTFFPreface/qtffPreface.html) |
| mp4            | `yolo predict source=video.mp4`  | [ISO 68939](https://www.iso.org/standard/68939.html)                                                           |
| mpeg           | `yolo predict source=video.mpeg` | [ISO 56021](https://www.iso.org/standard/56021.html)                                                           |
| mpg            | `yolo predict source=video.mpg`  | [ISO 56021](https://www.iso.org/standard/56021.html)                                                           |
| ts             | `yolo predict source=video.ts`   | [MPEG Transport Stream](https://en.wikipedia.org/wiki/MPEG_transport_stream)                                   |
| wmv            | `yolo predict source=video.wmv`  | [Microsoft](https://docs.microsoft.com/en-us/windows/win32/wmformat/wmv-file-structure)                        |
| webm           | `yolo predict source=video.webm` | [Google Developers](https://developers.google.com/media/vp9/getting-started/webm-file-format)                  |

## Working with Results

Results object consists of these component objects:

- `Results.boxes`: `Boxes` object with properties and methods for manipulating bboxes
- `Results.masks`: `Masks` object used to index masks or to get segment coordinates.
- `Results.probs`: `torch.Tensor` containing the class probabilities/logits.
- `Results.orig_img`: Original image loaded in memory.
- `Results.path`: `Path` containing the path to input image

Each result is composed of torch.Tensor by default, in which you can easily use following functionality:

```python
results = results.cuda()
results = results.cpu()
results = results.to("cpu")
results = results.numpy()
```

### Boxes

`Boxes` object can be used index, manipulate and convert bboxes to different formats. The box format conversion
operations are cached, which means they're only calculated once per object and those values are reused for future calls.

- Indexing a `Boxes` objects returns a `Boxes` object

```python
results = model(inputs)
boxes = results[0].boxes
box = boxes[0]  # returns one box
box.xyxy 
```

- Properties and conversions

```python
boxes.xyxy  # box with xyxy format, (N, 4)
boxes.xywh  # box with xywh format, (N, 4)
boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
boxes.xywhn  # box with xywh format but normalized, (N, 4)
boxes.conf  # confidence score, (N, 1)
boxes.cls  # cls, (N, 1)
boxes.data  # raw bboxes tensor, (N, 6) or boxes.boxes .
```

### Masks

`Masks` object can be used index, manipulate and convert masks to segments. The segment conversion operation is cached.

```python
results = model(inputs)
masks = results[0].masks  # Masks object
masks.segments  # bounding coordinates of masks, List[segment] * N
masks.data  # raw masks tensor, (N, H, W) or masks.masks 
```

### probs

`probs` attribute of `Results` class is a `Tensor` containing class probabilities of a classification operation.

```python
results = model(inputs)
results[0].probs  # cls prob, (num_class, )
```

Class reference documentation for `Results` module and its components can be found [here](../reference/results.md)

## Plotting results

You can use `plot()` function of `Result` object to plot results on in image object. It plots all components(boxes,
masks, classification logits, etc) found in the results object

```python
res = model(img)
res_plotted = res[0].plot()
cv2.imshow("result", res_plotted)
```

!!! example "`plot()` arguments"

    `show_conf (bool)`: Show confidence

    `line_width (Float)`: The line width of boxes. Automatically scaled to img size if not provided

    `font_size (Float)`: The font size of . Automatically scaled to img size if not provided
