---
comments: true
description: Learn how to view image results inside a compatible VSCode terminal.
keywords: YOLOv8, VSCode, Terminal, Remote Development, Ultralytics, SSH, Object Detection, Inference, Results, Remote Tunnel, Images, Helpful, Productivity Hack
---

# Viewing Inference Results in a Terminal

<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/saitoha/libsixel/data/data/sixel.gif" alt="Sixel example of image in Terminal">
</p>

Image from the [libsixel](https://saitoha.github.io/libsixel/) website.

## Motivation

When connecting to a remote machine, normally visualizing image results is not possible or requires moving data to a local device with a GUI. The VSCode integrated terminal allows for directly rendering images. This is a short demonstration on how to use this in conjunction with `ultralytics` with [prediction results](../modes/predict.md).

!!! warning

    Only compatible with Linux and MacOS. Check the VSCode [repository](https://github.com/microsoft/vscode), check [Issue status](https://github.com/microsoft/vscode/issues/198622), or [documentation](https://code.visualstudio.com/docs) for updates about Windows support to view images in terminal with `sixel`.

The VSCode compatible protocols for viewing images using the integrated terminal are [`sixel`](https://en.wikipedia.org/wiki/Sixel) and [`iTerm`](https://iterm2.com/documentation-images.html). This guide will demonstrate use of the `sixel` protocol.

## Process

1. First, you must enable settings `terminal.integrated.enableImages` and `terminal.integrated.gpuAcceleration` in VSCode.

    ```yaml
    "terminal.integrated.gpuAcceleration": "auto" # "auto" is default, can also use "on"
    "terminal.integrated.enableImages": false
    ```

<p align="center">
  <img width="800" src="https://github.com/ultralytics/ultralytics/assets/62214284/d158ab1c-893c-4397-a5de-2f9f74f81175" alt="VSCode enable terminal images setting">
</p>

1. Install the `python-sixel` library in your virtual environment. This is a [fork](https://github.com/lubosz/python-sixel?tab=readme-ov-file) of the `PySixel` library, which is no longer maintained.

    ```bash
    pip install sixel
    ```

1. Import the relevant libraries

    ```py
    import io

    import cv2 as cv

    from ultralytics import YOLO
    from sixel import SixelWriter
    ```

1. Load a model and execute inference, then plot the results and store in a variable. See more about inference arguments and working with results on the [predict mode](../modes/predict.md) page.

    ```{ .py .annotate }
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolov8n.pt")

    # Run inference on an image
    results = model.predict(source="ultralytics/assets/bus.jpg")
    
    # Plot inference results
    plot = results[0].plot() #(1)!
    ```

    1. See [plot method parameters](../modes/predict.md#plot-method-parameters) to see possible arguments to use.

1. Now, use OpenCV to convert the `numpy.ndarray` to `bytes` data. Then use `io.BytesIO` to make a "file-like" object.

    ```{ .py .annotate }
    # Results image as bytes
    im_bytes = cv.imencode(
        ".png", #(1)!
        plot,
        )[1].tobytes() #(2)!

    # Image bytes as a file-like object
    mem_file = io.BytesIO(im_bytes)
    ```

    1. It's possible to use other image extensions as well.
    2. Only the object at index `1` that is returned is needed.

1. Create a `SixelWriter` instance, and then use the `.draw()` method to draw the image in the terminal.

    ```py
    # Create sixel writer object
    w = SixelWriter()

    # Draw the sixel image in the terminal
    w.draw(mem_file)
    ```

## Example Inference Results

<p align="center">
  <img width="800" src="https://github.com/ultralytics/ultralytics/assets/62214284/6743ab64-300d-4429-bdce-e246455f7b68" alt="View Image in Terminal">
</p>

!!! danger

    Using this example with videos or animated GIF frames has **not** been tested. Attempt at your own risk.

## Full Code Example

```{ .py .annotate }
import io

import cv2 as cv

from ultralytics import YOLO
from sixel import SixelWriter

# Load a model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model.predict(source="ultralytics/assets/bus.jpg")

# Plot inference results
plot = results[0].plot() #(3)!

# Results image as bytes
im_bytes = cv.imencode(
    ".png", #(1)!
    plot,
    )[1].tobytes() #(2)!

mem_file = io.BytesIO(im_bytes)
w = SixelWriter()
w.draw(mem_file)
```

1. It's possible to use other image extensions as well.
2. Only the object at index `1` that is returned is needed.
3. See [plot method parameters](../modes/predict.md#plot-method-parameters) to see possible arguments to use.

---

!!! tip

    You may need to use `clear` to "erase" the view of the image in the terminal.
