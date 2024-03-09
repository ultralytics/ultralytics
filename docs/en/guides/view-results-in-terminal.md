---
comments: true
description: Learn how to view image results inside a compatible VSCode terminal.
keywords: YOLOv8, VSCode, terminal, remote development, Ultralytics, ssh, object detection, inference
---

# Viewing Inference Results in a Terminal

<p align="center">
  <img width="800" src="" alt="View Image in Terminal">
</p>

!!! warning

    Only compatible with Linux and MacOS. Check the VSCode [repository](https://github.com/microsoft/vscode), check [Issue status](https://github.com/microsoft/vscode/issues/198622), or [documentation](https://code.visualstudio.com/docs) for updates about Windows support to view images in terminal with `sixel`.

The VSCode compatible libraries for viewing images using the integrated terminal are `sixel` and `iTerm`. This guide will use `sixel` but it _might_ be possible to reuse some of the same code for `iTerm` (not tested).

## Process

1. First, you must enable settings `terminal.integrated.enableImages` and `terminal.integrated.gpuAcceleration` in VSCode.

    ```yaml
    "terminal.integrated.gpuAcceleration": "auto" # "auto" is default, can also use "on"
    "terminal.integrated.enableImages": false
    ```

<p align="center">
  <img width="800" src="" alt="VSCode enable terminal images setting">
</p>

1. Import the relevant libraries

    ```py
    import io

    import cv2 as cv

    from ultralytics import YOLO
    from sixel import SixelWriter
    ```

1. Load a model and perform inference, then plot the results and store in a variable.

    ```py
    model = YOLO("yolov8n.pt")
    results = model.predict(
        source="ultralytics/assets/bus.jpg",
        )

    # Plot inference results
    plot = results[0].plot()
    ```

1. Now, use OpenCV to convert the `numpy` array to `bytes` data. Then use `io.BytesIO` to make a "file-like" object.

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

## Full Code Example

```{ .py .annotate }
import io

import cv2 as cv

from ultralytics import YOLO
from sixel import SixelWriter

model = YOLO("yolov8n.pt")
results = model.predict(source="ultralytics/assets/bus.jpg")

# Plot inference results
plot = results[0].plot()

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

---

!!! tip
    You may need to use `clear` to "erase" the view of the image in the terminal.