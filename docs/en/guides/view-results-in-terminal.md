---
comments: true
description: Learn how to visualize YOLO inference results directly in a VSCode terminal using sixel on Linux and MacOS.
keywords: YOLO, inference results, VSCode terminal, sixel, display images, Linux, MacOS
---

# Viewing Inference Results in a Terminal

<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/saitoha/libsixel/data/data/sixel.gif" alt="Sixel example of image in Terminal">
</p>

Image from the [libsixel](https://saitoha.github.io/libsixel/) website.

## Motivation

When connecting to a remote machine, normally visualizing image results is not possible or requires moving data to a local device with a GUI. The VSCode integrated terminal allows for directly rendering images. This is a short demonstration on how to use this in conjunction with `ultralytics` with [prediction results](../modes/predict.md).

!!! warning

    Only compatible with Linux and MacOS. Check the [VSCode repository](https://github.com/microsoft/vscode), check [Issue status](https://github.com/microsoft/vscode/issues/198622), or [documentation](https://code.visualstudio.com/docs) for updates about Windows support to view images in terminal with `sixel`.

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

2. Install the `python-sixel` library in your virtual environment. This is a [fork](https://github.com/lubosz/python-sixel?tab=readme-ov-file) of the `PySixel` library, which is no longer maintained.

    ```bash
    pip install sixel
    ```

3. Load a model and execute inference, then plot the results and store in a variable. See more about inference arguments and working with results on the [predict mode](../modes/predict.md) page.

    ```{ .py .annotate }
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolov8n.pt")

    # Run inference on an image
    results = model.predict(source="ultralytics/assets/bus.jpg")

    # Plot inference results
    plot = results[0].plot()  # (1)!
    ```

    1. See [plot method parameters](../modes/predict.md#plot-method-parameters) to see possible arguments to use.

4. Now, use OpenCV to convert the `numpy.ndarray` to `bytes` data. Then use `io.BytesIO` to make a "file-like" object.

    ```{ .py .annotate }
    import io

    import cv2

    # Results image as bytes
    im_bytes = cv2.imencode(
        ".png",  # (1)!
        plot,
    )[1].tobytes()  # (2)!

    # Image bytes as a file-like object
    mem_file = io.BytesIO(im_bytes)
    ```

    1. It's possible to use other image extensions as well.
    2. Only the object at index `1` that is returned is needed.

5. Create a `SixelWriter` instance, and then use the `.draw()` method to draw the image in the terminal.

    ```python
    from sixel import SixelWriter

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

import cv2
from sixel import SixelWriter

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model.predict(source="ultralytics/assets/bus.jpg")

# Plot inference results
plot = results[0].plot()  # (3)!

# Results image as bytes
im_bytes = cv2.imencode(
    ".png",  # (1)!
    plot,
)[1].tobytes()  # (2)!

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

## FAQ

### How can I view YOLO inference results in a VSCode terminal on macOS or Linux?

To view YOLO inference results in a VSCode terminal on macOS or Linux, follow these steps:

1. Enable the necessary VSCode settings:

    ```yaml
    "terminal.integrated.enableImages": true
    "terminal.integrated.gpuAcceleration": "auto"
    ```

2. Install the sixel library:

    ```bash
    pip install sixel
    ```

3. Load your YOLO model and run inference:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model.predict(source="path_to_image")
    plot = results[0].plot()
    ```

4. Convert the inference result image to bytes and display it in the terminal:

    ```python
    import io

    import cv2
    from sixel import SixelWriter

    im_bytes = cv2.imencode(".png", plot)[1].tobytes()
    mem_file = io.BytesIO(im_bytes)
    SixelWriter().draw(mem_file)
    ```

For further details, visit the [predict mode](../modes/predict.md) page.

### Why does the sixel protocol only work on Linux and macOS?

The sixel protocol is currently only supported on Linux and macOS because these platforms have native terminal capabilities compatible with sixel graphics. Windows support for terminal graphics using sixel is still under development. For updates on Windows compatibility, check the [VSCode Issue status](https://github.com/microsoft/vscode/issues/198622) and [documentation](https://code.visualstudio.com/docs).

### What if I encounter issues with displaying images in the VSCode terminal?

If you encounter issues displaying images in the VSCode terminal using sixel:

1. Ensure the necessary settings in VSCode are enabled:

    ```yaml
    "terminal.integrated.enableImages": true
    "terminal.integrated.gpuAcceleration": "auto"
    ```

2. Verify the sixel library installation:

    ```bash
    pip install sixel
    ```

3. Check your image data conversion and plotting code for errors. For example:

    ```python
    import io

    import cv2
    from sixel import SixelWriter

    im_bytes = cv2.imencode(".png", plot)[1].tobytes()
    mem_file = io.BytesIO(im_bytes)
    SixelWriter().draw(mem_file)
    ```

If problems persist, consult the [VSCode repository](https://github.com/microsoft/vscode), and visit the [plot method parameters](../modes/predict.md#plot-method-parameters) section for additional guidance.

### Can YOLO display video inference results in the terminal using sixel?

Displaying video inference results or animated GIF frames using sixel in the terminal is currently untested and may not be supported. We recommend starting with static images and verifying compatibility. Attempt video results at your own risk, keeping in mind performance constraints. For more information on plotting inference results, visit the [predict mode](../modes/predict.md) page.

### How can I troubleshoot issues with the `python-sixel` library?

To troubleshoot issues with the `python-sixel` library:

1. Ensure the library is correctly installed in your virtual environment:

    ```bash
    pip install sixel
    ```

2. Verify that you have the necessary Python and system dependencies.

3. Refer to the [python-sixel GitHub repository](https://github.com/lubosz/python-sixel) for additional documentation and community support.

4. Double-check your code for potential errors, specifically the usage of `SixelWriter` and image data conversion steps.

For further assistance on working with YOLO models and sixel integration, see the [export](../modes/export.md) and [predict mode](../modes/predict.md) documentation pages.
