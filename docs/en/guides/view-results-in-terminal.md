---
comments: true
description: Display YOLO inference results directly in a VSCode terminal with the sixel protocol on Linux and macOS, ideal for remote SSH sessions and headless machines without a GUI.
keywords: YOLO, inference results, VSCode terminal, sixel, display images in terminal, remote SSH, headless server, no GUI, Linux, macOS, iTerm2, image visualization, Ultralytics
---

# How to View YOLO Inference Results in a VSCode Terminal

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/sixel-example-terminal.avif" alt="Sixel example of image in Terminal">
</p>

Image from the [libsixel](https://saitoha.github.io/libsixel/) website.

## Motivation

When connecting to a remote machine over SSH, visualizing image results normally is not possible or requires moving data to a local device with a GUI. The VSCode integrated terminal can render images directly, so you can inspect [prediction results](../modes/predict.md) right where you run inference, without copying files back to your laptop. This guide walks through enabling the [setup](#process), wiring up the [code](#full-code-example), and answers [common questions](#faq).

!!! warning "Linux and macOS only"

    Only compatible with Linux and macOS. Check the [VSCode repository](https://github.com/microsoft/vscode), check [Issue status](https://github.com/microsoft/vscode/issues/198622), or [documentation](https://code.visualstudio.com/docs) for updates about Windows support to view images in terminal with `sixel`.

The VSCode compatible protocols for viewing images using the integrated terminal are [`sixel`](https://en.wikipedia.org/wiki/Sixel) and [`iTerm`](https://iterm2.com/documentation-images.html). This guide will demonstrate use of the `sixel` protocol.

## Process

1. First, you must enable settings `terminal.integrated.enableImages` and `terminal.integrated.gpuAcceleration` in VSCode.

    ```yaml
    "terminal.integrated.gpuAcceleration": "auto" # "auto" is default, can also use "on"
    "terminal.integrated.enableImages": true
    ```

    <p align="center">
      <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vscode-enable-terminal-images-setting.avif" alt="VSCode enable terminal images setting">
    </p>

2. Install the `python-sixel` library in your virtual environment. This is a [fork](https://github.com/lubosz/python-sixel?tab=readme-ov-file) of the `PySixel` library, which is no longer maintained.

    ```bash
    pip install sixel
    ```

3. Load a model and execute inference, then plot the results and store in a variable. See more about inference arguments and working with results on the [predict mode](../modes/predict.md) page.

    ```{ .py .annotate }
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo26n.pt")

    # Run inference on an image
    results = model.predict(source="ultralytics/assets/bus.jpg")

    # Plot inference results
    plot = results[0].plot()  # (1)!
    ```

    1. See [plot method parameters](../modes/predict.md#plot-method-parameters) to see possible arguments to use.

4. Now, use [OpenCV](https://www.ultralytics.com/glossary/opencv) to convert the `np.ndarray` to `bytes` data. Then use `io.BytesIO` to make a "file-like" object.

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
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/view-image-in-terminal.avif" alt="YOLO inference results displayed in terminal">
</p>

!!! danger "Videos and GIFs are untested"

    Using this example with videos or animated GIF frames has **not** been tested. Attempt at your own risk.

## Full Code Example

```{ .py .annotate }
import io

import cv2
from sixel import SixelWriter

from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

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

!!! tip "Clearing the image"

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

    model = YOLO("yolo26n.pt")
    results = model.predict(source="ultralytics/assets/bus.jpg")
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

### What if I encounter issues displaying images in the VSCode terminal?

If nothing renders, work through these checks in order:

1. Confirm both `terminal.integrated.enableImages` and `terminal.integrated.gpuAcceleration` are enabled, as shown in the [Process](#process) section, then restart the integrated terminal so the settings take effect.
2. Verify that `sixel` is installed in the same virtual environment your script runs in (`pip install sixel`).
3. Make sure `plot` is a valid `np.ndarray` before encoding it, since `cv2.imencode` raises an error if the image is empty or not a valid array. See the [plot method parameters](../modes/predict.md#plot-method-parameters) for the values `results[0].plot()` accepts.

If problems persist, consult the [VSCode repository](https://github.com/microsoft/vscode) for terminal-image support status.

### Can YOLO display video inference results in the terminal using sixel?

Displaying video inference results or animated GIF frames using sixel in the terminal is currently untested and may not be supported. We recommend starting with static images and verifying compatibility. Attempt video results at your own risk, keeping in mind performance constraints. For more information on plotting inference results, visit the [predict mode](../modes/predict.md) page.

### How can I troubleshoot issues with the `python-sixel` library?

The `sixel` package is a [fork](https://github.com/lubosz/python-sixel) of the unmaintained `PySixel` library and is a thin Python wrapper over [Pillow](https://python-pillow.github.io/) (PIL). If `import sixel` fails or `SixelWriter().draw()` raises an error, confirm the package installed into your active virtual environment, ensure Pillow is available, and check the [python-sixel GitHub repository](https://github.com/lubosz/python-sixel) for platform-specific notes. For more on generating the image you pass to `draw()`, see the [predict mode](../modes/predict.md) documentation.
