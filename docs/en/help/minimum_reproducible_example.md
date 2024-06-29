---
comments: true
description: Learn how to create effective Minimum Reproducible Examples (MRE) for bug reports in Ultralytics YOLO repositories. Follow our guide for efficient issue resolution.
keywords: Ultralytics, YOLO, Minimum Reproducible Example, MRE, bug report, issue resolution, machine learning, deep learning
---

# Creating a Minimum Reproducible Example for Bug Reports in Ultralytics YOLO Repositories

When submitting a bug report for [Ultralytics](https://ultralytics.com) [YOLO](https://github.com/ultralytics) repositories, it's essential to provide a [Minimum Reproducible Example (MRE)](https://stackoverflow.com/help/minimal-reproducible-example). An MRE is a small, self-contained piece of code that demonstrates the problem you're experiencing. Providing an MRE helps maintainers and contributors understand the issue and work on a fix more efficiently. This guide explains how to create an MRE when submitting bug reports to Ultralytics YOLO repositories.

## 1. Isolate the Problem

The first step in creating an MRE is to isolate the problem. Remove any unnecessary code or dependencies that are not directly related to the issue. Focus on the specific part of the code that is causing the problem and eliminate any irrelevant sections.

## 2. Use Public Models and Datasets

When creating an MRE, use publicly available models and datasets to reproduce the issue. For example, use the `yolov8n.pt` model and the `coco8.yaml` dataset. This ensures that the maintainers and contributors can easily run your example and investigate the problem without needing access to proprietary data or custom models.

## 3. Include All Necessary Dependencies

Ensure all necessary dependencies are included in your MRE. If your code relies on external libraries, specify the required packages and their versions. Ideally, list the dependencies in your bug report using `yolo checks` if you have `ultralytics` installed or `pip list` for other tools.

## 4. Write a Clear Description of the Issue

Provide a clear and concise description of the issue you're experiencing. Explain the expected behavior and the actual behavior you're encountering. If applicable, include any relevant error messages or logs.

## 5. Format Your Code Properly

Format your code properly using code blocks in the issue description. This makes it easier for others to read and understand your code. In GitHub, you can create a code block by wrapping your code with triple backticks (\```) and specifying the language:

````bash
```python
# Your Python code goes here
```
````

## 6. Test Your MRE

Before submitting your MRE, test it to ensure that it accurately reproduces the issue. Make sure that others can run your example without any issues or modifications.

## Example of an MRE

Here's an example of an MRE for a hypothetical bug report:

**Bug description:**

When running inference on a 0-channel image, I get an error related to the dimensions of the input tensor.

**MRE:**

```python
import torch

from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Load a 0-channel image
image = torch.rand(1, 0, 640, 640)

# Run the model
results = model(image)
```

**Error message:**

```
RuntimeError: Expected input[1, 0, 640, 640] to have 3 channels, but got 0 channels instead
```

**Dependencies:**

- `torch==2.3.0`
- `ultralytics==8.2.0`

In this example, the MRE demonstrates the issue with a minimal amount of code, uses a public model (`"yolov8n.pt"`), includes all necessary dependencies, and provides a clear description of the problem along with the error message.

By following these guidelines, you'll help the maintainers and [contributors](https://github.com/ultralytics/ultralytics/graphs/contributors) of Ultralytics YOLO repositories to understand and resolve your issue more efficiently.
