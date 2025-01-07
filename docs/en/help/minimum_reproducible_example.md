---
comments: true
description: Learn how to create effective Minimum Reproducible Examples (MRE) for bug reports in Ultralytics YOLO repositories. Follow our guide for efficient issue resolution.
keywords: Ultralytics, YOLO, Minimum Reproducible Example, MRE, bug report, issue resolution, machine learning, deep learning
---

# Creating a Minimum Reproducible Example for Bug Reports in Ultralytics YOLO Repositories

When submitting a bug report for [Ultralytics](https://www.ultralytics.com/) [YOLO](https://github.com/ultralytics) repositories, it's essential to provide a [Minimum Reproducible Example (MRE)](https://stackoverflow.com/help/minimal-reproducible-example). An MRE is a small, self-contained piece of code that demonstrates the problem you're experiencing. Providing an MRE helps maintainers and contributors understand the issue and work on a fix more efficiently. This guide explains how to create an MRE when submitting bug reports to Ultralytics YOLO repositories.

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

## FAQ

### How do I create an effective Minimum Reproducible Example (MRE) for bug reports in Ultralytics YOLO repositories?

To create an effective Minimum Reproducible Example (MRE) for bug reports in Ultralytics YOLO repositories, follow these steps:

1. **Isolate the Problem**: Remove any code or dependencies that are not directly related to the issue.
2. **Use Public Models and Datasets**: Utilize public resources like `yolov8n.pt` and `coco8.yaml` for easier reproducibility.
3. **Include All Necessary Dependencies**: Specify required packages and their versions. You can list dependencies using `yolo checks` if you have `ultralytics` installed or `pip list`.
4. **Write a Clear Description of the Issue**: Explain the expected and actual behavior, including any error messages or logs.
5. **Format Your Code Properly**: Use code blocks to format your code, making it easier to read.
6. **Test Your MRE**: Ensure your MRE reproduces the issue without modifications.

For a detailed guide, see [Creating a Minimum Reproducible Example](#creating-a-minimum-reproducible-example-for-bug-reports-in-ultralytics-yolo-repositories).

### Why should I use publicly available models and datasets in my MRE for Ultralytics YOLO bug reports?

Using publicly available models and datasets in your MRE ensures that maintainers can easily run your example without needing access to proprietary data. This allows for quicker and more efficient issue resolution. For instance, using the `yolov8n.pt` model and `coco8.yaml` dataset helps standardize and simplify the debugging process. Learn more about public models and datasets in the [Use Public Models and Datasets](#2-use-public-models-and-datasets) section.

### What information should I include in my bug report for Ultralytics YOLO?

A comprehensive bug report for Ultralytics YOLO should include:

- **Clear Description**: Explain the issue, expected behavior, and actual behavior.
- **Error Messages**: Include any relevant error messages or logs.
- **Dependencies**: List required dependencies and their versions.
- **MRE**: Provide a Minimum Reproducible Example.
- **Steps to Reproduce**: Outline the steps needed to reproduce the issue.

For a complete checklist, refer to the [Write a Clear Description of the Issue](#4-write-a-clear-description-of-the-issue) section.

### How can I format my code properly when submitting a bug report on GitHub?

To format your code properly when submitting a bug report on GitHub:

- Use triple backticks (\```) to create code blocks.
- Specify the programming language for syntax highlighting, e.g., \```python.
- Ensure your code is indented correctly for readability.

Example:

````bash
```python
# Your Python code goes here
```
````

For more tips on code formatting, see [Format Your Code Properly](#5-format-your-code-properly).

### What are some common errors to check before submitting my MRE for a bug report?

Before submitting your MRE, make sure to:

- Verify the issue is reproducible.
- Ensure all dependencies are listed and correct.
- Remove any unnecessary code.
- Test the MRE to ensure it reproduces the issue without modifications.

For a detailed checklist, visit the [Test Your MRE](#6-test-your-mre) section.
