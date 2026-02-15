---
description: Detailed documentation for the ObjectCropper class, part of the Ultralytics solutions package, enabling real-time cropping of detected objects from images and video streams.
keywords: Ultralytics, ObjectCropper, object detection, cropping, real-time processing, Python, computer vision
---

# Reference for `ultralytics/solutions/object_cropper.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/object_cropper.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/object_cropper.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`ObjectCropper`](#ultralytics.solutions.object_cropper.ObjectCropper)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`ObjectCropper.process`](#ultralytics.solutions.object_cropper.ObjectCropper.process)


## Class `ultralytics.solutions.object_cropper.ObjectCropper` {#ultralytics.solutions.object\_cropper.ObjectCropper}

```python
ObjectCropper(self, **kwargs: Any) -> None
```

**Bases:** `BaseSolution`

A class to manage the cropping of detected objects in a real-time video stream or images.

This class extends the BaseSolution class and provides functionality for cropping objects based on detected bounding boxes. The cropped images are saved to a specified directory for further analysis or usage.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `**kwargs` | `Any` | Keyword arguments passed to the parent class and used for configuration including:<br>    - crop_dir (str): Path to the directory for saving cropped object images. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `crop_dir` | `str` | Directory where cropped object images are stored. |
| `crop_idx` | `int` | Counter for the total number of cropped objects. |
| `iou` | `float` | IoU (Intersection over Union) threshold for non-maximum suppression. |
| `conf` | `float` | Confidence threshold for filtering detections. |

**Methods**

| Name | Description |
| --- | --- |
| [`process`](#ultralytics.solutions.object_cropper.ObjectCropper.process) | Crop detected objects from the input image and save them as separate images. |

**Examples**

```python
>>> cropper = ObjectCropper()
>>> frame = cv2.imread("frame.jpg")
>>> processed_results = cropper.process(frame)
>>> print(f"Total cropped objects: {cropper.crop_idx}")
```

<details>
<summary>Source code in <code>ultralytics/solutions/object_cropper.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/object_cropper.py#L10-L87"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ObjectCropper(BaseSolution):
    """A class to manage the cropping of detected objects in a real-time video stream or images.

    This class extends the BaseSolution class and provides functionality for cropping objects based on detected bounding
    boxes. The cropped images are saved to a specified directory for further analysis or usage.

    Attributes:
        crop_dir (str): Directory where cropped object images are stored.
        crop_idx (int): Counter for the total number of cropped objects.
        iou (float): IoU (Intersection over Union) threshold for non-maximum suppression.
        conf (float): Confidence threshold for filtering detections.

    Methods:
        process: Crop detected objects from the input image and save them to the output directory.

    Examples:
        >>> cropper = ObjectCropper()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_results = cropper.process(frame)
        >>> print(f"Total cropped objects: {cropper.crop_idx}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the ObjectCropper class for cropping objects from detected bounding boxes.

        Args:
            **kwargs (Any): Keyword arguments passed to the parent class and used for configuration including:
                - crop_dir (str): Path to the directory for saving cropped object images.
        """
        super().__init__(**kwargs)

        self.crop_dir = self.CFG["crop_dir"]  # Directory for storing cropped detections
        Path(self.crop_dir).mkdir(parents=True, exist_ok=True)
        if self.CFG["show"]:
            self.LOGGER.warning(f"show=True is not supported for ObjectCropper; saving crops to '{self.crop_dir}'.")
            self.CFG["show"] = False
        self.crop_idx = 0  # Initialize counter for total cropped objects
        self.iou = self.CFG["iou"]
        self.conf = self.CFG["conf"]
```
</details>

<br>

### Method `ultralytics.solutions.object_cropper.ObjectCropper.process` {#ultralytics.solutions.object\_cropper.ObjectCropper.process}

```python
def process(self, im0) -> SolutionResults
```

Crop detected objects from the input image and save them as separate images.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im0` | `np.ndarray` | The input image containing detected objects. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `SolutionResults` | A SolutionResults object containing the total number of cropped objects and processed |

**Examples**

```python
>>> cropper = ObjectCropper()
>>> frame = cv2.imread("image.jpg")
>>> results = cropper.process(frame)
>>> print(f"Total cropped objects: {results.total_crop_objects}")
```

<details>
<summary>Source code in <code>ultralytics/solutions/object_cropper.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/object_cropper.py#L50-L87"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def process(self, im0) -> SolutionResults:
    """Crop detected objects from the input image and save them as separate images.

    Args:
        im0 (np.ndarray): The input image containing detected objects.

    Returns:
        (SolutionResults): A SolutionResults object containing the total number of cropped objects and processed
            image.

    Examples:
        >>> cropper = ObjectCropper()
        >>> frame = cv2.imread("image.jpg")
        >>> results = cropper.process(frame)
        >>> print(f"Total cropped objects: {results.total_crop_objects}")
    """
    with self.profilers[0]:
        results = self.model.predict(
            im0,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            device=self.CFG["device"],
            verbose=False,
        )[0]
        self.clss = results.boxes.cls.tolist()  # required for logging only.

    for box in results.boxes:
        self.crop_idx += 1
        save_one_box(
            box.xyxy,
            im0,
            file=Path(self.crop_dir) / f"crop_{self.crop_idx}.jpg",
            BGR=True,
        )

    # Return SolutionResults
    return SolutionResults(plot_im=im0, total_crop_objects=self.crop_idx)
```
</details>

<br><br>
