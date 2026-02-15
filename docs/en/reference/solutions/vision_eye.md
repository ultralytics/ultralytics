---
description: Discover the Ultralytics VisionEye solution for object tracking and analysis. Learn how to initialize parameters, map vision points, and track objects in real-time.
keywords: Ultralytics, VisionEye, Object Tracking, Computer Vision, Real-time Analysis, Python, AI
---

# Reference for `ultralytics/solutions/vision_eye.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/vision_eye.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/vision_eye.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`VisionEye`](#ultralytics.solutions.vision_eye.VisionEye)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`VisionEye.process`](#ultralytics.solutions.vision_eye.VisionEye.process)


## Class `ultralytics.solutions.vision_eye.VisionEye` {#ultralytics.solutions.vision\_eye.VisionEye}

```python
VisionEye(self, **kwargs: Any) -> None
```

**Bases:** `BaseSolution`

A class to manage object detection and vision mapping in images or video streams.

This class extends the BaseSolution class and provides functionality for detecting objects, mapping vision points, and annotating results with bounding boxes and labels.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `**kwargs` | `Any` | Keyword arguments passed to the parent class and for configuring vision_point. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `vision_point` | `tuple[int, int]` | Coordinates (x, y) where vision will view objects and draw tracks. |

**Methods**

| Name | Description |
| --- | --- |
| [`process`](#ultralytics.solutions.vision_eye.VisionEye.process) | Perform object detection, vision mapping, and annotation on the input image. |

**Examples**

```python
>>> vision_eye = VisionEye()
>>> frame = cv2.imread("frame.jpg")
>>> results = vision_eye.process(frame)
>>> print(f"Total detected instances: {results.total_tracks}")
```

<details>
<summary>Source code in <code>ultralytics/solutions/vision_eye.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/vision_eye.py#L9-L67"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class VisionEye(BaseSolution):
    """A class to manage object detection and vision mapping in images or video streams.

    This class extends the BaseSolution class and provides functionality for detecting objects, mapping vision points,
    and annotating results with bounding boxes and labels.

    Attributes:
        vision_point (tuple[int, int]): Coordinates (x, y) where vision will view objects and draw tracks.

    Methods:
        process: Process the input image to detect objects, annotate them, and apply vision mapping.

    Examples:
        >>> vision_eye = VisionEye()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = vision_eye.process(frame)
        >>> print(f"Total detected instances: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the VisionEye class for detecting objects and applying vision mapping.

        Args:
            **kwargs (Any): Keyword arguments passed to the parent class and for configuring vision_point.
        """
        super().__init__(**kwargs)
        # Set the vision point where the system will view objects and draw tracks
        self.vision_point = self.CFG["vision_point"]
```
</details>

<br>

### Method `ultralytics.solutions.vision_eye.VisionEye.process` {#ultralytics.solutions.vision\_eye.VisionEye.process}

```python
def process(self, im0) -> SolutionResults
```

Perform object detection, vision mapping, and annotation on the input image.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im0` | `np.ndarray` | The input image for detection and annotation. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `SolutionResults` | Object containing the annotated image and tracking statistics. |

**Examples**

```python
>>> vision_eye = VisionEye()
>>> frame = cv2.imread("image.jpg")
>>> results = vision_eye.process(frame)
>>> print(f"Detected {results.total_tracks} objects")
```

<details>
<summary>Source code in <code>ultralytics/solutions/vision_eye.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/vision_eye.py#L38-L67"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def process(self, im0) -> SolutionResults:
    """Perform object detection, vision mapping, and annotation on the input image.

    Args:
        im0 (np.ndarray): The input image for detection and annotation.

    Returns:
        (SolutionResults): Object containing the annotated image and tracking statistics.
            - plot_im: Annotated output image with bounding boxes and vision mapping
            - total_tracks: Number of tracked objects in the frame

    Examples:
        >>> vision_eye = VisionEye()
        >>> frame = cv2.imread("image.jpg")
        >>> results = vision_eye.process(frame)
        >>> print(f"Detected {results.total_tracks} objects")
    """
    self.extract_tracks(im0)  # Extract tracks (bounding boxes, classes, and masks)
    annotator = SolutionAnnotator(im0, self.line_width)

    for cls, t_id, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
        # Annotate the image with bounding boxes, labels, and vision mapping
        annotator.box_label(box, label=self.adjust_box_label(cls, conf, t_id), color=colors(int(t_id), True))
        annotator.visioneye(box, self.vision_point)

    plot_im = annotator.result()
    self.display_output(plot_im)  # Display the annotated output using the base class function

    # Return a SolutionResults object with the annotated image and tracking statistics
    return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
```
</details>

<br><br>
