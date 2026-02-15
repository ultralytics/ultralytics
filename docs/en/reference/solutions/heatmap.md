---
description: Learn how to use the Ultralytics Heatmap module for real-time video analysis with object tracking and heatmap generation.
keywords: Ultralytics, Heatmap, Python, Real-time Video, Object Tracking, cv2, Shapely, Computer Vision, AI
---

# Reference for `ultralytics/solutions/heatmap.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/heatmap.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/heatmap.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`Heatmap`](#ultralytics.solutions.heatmap.Heatmap)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`Heatmap.heatmap_effect`](#ultralytics.solutions.heatmap.Heatmap.heatmap_effect)
        - [`Heatmap.process`](#ultralytics.solutions.heatmap.Heatmap.process)


## Class `ultralytics.solutions.heatmap.Heatmap` {#ultralytics.solutions.heatmap.Heatmap}

```python
Heatmap(self, **kwargs: Any) -> None
```

**Bases:** `ObjectCounter`

A class to draw heatmaps in real-time video streams based on object tracks.

This class extends the ObjectCounter class to generate and visualize heatmaps of object movements in video streams. It uses tracked object positions to create a cumulative heatmap effect over time.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `**kwargs` | `Any` | Keyword arguments passed to the parent ObjectCounter class. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `initialized` | `bool` | Flag indicating whether the heatmap has been initialized. |
| `colormap` | `int` | OpenCV colormap used for heatmap visualization. |
| `heatmap` | `np.ndarray` | Array storing the cumulative heatmap data. |
| `annotator` | `SolutionAnnotator` | Object for drawing annotations on the image. |

**Methods**

| Name | Description |
| --- | --- |
| [`heatmap_effect`](#ultralytics.solutions.heatmap.Heatmap.heatmap_effect) | Efficiently calculate heatmap area and effect location for applying colormap. |
| [`process`](#ultralytics.solutions.heatmap.Heatmap.process) | Generate heatmap for each frame using Ultralytics tracking. |

**Examples**

```python
>>> from ultralytics.solutions import Heatmap
>>> heatmap = Heatmap(model="yolo26n.pt", colormap=cv2.COLORMAP_JET)
>>> frame = cv2.imread("frame.jpg")
>>> processed_frame = heatmap.process(frame)
```

<details>
<summary>Source code in <code>ultralytics/solutions/heatmap.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/heatmap.py#L14-L125"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Heatmap(ObjectCounter):
    """A class to draw heatmaps in real-time video streams based on object tracks.

    This class extends the ObjectCounter class to generate and visualize heatmaps of object movements in video
    streams. It uses tracked object positions to create a cumulative heatmap effect over time.

    Attributes:
        initialized (bool): Flag indicating whether the heatmap has been initialized.
        colormap (int): OpenCV colormap used for heatmap visualization.
        heatmap (np.ndarray): Array storing the cumulative heatmap data.
        annotator (SolutionAnnotator): Object for drawing annotations on the image.

    Methods:
        heatmap_effect: Calculate and update the heatmap effect for a given bounding box.
        process: Generate and apply the heatmap effect to each frame.

    Examples:
        >>> from ultralytics.solutions import Heatmap
        >>> heatmap = Heatmap(model="yolo26n.pt", colormap=cv2.COLORMAP_JET)
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = heatmap.process(frame)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Heatmap class for real-time video stream heatmap generation based on object tracks.

        Args:
            **kwargs (Any): Keyword arguments passed to the parent ObjectCounter class.
        """
        super().__init__(**kwargs)

        self.initialized = False  # Flag for heatmap initialization
        if self.region is not None:  # Check if user provided the region coordinates
            self.initialize_region()

        # Store colormap
        self.colormap = self.CFG["colormap"]
        self.heatmap = None
```
</details>

<br>

### Method `ultralytics.solutions.heatmap.Heatmap.heatmap_effect` {#ultralytics.solutions.heatmap.Heatmap.heatmap\_effect}

```python
def heatmap_effect(self, box: list[float]) -> None
```

Efficiently calculate heatmap area and effect location for applying colormap.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `box` | `list[float]` | Bounding box coordinates [x0, y0, x1, y1]. | *required* |

<details>
<summary>Source code in <code>ultralytics/solutions/heatmap.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/heatmap.py#L53-L72"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def heatmap_effect(self, box: list[float]) -> None:
    """Efficiently calculate heatmap area and effect location for applying colormap.

    Args:
        box (list[float]): Bounding box coordinates [x0, y0, x1, y1].
    """
    x0, y0, x1, y1 = map(int, box)
    radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

    # Create a meshgrid with region of interest (ROI) for vectorized distance calculations
    xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

    # Calculate squared distances from the center
    dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

    # Create a mask of points within the radius
    within_radius = dist_squared <= radius_squared

    # Update only the values within the bounding box in a single vectorized operation
    self.heatmap[y0:y1, x0:x1][within_radius] += 2
```
</details>

<br>

### Method `ultralytics.solutions.heatmap.Heatmap.process` {#ultralytics.solutions.heatmap.Heatmap.process}

```python
def process(self, im0: np.ndarray) -> SolutionResults
```

Generate heatmap for each frame using Ultralytics tracking.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im0` | `np.ndarray` | Input image array for processing. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `SolutionResults` | Contains processed image `plot_im`, 'in_count' (int, count of objects entering the |

<details>
<summary>Source code in <code>ultralytics/solutions/heatmap.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/heatmap.py#L74-L125"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def process(self, im0: np.ndarray) -> SolutionResults:
    """Generate heatmap for each frame using Ultralytics tracking.

    Args:
        im0 (np.ndarray): Input image array for processing.

    Returns:
        (SolutionResults): Contains processed image `plot_im`, 'in_count' (int, count of objects entering the
            region), 'out_count' (int, count of objects exiting the region), 'classwise_count' (dict, per-class
            object count), and 'total_tracks' (int, total number of tracked objects).
    """
    if not self.initialized:
        self.heatmap = np.zeros_like(im0, dtype=np.float32) * 0.99
        self.initialized = True  # Initialize heatmap only once

    self.extract_tracks(im0)  # Extract tracks
    self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

    # Iterate over bounding boxes, track ids and classes index
    for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
        # Apply heatmap effect for the bounding box
        self.heatmap_effect(box)

        if self.region is not None:
            self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)
            self.store_tracking_history(track_id, box)  # Store track history
            # Get previous position if available
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)  # object counting

    plot_im = self.annotator.result()
    if self.region is not None:
        self.display_counts(plot_im)  # Display the counts on the frame

    # Normalize, apply colormap to heatmap and combine with original image
    if self.track_data.is_track:
        normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized_heatmap, self.colormap)
        plot_im = cv2.addWeighted(plot_im, 0.5, colored_heatmap, 0.5, 0)

    self.display_output(plot_im)  # Display output with base class function

    # Return SolutionResults
    return SolutionResults(
        plot_im=plot_im,
        in_count=self.in_count,
        out_count=self.out_count,
        classwise_count=dict(self.classwise_count),
        total_tracks=len(self.track_ids),
    )
```
</details>

<br><br>
