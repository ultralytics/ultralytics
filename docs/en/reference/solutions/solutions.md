---
description: Explore the Ultralytics Solution Base class for real-time object counting,virtual gym, heatmaps, speed estimation using Ultralytics YOLO. Learn to implement Ultralytics solutions effectively.
keywords: Ultralytics, Solutions, Object counting, Speed Estimation, Heatmaps, Queue Management, AI Gym, YOLO, pose detection, gym step counting, real-time pose estimation, Python
---

# Reference for `ultralytics/solutions/solutions.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`BaseSolution`](#ultralytics.solutions.solutions.BaseSolution)
        - [`SolutionAnnotator`](#ultralytics.solutions.solutions.SolutionAnnotator)
        - [`SolutionResults`](#ultralytics.solutions.solutions.SolutionResults)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`BaseSolution.adjust_box_label`](#ultralytics.solutions.solutions.BaseSolution.adjust_box_label)
        - [`BaseSolution.extract_tracks`](#ultralytics.solutions.solutions.BaseSolution.extract_tracks)
        - [`BaseSolution.store_tracking_history`](#ultralytics.solutions.solutions.BaseSolution.store_tracking_history)
        - [`BaseSolution.initialize_region`](#ultralytics.solutions.solutions.BaseSolution.initialize_region)
        - [`BaseSolution.display_output`](#ultralytics.solutions.solutions.BaseSolution.display_output)
        - [`BaseSolution.process`](#ultralytics.solutions.solutions.BaseSolution.process)
        - [`BaseSolution.__call__`](#ultralytics.solutions.solutions.BaseSolution.__call__)
        - [`SolutionAnnotator.draw_region`](#ultralytics.solutions.solutions.SolutionAnnotator.draw_region)
        - [`SolutionAnnotator.queue_counts_display`](#ultralytics.solutions.solutions.SolutionAnnotator.queue_counts_display)
        - [`SolutionAnnotator.display_analytics`](#ultralytics.solutions.solutions.SolutionAnnotator.display_analytics)
        - [`SolutionAnnotator._point_xy`](#ultralytics.solutions.solutions.SolutionAnnotator._point_xy)
        - [`SolutionAnnotator._estimate_pose_angle_cached`](#ultralytics.solutions.solutions.SolutionAnnotator._estimate_pose_angle_cached)
        - [`SolutionAnnotator.estimate_pose_angle`](#ultralytics.solutions.solutions.SolutionAnnotator.estimate_pose_angle)
        - [`SolutionAnnotator.draw_specific_kpts`](#ultralytics.solutions.solutions.SolutionAnnotator.draw_specific_kpts)
        - [`SolutionAnnotator.plot_workout_information`](#ultralytics.solutions.solutions.SolutionAnnotator.plot_workout_information)
        - [`SolutionAnnotator.plot_angle_and_count_and_stage`](#ultralytics.solutions.solutions.SolutionAnnotator.plot_angle_and_count_and_stage)
        - [`SolutionAnnotator.plot_distance_and_line`](#ultralytics.solutions.solutions.SolutionAnnotator.plot_distance_and_line)
        - [`SolutionAnnotator.display_objects_labels`](#ultralytics.solutions.solutions.SolutionAnnotator.display_objects_labels)
        - [`SolutionAnnotator.sweep_annotator`](#ultralytics.solutions.solutions.SolutionAnnotator.sweep_annotator)
        - [`SolutionAnnotator.visioneye`](#ultralytics.solutions.solutions.SolutionAnnotator.visioneye)
        - [`SolutionAnnotator.adaptive_label`](#ultralytics.solutions.solutions.SolutionAnnotator.adaptive_label)
        - [`SolutionResults.__str__`](#ultralytics.solutions.solutions.SolutionResults.__str__)


## Class `ultralytics.solutions.solutions.BaseSolution` {#ultralytics.solutions.solutions.BaseSolution}

```python
BaseSolution(self, is_cli: bool = False, **kwargs: Any) -> None
```

A base class for managing Ultralytics Solutions.

This class provides core functionality for various Ultralytics Solutions, including model loading, object tracking, and region initialization. It serves as the foundation for implementing specific computer vision solutions such as object counting, pose estimation, and analytics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `is_cli` | `bool` | Enable CLI mode if set to True. | `False` |
| `**kwargs` | `Any` | Additional configuration parameters that override defaults. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `LineString` |  | Class for creating line string geometries from shapely. |
| `Polygon` |  | Class for creating polygon geometries from shapely. |
| `Point` |  | Class for creating point geometries from shapely. |
| `prep` |  | Prepared geometry function from shapely for optimized spatial operations. |
| `CFG` | `dict[str, Any]` | Configuration dictionary loaded from YAML file and updated with kwargs. |
| `LOGGER` |  | Logger instance for solution-specific logging. |
| `annotator` |  | Annotator instance for drawing on images. |
| `tracks` |  | YOLO tracking results from the latest inference. |
| `track_data` |  | Extracted tracking data (boxes or OBB) from tracks. |
| `boxes` | `list` | Bounding box coordinates from tracking results. |
| `clss` | `list[int]` | Class indices from tracking results. |
| `track_ids` | `list[int]` | Track IDs from tracking results. |
| `confs` | `list[float]` | Confidence scores from tracking results. |
| `track_line` |  | Current track line for storing tracking history. |
| `masks` |  | Segmentation masks from tracking results. |
| `r_s` |  | Region or line geometry object for spatial operations. |
| `frame_no` | `int` | Current frame number for logging purposes. |
| `region` | `list[tuple[int, int]]` | List of coordinate tuples defining region of interest. |
| `line_width` | `int` | Width of lines used in visualizations. |
| `model` | `YOLO` | Loaded YOLO model instance. |
| `names` | `dict[int, str]` | Dictionary mapping class indices to class names. |
| `classes` | `list[int]` | List of class indices to track. |
| `show_conf` | `bool` | Flag to show confidence scores in annotations. |
| `show_labels` | `bool` | Flag to show class labels in annotations. |
| `device` | `str` | Device for model inference. |
| `track_add_args` | `dict[str, Any]` | Additional arguments for tracking configuration. |
| `env_check` | `bool` | Flag indicating whether environment supports image display. |
| `track_history` | `defaultdict` | Dictionary storing tracking history for each object. |
| `profilers` | `tuple` | Profiler instances for performance monitoring. |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.solutions.solutions.BaseSolution.__call__) | Allow instances to be called like a function with flexible arguments. |
| [`adjust_box_label`](#ultralytics.solutions.solutions.BaseSolution.adjust_box_label) | Generate a formatted label for a bounding box. |
| [`display_output`](#ultralytics.solutions.solutions.BaseSolution.display_output) | Display the results of the processing, which could involve showing frames, printing counts, or saving |
| [`extract_tracks`](#ultralytics.solutions.solutions.BaseSolution.extract_tracks) | Apply object tracking and extract tracks from an input image or frame. |
| [`initialize_region`](#ultralytics.solutions.solutions.BaseSolution.initialize_region) | Initialize the counting region and line segment based on configuration settings. |
| [`process`](#ultralytics.solutions.solutions.BaseSolution.process) | Process method should be implemented by each Solution subclass. |
| [`store_tracking_history`](#ultralytics.solutions.solutions.BaseSolution.store_tracking_history) | Store the tracking history of an object. |

**Examples**

```python
>>> solution = BaseSolution(model="yolo26n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
>>> solution.initialize_region()
>>> image = cv2.imread("image.jpg")
>>> solution.extract_tracks(image)
>>> solution.display_output(image)
```

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L20-L258"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class BaseSolution:
    """A base class for managing Ultralytics Solutions.

    This class provides core functionality for various Ultralytics Solutions, including model loading, object tracking,
    and region initialization. It serves as the foundation for implementing specific computer vision solutions such as
    object counting, pose estimation, and analytics.

    Attributes:
        LineString: Class for creating line string geometries from shapely.
        Polygon: Class for creating polygon geometries from shapely.
        Point: Class for creating point geometries from shapely.
        prep: Prepared geometry function from shapely for optimized spatial operations.
        CFG (dict[str, Any]): Configuration dictionary loaded from YAML file and updated with kwargs.
        LOGGER: Logger instance for solution-specific logging.
        annotator: Annotator instance for drawing on images.
        tracks: YOLO tracking results from the latest inference.
        track_data: Extracted tracking data (boxes or OBB) from tracks.
        boxes (list): Bounding box coordinates from tracking results.
        clss (list[int]): Class indices from tracking results.
        track_ids (list[int]): Track IDs from tracking results.
        confs (list[float]): Confidence scores from tracking results.
        track_line: Current track line for storing tracking history.
        masks: Segmentation masks from tracking results.
        r_s: Region or line geometry object for spatial operations.
        frame_no (int): Current frame number for logging purposes.
        region (list[tuple[int, int]]): List of coordinate tuples defining region of interest.
        line_width (int): Width of lines used in visualizations.
        model (YOLO): Loaded YOLO model instance.
        names (dict[int, str]): Dictionary mapping class indices to class names.
        classes (list[int]): List of class indices to track.
        show_conf (bool): Flag to show confidence scores in annotations.
        show_labels (bool): Flag to show class labels in annotations.
        device (str): Device for model inference.
        track_add_args (dict[str, Any]): Additional arguments for tracking configuration.
        env_check (bool): Flag indicating whether environment supports image display.
        track_history (defaultdict): Dictionary storing tracking history for each object.
        profilers (tuple): Profiler instances for performance monitoring.

    Methods:
        adjust_box_label: Generate formatted label for bounding box.
        extract_tracks: Apply object tracking and extract tracks from input image.
        store_tracking_history: Store object tracking history for given track ID and bounding box.
        initialize_region: Initialize counting region and line segment based on configuration.
        display_output: Display processing results including frames or saved results.
        process: Process method to be implemented by each Solution subclass.

    Examples:
        >>> solution = BaseSolution(model="yolo26n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> solution.initialize_region()
        >>> image = cv2.imread("image.jpg")
        >>> solution.extract_tracks(image)
        >>> solution.display_output(image)
    """

    def __init__(self, is_cli: bool = False, **kwargs: Any) -> None:
        """Initialize the BaseSolution class with configuration settings and YOLO model.

        Args:
            is_cli (bool): Enable CLI mode if set to True.
            **kwargs (Any): Additional configuration parameters that override defaults.
        """
        self.CFG = vars(SolutionConfig().update(**kwargs))
        self.LOGGER = LOGGER  # Store logger object to be used in multiple solution classes

        check_requirements("shapely>=2.0.0")
        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep
        self.annotator = None  # Initialize annotator
        self.tracks = None
        self.track_data = None
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.masks = None
        self.r_s = None
        self.frame_no = -1  # Only for logging

        self.LOGGER.info(f"Ultralytics Solutions: ✅ {self.CFG}")
        self.region = self.CFG["region"]  # Store region data for other classes usage
        self.line_width = self.CFG["line_width"]

        # Load Model and store additional information (classes, show_conf, show_label)
        if self.CFG["model"] is None:
            self.CFG["model"] = "yolo26n.pt"
        self.model = YOLO(self.CFG["model"])
        self.names = self.model.names
        self.classes = self.CFG["classes"]
        self.show_conf = self.CFG["show_conf"]
        self.show_labels = self.CFG["show_labels"]
        self.device = self.CFG["device"]

        self.track_add_args = {  # Tracker additional arguments for advance configuration
            k: self.CFG[k] for k in {"iou", "conf", "device", "max_det", "half", "tracker"}
        }  # verbose must be passed to track method; setting it False in YOLO still logs the track information.

        if is_cli and self.CFG["source"] is None:
            d_s = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"
            self.LOGGER.warning(f"source not provided. using default source {ASSETS_URL}/{d_s}")
            from ultralytics.utils.downloads import safe_download

            safe_download(f"{ASSETS_URL}/{d_s}")  # download source from ultralytics assets
            self.CFG["source"] = d_s  # set default source

        # Initialize environment and region setup
        self.env_check = check_imshow(warn=True)
        self.track_history = defaultdict(list)

        self.profilers = (
            ops.Profile(device=self.device),  # track
            ops.Profile(device=self.device),  # solution
        )
```
</details>

<br>

### Method `ultralytics.solutions.solutions.BaseSolution.__call__` {#ultralytics.solutions.solutions.BaseSolution.\_\_call\_\_}

```python
def __call__(self, *args: Any, **kwargs: Any)
```

Allow instances to be called like a function with flexible arguments.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` | `Any` |  | *required* |
| `**kwargs` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L240-L258"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __call__(self, *args: Any, **kwargs: Any):
    """Allow instances to be called like a function with flexible arguments."""
    with self.profilers[1]:
        result = self.process(*args, **kwargs)  # Call the subclass-specific process method
    track_or_predict = "predict" if type(self).__name__ == "ObjectCropper" else "track"
    track_or_predict_speed = self.profilers[0].dt * 1e3
    solution_speed = (self.profilers[1].dt - self.profilers[0].dt) * 1e3  # solution time = process - track
    result.speed = {track_or_predict: track_or_predict_speed, "solution": solution_speed}
    if self.CFG["verbose"]:
        self.frame_no += 1
        counts = Counter(self.clss)  # Only for logging.
        LOGGER.info(
            f"{self.frame_no}: {result.plot_im.shape[0]}x{result.plot_im.shape[1]} {solution_speed:.1f}ms,"
            f" {', '.join([f'{v} {self.names[k]}' for k, v in counts.items()])}\n"
            f"Speed: {track_or_predict_speed:.1f}ms {track_or_predict}, "
            f"{solution_speed:.1f}ms solution per image at shape "
            f"(1, {getattr(self.model, 'ch', 3)}, {result.plot_im.shape[0]}, {result.plot_im.shape[1]})\n"
        )
    return result
```
</details>

<br>

### Method `ultralytics.solutions.solutions.BaseSolution.adjust_box_label` {#ultralytics.solutions.solutions.BaseSolution.adjust\_box\_label}

```python
def adjust_box_label(self, cls: int, conf: float, track_id: int | None = None) -> str | None
```

Generate a formatted label for a bounding box.

This method constructs a label string for a bounding box using the class index and confidence score. Optionally includes the track ID if provided. The label format adapts based on the display settings defined in `self.show_conf` and `self.show_labels`.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cls` | `int` | The class index of the detected object. | *required* |
| `conf` | `float` | The confidence score of the detection. | *required* |
| `track_id` | `int, optional` | The unique identifier for the tracked object. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `str | None` | The formatted label string if `self.show_labels` is True; otherwise, None. |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L138-L154"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def adjust_box_label(self, cls: int, conf: float, track_id: int | None = None) -> str | None:
    """Generate a formatted label for a bounding box.

    This method constructs a label string for a bounding box using the class index and confidence score. Optionally
    includes the track ID if provided. The label format adapts based on the display settings defined in
    `self.show_conf` and `self.show_labels`.

    Args:
        cls (int): The class index of the detected object.
        conf (float): The confidence score of the detection.
        track_id (int, optional): The unique identifier for the tracked object.

    Returns:
        (str | None): The formatted label string if `self.show_labels` is True; otherwise, None.
    """
    name = ("" if track_id is None else f"{track_id} ") + self.names[cls]
    return (f"{name} {conf:.2f}" if self.show_conf else name) if self.show_labels else None
```
</details>

<br>

### Method `ultralytics.solutions.solutions.BaseSolution.display_output` {#ultralytics.solutions.solutions.BaseSolution.display\_output}

```python
def display_output(self, plot_im: np.ndarray) -> None
```

Display the results of the processing, which could involve showing frames, printing counts, or saving

results.

This method is responsible for visualizing the output of the object detection and tracking process. It displays the processed frame with annotations, and allows for user interaction to close the display.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `plot_im` | `np.ndarray` | The image or frame that has been processed and annotated. | *required* |

**Examples**

```python
>>> solution = BaseSolution()
>>> frame = cv2.imread("path/to/image.jpg")
>>> solution.display_output(frame)
```

!!! note "Notes"

    - This method will only display output if the 'show' configuration is set to True and the environment
      supports image display.
    - The display can be closed by pressing the 'q' key.

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L211-L235"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def display_output(self, plot_im: np.ndarray) -> None:
    """Display the results of the processing, which could involve showing frames, printing counts, or saving
    results.

    This method is responsible for visualizing the output of the object detection and tracking process. It displays
    the processed frame with annotations, and allows for user interaction to close the display.

    Args:
        plot_im (np.ndarray): The image or frame that has been processed and annotated.

    Examples:
        >>> solution = BaseSolution()
        >>> frame = cv2.imread("path/to/image.jpg")
        >>> solution.display_output(frame)

    Notes:
        - This method will only display output if the 'show' configuration is set to True and the environment
          supports image display.
        - The display can be closed by pressing the 'q' key.
    """
    if self.CFG.get("show") and self.env_check:
        cv2.imshow("Ultralytics Solutions", plot_im)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()  # Closes current frame window
            return
```
</details>

<br>

### Method `ultralytics.solutions.solutions.BaseSolution.extract_tracks` {#ultralytics.solutions.solutions.BaseSolution.extract\_tracks}

```python
def extract_tracks(self, im0: np.ndarray) -> None
```

Apply object tracking and extract tracks from an input image or frame.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im0` | `np.ndarray` | The input image or frame. | *required* |

**Examples**

```python
>>> solution = BaseSolution()
>>> frame = cv2.imread("path/to/image.jpg")
>>> solution.extract_tracks(frame)
```

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L156-L181"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def extract_tracks(self, im0: np.ndarray) -> None:
    """Apply object tracking and extract tracks from an input image or frame.

    Args:
        im0 (np.ndarray): The input image or frame.

    Examples:
        >>> solution = BaseSolution()
        >>> frame = cv2.imread("path/to/image.jpg")
        >>> solution.extract_tracks(frame)
    """
    with self.profilers[0]:
        self.tracks = self.model.track(
            source=im0, persist=True, classes=self.classes, verbose=False, **self.track_add_args
        )[0]
    is_obb = self.tracks.obb is not None
    self.track_data = self.tracks.obb if is_obb else self.tracks.boxes  # Extract tracks for OBB or object detection

    if self.track_data and self.track_data.is_track:
        self.boxes = (self.track_data.xyxyxyxy if is_obb else self.track_data.xyxy).cpu()
        self.clss = self.track_data.cls.cpu().tolist()
        self.track_ids = self.track_data.id.int().cpu().tolist()
        self.confs = self.track_data.conf.cpu().tolist()
    else:
        self.LOGGER.warning("No tracks found.")
        self.boxes, self.clss, self.track_ids, self.confs = [], [], [], []
```
</details>

<br>

### Method `ultralytics.solutions.solutions.BaseSolution.initialize_region` {#ultralytics.solutions.solutions.BaseSolution.initialize\_region}

```python
def initialize_region(self) -> None
```

Initialize the counting region and line segment based on configuration settings.

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L203-L209"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def initialize_region(self) -> None:
    """Initialize the counting region and line segment based on configuration settings."""
    if self.region is None:
        self.region = [(10, 200), (540, 200), (540, 180), (10, 180)]
    self.r_s = (
        self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
    )  # region or line
```
</details>

<br>

### Method `ultralytics.solutions.solutions.BaseSolution.process` {#ultralytics.solutions.solutions.BaseSolution.process}

```python
def process(self, *args: Any, **kwargs: Any)
```

Process method should be implemented by each Solution subclass.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` | `Any` |  | *required* |
| `**kwargs` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L237-L238"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def process(self, *args: Any, **kwargs: Any):
    """Process method should be implemented by each Solution subclass."""
```
</details>

<br>

### Method `ultralytics.solutions.solutions.BaseSolution.store_tracking_history` {#ultralytics.solutions.solutions.BaseSolution.store\_tracking\_history}

```python
def store_tracking_history(self, track_id: int, box) -> None
```

Store the tracking history of an object.

This method updates the tracking history for a given object by appending the center point of its bounding box to the track line. It maintains a maximum of 30 points in the tracking history.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `track_id` | `int` | The unique identifier for the tracked object. | *required* |
| `box` | `list[float]` | The bounding box coordinates of the object in the format [x1, y1, x2, y2]. | *required* |

**Examples**

```python
>>> solution = BaseSolution()
>>> solution.store_tracking_history(1, [100, 200, 300, 400])
```

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L183-L201"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def store_tracking_history(self, track_id: int, box) -> None:
    """Store the tracking history of an object.

    This method updates the tracking history for a given object by appending the center point of its bounding box to
    the track line. It maintains a maximum of 30 points in the tracking history.

    Args:
        track_id (int): The unique identifier for the tracked object.
        box (list[float]): The bounding box coordinates of the object in the format [x1, y1, x2, y2].

    Examples:
        >>> solution = BaseSolution()
        >>> solution.store_tracking_history(1, [100, 200, 300, 400])
    """
    # Store tracking history
    self.track_line = self.track_history[track_id]
    self.track_line.append(tuple(box.mean(dim=0)) if box.numel() > 4 else (box[:4:2].mean(), box[1:4:2].mean()))
    if len(self.track_line) > 30:
        self.track_line.pop(0)
```
</details>


<br><br><hr><br>

## Class `ultralytics.solutions.solutions.SolutionAnnotator` {#ultralytics.solutions.solutions.SolutionAnnotator}

```python
def __init__(
    self,
    im: np.ndarray,
    line_width: int | None = None,
    font_size: int | None = None,
    font: str = "Arial.ttf",
    pil: bool = False,
    example: str = "abc",
)
```

**Bases:** `Annotator`

A specialized annotator class for visualizing and analyzing computer vision tasks.

This class extends the base Annotator class, providing additional methods for drawing regions, centroids, tracking trails, and visual annotations for Ultralytics Solutions. It offers comprehensive visualization capabilities for various computer vision applications including object detection, tracking, pose estimation, and analytics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `np.ndarray` | The image to be annotated. | *required* |
| `line_width` | `int, optional` | Line thickness for drawing on the image. | `None` |
| `font_size` | `int, optional` | Font size for text annotations. | `None` |
| `font` | `str` | Path to the font file. | `"Arial.ttf"` |
| `pil` | `bool` | Indicates whether to use PIL for rendering text. | `False` |
| `example` | `str` | Example text used to detect non-ASCII labels for PIL rendering. | `"abc"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `im` | `np.ndarray` | The image being annotated. |
| `line_width` | `int` | Thickness of lines used in annotations. |
| `font_size` | `int` | Size of the font used for text annotations. |
| `font` | `str` | Path to the font file used for text rendering. |
| `pil` | `bool` | Whether to use PIL for text rendering. |
| `example` | `str` | Example text used to detect non-ASCII labels for PIL rendering. |

**Methods**

| Name | Description |
| --- | --- |
| [`_estimate_pose_angle_cached`](#ultralytics.solutions.solutions.SolutionAnnotator._estimate_pose_angle_cached) | Calculate the angle between three points for workout monitoring (cached). |
| [`_point_xy`](#ultralytics.solutions.solutions.SolutionAnnotator._point_xy) | Convert a keypoint-like object to an (x, y) tuple of floats. |
| [`adaptive_label`](#ultralytics.solutions.solutions.SolutionAnnotator.adaptive_label) | Draw a label with a background rectangle or circle centered within a given bounding box. |
| [`display_analytics`](#ultralytics.solutions.solutions.SolutionAnnotator.display_analytics) | Display overall statistics for Solutions (e.g., parking management and object counting). |
| [`display_objects_labels`](#ultralytics.solutions.solutions.SolutionAnnotator.display_objects_labels) | Display the bounding boxes labels in parking management app. |
| [`draw_region`](#ultralytics.solutions.solutions.SolutionAnnotator.draw_region) | Draw a region or line on the image. |
| [`draw_specific_kpts`](#ultralytics.solutions.solutions.SolutionAnnotator.draw_specific_kpts) | Draw specific keypoints for gym steps counting. |
| [`estimate_pose_angle`](#ultralytics.solutions.solutions.SolutionAnnotator.estimate_pose_angle) | Calculate the angle between three points for workout monitoring. |
| [`plot_angle_and_count_and_stage`](#ultralytics.solutions.solutions.SolutionAnnotator.plot_angle_and_count_and_stage) | Plot the pose angle, count value, and step stage for workout monitoring. |
| [`plot_distance_and_line`](#ultralytics.solutions.solutions.SolutionAnnotator.plot_distance_and_line) | Plot the distance and line between two centroids on the frame. |
| [`plot_workout_information`](#ultralytics.solutions.solutions.SolutionAnnotator.plot_workout_information) | Draw workout text with a background on the image. |
| [`queue_counts_display`](#ultralytics.solutions.solutions.SolutionAnnotator.queue_counts_display) | Display queue counts on an image centered at the points with customizable font size and colors. |
| [`sweep_annotator`](#ultralytics.solutions.solutions.SolutionAnnotator.sweep_annotator) | Draw a sweep annotation line and an optional label. |
| [`visioneye`](#ultralytics.solutions.solutions.SolutionAnnotator.visioneye) | Perform pinpoint human-vision eye mapping and plotting. |

**Examples**

```python
>>> annotator = SolutionAnnotator(image)
>>> annotator.draw_region([(0, 0), (100, 100)], color=(0, 255, 0), thickness=5)
>>> annotator.display_analytics(
...     image, text={"Available Spots": 5}, txt_color=(0, 0, 0), bg_color=(255, 255, 255), margin=10
... )
```

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L261-L760"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SolutionAnnotator(Annotator):
    """A specialized annotator class for visualizing and analyzing computer vision tasks.

    This class extends the base Annotator class, providing additional methods for drawing regions, centroids, tracking
    trails, and visual annotations for Ultralytics Solutions. It offers comprehensive visualization capabilities for
    various computer vision applications including object detection, tracking, pose estimation, and analytics.

    Attributes:
        im (np.ndarray): The image being annotated.
        line_width (int): Thickness of lines used in annotations.
        font_size (int): Size of the font used for text annotations.
        font (str): Path to the font file used for text rendering.
        pil (bool): Whether to use PIL for text rendering.
        example (str): Example text used to detect non-ASCII labels for PIL rendering.

    Methods:
        draw_region: Draw a region using specified points, colors, and thickness.
        queue_counts_display: Display queue counts in the specified region.
        display_analytics: Display overall statistics for parking lot management.
        estimate_pose_angle: Calculate the angle between three points in an object pose.
        draw_specific_kpts: Draw specific keypoints on the image.
        plot_workout_information: Draw a labeled text box on the image.
        plot_angle_and_count_and_stage: Visualize angle, step count, and stage for workout monitoring.
        plot_distance_and_line: Display the distance between centroids and connect them with a line.
        display_objects_labels: Annotate bounding boxes with object class labels.
        sweep_annotator: Visualize a vertical sweep line and optional label.
        visioneye: Map and connect object centroids to a visual "eye" point.
        adaptive_label: Draw a circular or rectangle background shape label in center of a bounding box.

    Examples:
        >>> annotator = SolutionAnnotator(image)
        >>> annotator.draw_region([(0, 0), (100, 100)], color=(0, 255, 0), thickness=5)
        >>> annotator.display_analytics(
        ...     image, text={"Available Spots": 5}, txt_color=(0, 0, 0), bg_color=(255, 255, 255), margin=10
        ... )
    """

    def __init__(
        self,
        im: np.ndarray,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """Initialize the SolutionAnnotator class with an image for annotation.

        Args:
            im (np.ndarray): The image to be annotated.
            line_width (int, optional): Line thickness for drawing on the image.
            font_size (int, optional): Font size for text annotations.
            font (str): Path to the font file.
            pil (bool): Indicates whether to use PIL for rendering text.
            example (str): Example text used to detect non-ASCII labels for PIL rendering.
        """
        super().__init__(im, line_width, font_size, font, pil, example)
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator._estimate_pose_angle_cached` {#ultralytics.solutions.solutions.SolutionAnnotator.\_estimate\_pose\_angle\_cached}

```python
def _estimate_pose_angle_cached(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float
```

Calculate the angle between three points for workout monitoring (cached).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `a` | `tuple[float, float]` |  | *required* |
| `b` | `tuple[float, float]` |  | *required* |
| `c` | `tuple[float, float]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L433-L437"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
@lru_cache(maxsize=256)
def _estimate_pose_angle_cached(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Calculate the angle between three points for workout monitoring (cached)."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    return angle if angle <= 180.0 else (360 - angle)
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator._point_xy` {#ultralytics.solutions.solutions.SolutionAnnotator.\_point\_xy}

```python
def _point_xy(point: Any) -> tuple[float, float]
```

Convert a keypoint-like object to an (x, y) tuple of floats.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `point` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L419-L429"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _point_xy(point: Any) -> tuple[float, float]:
    """Convert a keypoint-like object to an (x, y) tuple of floats."""
    if hasattr(point, "detach"):  # torch.Tensor
        point = point.detach()
    if hasattr(point, "cpu"):  # torch.Tensor
        point = point.cpu()
    if hasattr(point, "numpy"):  # torch.Tensor
        point = point.numpy()
    if hasattr(point, "tolist"):  # numpy / torch
        point = point.tolist()
    return float(point[0]), float(point[1])
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.adaptive_label` {#ultralytics.solutions.solutions.SolutionAnnotator.adaptive\_label}

```python
def adaptive_label(
    self,
    box: tuple[float, float, float, float],
    label: str = "",
    color: tuple[int, int, int] = (128, 128, 128),
    txt_color: tuple[int, int, int] = (255, 255, 255),
    shape: str = "rect",
    margin: int = 5,
)
```

Draw a label with a background rectangle or circle centered within a given bounding box.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `box` | `tuple[float, float, float, float]` | The bounding box coordinates (x1, y1, x2, y2). | *required* |
| `label` | `str` | The text label to be displayed. | `""` |
| `color` | `tuple[int, int, int]` | The background color of the rectangle (B, G, R). | `(128, 128, 128)` |
| `txt_color` | `tuple[int, int, int]` | The color of the text (B, G, R). | `(255, 255, 255)` |
| `shape` | `str` | Label shape. Options: "circle" or "rect". | `"rect"` |
| `margin` | `int` | The margin between the text and the rectangle border. | `5` |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L706-L760"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def adaptive_label(
    self,
    box: tuple[float, float, float, float],
    label: str = "",
    color: tuple[int, int, int] = (128, 128, 128),
    txt_color: tuple[int, int, int] = (255, 255, 255),
    shape: str = "rect",
    margin: int = 5,
):
    """Draw a label with a background rectangle or circle centered within a given bounding box.

    Args:
        box (tuple[float, float, float, float]): The bounding box coordinates (x1, y1, x2, y2).
        label (str): The text label to be displayed.
        color (tuple[int, int, int]): The background color of the rectangle (B, G, R).
        txt_color (tuple[int, int, int]): The color of the text (B, G, R).
        shape (str): Label shape. Options: "circle" or "rect".
        margin (int): The margin between the text and the rectangle border.
    """
    if shape == "circle" and len(label) > 3:
        LOGGER.warning(f"Length of label is {len(label)}, only first 3 letters will be used for circle annotation.")
        label = label[:3]

    x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)  # Bounding-box center
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]  # Get size of the text
    text_x, text_y = x_center - text_size[0] // 2, y_center + text_size[1] // 2  # Calculate top-left corner of text

    if shape == "circle":
        cv2.circle(
            self.im,
            (x_center, y_center),
            int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin,  # Calculate the radius
            color,
            -1,
        )
    else:
        cv2.rectangle(
            self.im,
            (text_x - margin, text_y - text_size[1] - margin),  # Calculate coordinates of the rectangle
            (text_x + text_size[0] + margin, text_y + margin),  # Calculate coordinates of the rectangle
            color,
            -1,
        )

    # Draw the text on top of the rectangle
    cv2.putText(
        self.im,
        label,
        (text_x, text_y),  # Calculate top-left corner of the text
        cv2.FONT_HERSHEY_SIMPLEX,
        self.sf - 0.15,
        self.get_txt_color(color, txt_color),
        self.tf,
        lineType=cv2.LINE_AA,
    )
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.display_analytics` {#ultralytics.solutions.solutions.SolutionAnnotator.display\_analytics}

```python
def display_analytics(
    self,
    im0: np.ndarray,
    text: dict[str, Any],
    txt_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    margin: int,
)
```

Display overall statistics for Solutions (e.g., parking management and object counting).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im0` | `np.ndarray` | Inference image. | *required* |
| `text` | `dict[str, Any]` | Labels dictionary. | *required* |
| `txt_color` | `tuple[int, int, int]` | Text color (BGR, OpenCV format). | *required* |
| `bg_color` | `tuple[int, int, int]` | Background color (BGR, OpenCV format). | *required* |
| `margin` | `int` | Gap between text and rectangle for better display. | *required* |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L383-L416"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def display_analytics(
    self,
    im0: np.ndarray,
    text: dict[str, Any],
    txt_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    margin: int,
):
    """Display overall statistics for Solutions (e.g., parking management and object counting).

    Args:
        im0 (np.ndarray): Inference image.
        text (dict[str, Any]): Labels dictionary.
        txt_color (tuple[int, int, int]): Text color (BGR, OpenCV format).
        bg_color (tuple[int, int, int]): Background color (BGR, OpenCV format).
        margin (int): Gap between text and rectangle for better display.
    """
    horizontal_gap = int(im0.shape[1] * 0.02)
    vertical_gap = int(im0.shape[0] * 0.01)
    text_y_offset = 0
    for label, value in text.items():
        txt = f"{label}: {value}"
        text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
        if text_size[0] < 5 or text_size[1] < 5:
            text_size = (5, 5)
        text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
        text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
        rect_x1 = text_x - margin * 2
        rect_y1 = text_y - text_size[1] - margin * 2
        rect_x2 = text_x + text_size[0] + margin * 2
        rect_y2 = text_y + margin * 2
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
        text_y_offset = rect_y2
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.display_objects_labels` {#ultralytics.solutions.solutions.SolutionAnnotator.display\_objects\_labels}

```python
def display_objects_labels(
    self,
    im0: np.ndarray,
    text: str,
    txt_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    x_center: float,
    y_center: float,
    margin: int,
)
```

Display the bounding boxes labels in parking management app.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im0` | `np.ndarray` | Inference image. | *required* |
| `text` | `str` | Object/class name. | *required* |
| `txt_color` | `tuple[int, int, int]` | Display color for text foreground. | *required* |
| `bg_color` | `tuple[int, int, int]` | Display color for text background. | *required* |
| `x_center` | `float` | The x position center point for bounding box. | *required* |
| `y_center` | `float` | The y position center point for bounding box. | *required* |
| `margin` | `int` | The gap between text and rectangle for better display. | *required* |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L598-L644"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def display_objects_labels(
    self,
    im0: np.ndarray,
    text: str,
    txt_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    x_center: float,
    y_center: float,
    margin: int,
):
    """Display the bounding boxes labels in parking management app.

    Args:
        im0 (np.ndarray): Inference image.
        text (str): Object/class name.
        txt_color (tuple[int, int, int]): Display color for text foreground.
        bg_color (tuple[int, int, int]): Display color for text background.
        x_center (float): The x position center point for bounding box.
        y_center (float): The y position center point for bounding box.
        margin (int): The gap between text and rectangle for better display.
    """
    text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
    text_x = x_center - text_size[0] // 2
    text_y = y_center + text_size[1] // 2

    rect_x1 = text_x - margin
    rect_y1 = text_y - text_size[1] - margin
    rect_x2 = text_x + text_size[0] + margin
    rect_y2 = text_y + margin
    cv2.rectangle(
        im0,
        (int(rect_x1), int(rect_y1)),
        (int(rect_x2), int(rect_y2)),
        tuple(map(int, bg_color)),  # Ensure color values are int
        -1,
    )

    cv2.putText(
        im0,
        text,
        (int(text_x), int(text_y)),
        0,
        self.sf,
        tuple(map(int, txt_color)),  # Ensure color values are int
        self.tf,
        lineType=cv2.LINE_AA,
    )
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.draw_region` {#ultralytics.solutions.solutions.SolutionAnnotator.draw\_region}

```python
def draw_region(
    self,
    reg_pts: list[tuple[int, int]] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 5,
)
```

Draw a region or line on the image.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `reg_pts` | `list[tuple[int, int]], optional` | Region points (for line 2 points, for region 4+ points). | `None` |
| `color` | `tuple[int, int, int]` | BGR color value for the region (OpenCV format). | `(0, 255, 0)` |
| `thickness` | `int` | Line thickness for drawing the region. | `5` |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L319-L336"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def draw_region(
    self,
    reg_pts: list[tuple[int, int]] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 5,
):
    """Draw a region or line on the image.

    Args:
        reg_pts (list[tuple[int, int]], optional): Region points (for line 2 points, for region 4+ points).
        color (tuple[int, int, int]): BGR color value for the region (OpenCV format).
        thickness (int): Line thickness for drawing the region.
    """
    cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

    # Draw small circles at the corner points
    for point in reg_pts:
        cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 fills the circle
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.draw_specific_kpts` {#ultralytics.solutions.solutions.SolutionAnnotator.draw\_specific\_kpts}

```python
def draw_specific_kpts(
    self,
    keypoints: list[list[float]],
    indices: list[int] | None = None,
    radius: int = 2,
    conf_thresh: float = 0.25,
) -> np.ndarray
```

Draw specific keypoints for gym steps counting.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `keypoints` | `list[list[float]]` | Keypoints data to be plotted, each in format [x, y, confidence]. | *required* |
| `indices` | `list[int], optional` | Keypoint indices to be plotted. | `None` |
| `radius` | `int` | Keypoint radius. | `2` |
| `conf_thresh` | `float` | Confidence threshold for keypoints. | `0.25` |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | Image with drawn keypoints. |

!!! note "Notes"

    Keypoint format: [x, y] or [x, y, confidence].
    Modifies self.im in-place.

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L458-L491"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def draw_specific_kpts(
    self,
    keypoints: list[list[float]],
    indices: list[int] | None = None,
    radius: int = 2,
    conf_thresh: float = 0.25,
) -> np.ndarray:
    """Draw specific keypoints for gym steps counting.

    Args:
        keypoints (list[list[float]]): Keypoints data to be plotted, each in format [x, y, confidence].
        indices (list[int], optional): Keypoint indices to be plotted.
        radius (int): Keypoint radius.
        conf_thresh (float): Confidence threshold for keypoints.

    Returns:
        (np.ndarray): Image with drawn keypoints.

    Notes:
        Keypoint format: [x, y] or [x, y, confidence].
        Modifies self.im in-place.
    """
    indices = indices or [2, 5, 7]
    points = [(int(k[0]), int(k[1])) for i, k in enumerate(keypoints) if i in indices and k[2] >= conf_thresh]

    # Draw lines between consecutive points
    for start, end in zip(points[:-1], points[1:]):
        cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    # Draw circles for keypoints
    for pt in points:
        cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    return self.im
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.estimate_pose_angle` {#ultralytics.solutions.solutions.SolutionAnnotator.estimate\_pose\_angle}

```python
def estimate_pose_angle(a: Any, b: Any, c: Any) -> float
```

Calculate the angle between three points for workout monitoring.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `a` | `Any` | The coordinates of the first point (e.g. list/tuple/NumPy array/torch tensor). | *required* |
| `b` | `Any` | The coordinates of the second point (vertex). | *required* |
| `c` | `Any` | The coordinates of the third point. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `float` | The angle in degrees between the three points. |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L440-L456"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def estimate_pose_angle(a: Any, b: Any, c: Any) -> float:
    """Calculate the angle between three points for workout monitoring.

    Args:
        a (Any): The coordinates of the first point (e.g. list/tuple/NumPy array/torch tensor).
        b (Any): The coordinates of the second point (vertex).
        c (Any): The coordinates of the third point.

    Returns:
        (float): The angle in degrees between the three points.
    """
    a_xy, b_xy, c_xy = (
        SolutionAnnotator._point_xy(a),
        SolutionAnnotator._point_xy(b),
        SolutionAnnotator._point_xy(c),
    )
    return SolutionAnnotator._estimate_pose_angle_cached(a_xy, b_xy, c_xy)
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.plot_angle_and_count_and_stage` {#ultralytics.solutions.solutions.SolutionAnnotator.plot\_angle\_and\_count\_and\_stage}

```python
def plot_angle_and_count_and_stage(
    self,
    angle_text: str,
    count_text: str,
    stage_text: str,
    center_kpt: list[int],
    color: tuple[int, int, int] = (104, 31, 17),
    txt_color: tuple[int, int, int] = (255, 255, 255),
)
```

Plot the pose angle, count value, and step stage for workout monitoring.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `angle_text` | `str` | Angle value for workout monitoring. | *required* |
| `count_text` | `str` | Counts value for workout monitoring. | *required* |
| `stage_text` | `str` | Stage decision for workout monitoring. | *required* |
| `center_kpt` | `list[int]` | Centroid pose index for workout monitoring. | *required* |
| `color` | `tuple[int, int, int]` | Text background color. | `(104, 31, 17)` |
| `txt_color` | `tuple[int, int, int]` | Text foreground color. | `(255, 255, 255)` |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L526-L557"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_angle_and_count_and_stage(
    self,
    angle_text: str,
    count_text: str,
    stage_text: str,
    center_kpt: list[int],
    color: tuple[int, int, int] = (104, 31, 17),
    txt_color: tuple[int, int, int] = (255, 255, 255),
):
    """Plot the pose angle, count value, and step stage for workout monitoring.

    Args:
        angle_text (str): Angle value for workout monitoring.
        count_text (str): Counts value for workout monitoring.
        stage_text (str): Stage decision for workout monitoring.
        center_kpt (list[int]): Centroid pose index for workout monitoring.
        color (tuple[int, int, int]): Text background color.
        txt_color (tuple[int, int, int]): Text foreground color.
    """
    # Format text
    angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"

    # Draw angle, count and stage text
    angle_height = self.plot_workout_information(
        angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color
    )
    count_height = self.plot_workout_information(
        count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color
    )
    self.plot_workout_information(
        stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color
    )
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.plot_distance_and_line` {#ultralytics.solutions.solutions.SolutionAnnotator.plot\_distance\_and\_line}

```python
def plot_distance_and_line(
    self,
    pixels_distance: float,
    centroids: list[tuple[int, int]],
    line_color: tuple[int, int, int] = (104, 31, 17),
    centroid_color: tuple[int, int, int] = (255, 0, 255),
)
```

Plot the distance and line between two centroids on the frame.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pixels_distance` | `float` | Pixel distance between two bounding-box centroids. | *required* |
| `centroids` | `list[tuple[int, int]]` | Bounding box centroids data. | *required* |
| `line_color` | `tuple[int, int, int]` | Distance line color. | `(104, 31, 17)` |
| `centroid_color` | `tuple[int, int, int]` | Bounding box centroid color. | `(255, 0, 255)` |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L559-L596"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_distance_and_line(
    self,
    pixels_distance: float,
    centroids: list[tuple[int, int]],
    line_color: tuple[int, int, int] = (104, 31, 17),
    centroid_color: tuple[int, int, int] = (255, 0, 255),
):
    """Plot the distance and line between two centroids on the frame.

    Args:
        pixels_distance (float): Pixel distance between two bounding-box centroids.
        centroids (list[tuple[int, int]]): Bounding box centroids data.
        line_color (tuple[int, int, int]): Distance line color.
        centroid_color (tuple[int, int, int]): Bounding box centroid color.
    """
    # Get the text size
    text = f"Pixels Distance: {pixels_distance:.2f}"
    (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)

    # Define corners with 10-pixel margin and draw rectangle
    cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)

    # Calculate the position for the text with a 10-pixel margin and draw text
    text_position = (25, 25 + text_height_m + 10)
    cv2.putText(
        self.im,
        text,
        text_position,
        0,
        self.sf,
        (255, 255, 255),
        self.tf,
        cv2.LINE_AA,
    )

    cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
    cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
    cv2.circle(self.im, centroids[1], 6, centroid_color, -1)
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.plot_workout_information` {#ultralytics.solutions.solutions.SolutionAnnotator.plot\_workout\_information}

```python
def plot_workout_information(
    self,
    display_text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (104, 31, 17),
    txt_color: tuple[int, int, int] = (255, 255, 255),
) -> int
```

Draw workout text with a background on the image.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `display_text` | `str` | The text to be displayed. | *required* |
| `position` | `tuple[int, int]` | Coordinates (x, y) on the image where the text will be placed. | *required* |
| `color` | `tuple[int, int, int]` | Text background color. | `(104, 31, 17)` |
| `txt_color` | `tuple[int, int, int]` | Text foreground color. | `(255, 255, 255)` |

**Returns**

| Type | Description |
| --- | --- |
| `int` | The height of the text. |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L493-L524"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def plot_workout_information(
    self,
    display_text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (104, 31, 17),
    txt_color: tuple[int, int, int] = (255, 255, 255),
) -> int:
    """Draw workout text with a background on the image.

    Args:
        display_text (str): The text to be displayed.
        position (tuple[int, int]): Coordinates (x, y) on the image where the text will be placed.
        color (tuple[int, int, int]): Text background color.
        txt_color (tuple[int, int, int]): Text foreground color.

    Returns:
        (int): The height of the text.
    """
    (text_width, text_height), _ = cv2.getTextSize(display_text, 0, fontScale=self.sf, thickness=self.tf)

    # Draw background rectangle
    cv2.rectangle(
        self.im,
        (position[0], position[1] - text_height - 5),
        (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),
        color,
        -1,
    )
    # Draw text
    cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)

    return text_height
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.queue_counts_display` {#ultralytics.solutions.solutions.SolutionAnnotator.queue\_counts\_display}

```python
def queue_counts_display(
    self,
    label: str,
    points: list[tuple[int, int]] | None = None,
    region_color: tuple[int, int, int] = (255, 255, 255),
    txt_color: tuple[int, int, int] = (0, 0, 0),
)
```

Display queue counts on an image centered at the points with customizable font size and colors.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `label` | `str` | Queue counts label. | *required* |
| `points` | `list[tuple[int, int]], optional` | Region points for center point calculation to display text. | `None` |
| `region_color` | `tuple[int, int, int]` | BGR queue region color (OpenCV format). | `(255, 255, 255)` |
| `txt_color` | `tuple[int, int, int]` | BGR text color (OpenCV format). | `(0, 0, 0)` |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L338-L381"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def queue_counts_display(
    self,
    label: str,
    points: list[tuple[int, int]] | None = None,
    region_color: tuple[int, int, int] = (255, 255, 255),
    txt_color: tuple[int, int, int] = (0, 0, 0),
):
    """Display queue counts on an image centered at the points with customizable font size and colors.

    Args:
        label (str): Queue counts label.
        points (list[tuple[int, int]], optional): Region points for center point calculation to display text.
        region_color (tuple[int, int, int]): BGR queue region color (OpenCV format).
        txt_color (tuple[int, int, int]): BGR text color (OpenCV format).
    """
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    center_x = sum(x_values) // len(points)
    center_y = sum(y_values) // len(points)

    text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    rect_width = text_width + 20
    rect_height = text_height + 20
    rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
    rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
    cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2

    # Draw text
    cv2.putText(
        self.im,
        label,
        (text_x, text_y),
        0,
        fontScale=self.sf,
        color=txt_color,
        thickness=self.tf,
        lineType=cv2.LINE_AA,
    )
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.sweep_annotator` {#ultralytics.solutions.solutions.SolutionAnnotator.sweep\_annotator}

```python
def sweep_annotator(
    self,
    line_x: int = 0,
    line_y: int = 0,
    label: str | None = None,
    color: tuple[int, int, int] = (221, 0, 186),
    txt_color: tuple[int, int, int] = (255, 255, 255),
)
```

Draw a sweep annotation line and an optional label.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `line_x` | `int` | The x-coordinate of the sweep line. | `0` |
| `line_y` | `int` | The y-coordinate limit of the sweep line. | `0` |
| `label` | `str, optional` | Text label to be drawn in center of sweep line. If None, no label is drawn. | `None` |
| `color` | `tuple[int, int, int]` | BGR color for the line and label background (OpenCV format). | `(221, 0, 186)` |
| `txt_color` | `tuple[int, int, int]` | BGR color for the label text (OpenCV format). | `(255, 255, 255)` |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L646-L684"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def sweep_annotator(
    self,
    line_x: int = 0,
    line_y: int = 0,
    label: str | None = None,
    color: tuple[int, int, int] = (221, 0, 186),
    txt_color: tuple[int, int, int] = (255, 255, 255),
):
    """Draw a sweep annotation line and an optional label.

    Args:
        line_x (int): The x-coordinate of the sweep line.
        line_y (int): The y-coordinate limit of the sweep line.
        label (str, optional): Text label to be drawn in center of sweep line. If None, no label is drawn.
        color (tuple[int, int, int]): BGR color for the line and label background (OpenCV format).
        txt_color (tuple[int, int, int]): BGR color for the label text (OpenCV format).
    """
    # Draw the sweep line
    cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)

    # Draw label, if provided
    if label:
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
        cv2.rectangle(
            self.im,
            (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10),
            (line_x + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),
            color,
            -1,
        )
        cv2.putText(
            self.im,
            label,
            (line_x - text_width // 2, line_y // 2 + text_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf,
            txt_color,
            self.tf,
        )
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionAnnotator.visioneye` {#ultralytics.solutions.solutions.SolutionAnnotator.visioneye}

```python
def visioneye(
    self,
    box: list[float],
    center_point: tuple[int, int],
    color: tuple[int, int, int] = (235, 219, 11),
    pin_color: tuple[int, int, int] = (255, 0, 255),
)
```

Perform pinpoint human-vision eye mapping and plotting.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `box` | `list[float]` | Bounding box coordinates in format [x1, y1, x2, y2]. | *required* |
| `center_point` | `tuple[int, int]` | Center point for vision eye view. | *required* |
| `color` | `tuple[int, int, int]` | Object centroid and line color. | `(235, 219, 11)` |
| `pin_color` | `tuple[int, int, int]` | Visioneye point color. | `(255, 0, 255)` |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L686-L704"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def visioneye(
    self,
    box: list[float],
    center_point: tuple[int, int],
    color: tuple[int, int, int] = (235, 219, 11),
    pin_color: tuple[int, int, int] = (255, 0, 255),
):
    """Perform pinpoint human-vision eye mapping and plotting.

    Args:
        box (list[float]): Bounding box coordinates in format [x1, y1, x2, y2].
        center_point (tuple[int, int]): Center point for vision eye view.
        color (tuple[int, int, int]): Object centroid and line color.
        pin_color (tuple[int, int, int]): Visioneye point color.
    """
    center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
    cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
    cv2.line(self.im, center_point, center_bbox, color, self.tf)
```
</details>


<br><br><hr><br>

## Class `ultralytics.solutions.solutions.SolutionResults` {#ultralytics.solutions.solutions.SolutionResults}

```python
SolutionResults(self, **kwargs)
```

A class to encapsulate the results of Ultralytics Solutions.

This class is designed to store and manage various outputs generated by the solution pipeline, including counts, angles, workout stages, and other analytics data. It provides a structured way to access and manipulate results from different computer vision solutions such as object counting, pose estimation, and tracking analytics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `**kwargs` | `Any` | Optional arguments to override default attribute values. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `plot_im` | `np.ndarray` | Processed image with counts, blurred, or other effects from solutions. |
| `in_count` | `int` | The total number of "in" counts in a video stream. |
| `out_count` | `int` | The total number of "out" counts in a video stream. |
| `classwise_count` | `dict[str, int]` | A dictionary containing counts of objects categorized by class. |
| `queue_count` | `int` | The count of objects in a queue or waiting area. |
| `workout_count` | `int` | The count of workout repetitions. |
| `workout_angle` | `float` | The angle calculated during a workout exercise. |
| `workout_stage` | `str` | The current stage of the workout. |
| `pixels_distance` | `float` | The calculated distance in pixels between two points or objects. |
| `available_slots` | `int` | The number of available slots in a monitored area. |
| `filled_slots` | `int` | The number of filled slots in a monitored area. |
| `email_sent` | `bool` | A flag indicating whether an email notification was sent. |
| `total_tracks` | `int` | The total number of tracked objects. |
| `region_counts` | `dict[str, int]` | The count of objects within a specific region. |
| `speed_dict` | `dict[str, float]` | A dictionary containing speed information for tracked objects. |
| `total_crop_objects` | `int` | Total number of cropped objects using ObjectCropper class. |
| `speed` | `dict[str, float]` | Performance timing information for tracking and solution processing. |

**Methods**

| Name | Description |
| --- | --- |
| [`__str__`](#ultralytics.solutions.solutions.SolutionResults.__str__) | Return a formatted string representation of the SolutionResults object. |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L763-L828"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SolutionResults:
    """A class to encapsulate the results of Ultralytics Solutions.

    This class is designed to store and manage various outputs generated by the solution pipeline, including counts,
    angles, workout stages, and other analytics data. It provides a structured way to access and manipulate results from
    different computer vision solutions such as object counting, pose estimation, and tracking analytics.

    Attributes:
        plot_im (np.ndarray): Processed image with counts, blurred, or other effects from solutions.
        in_count (int): The total number of "in" counts in a video stream.
        out_count (int): The total number of "out" counts in a video stream.
        classwise_count (dict[str, int]): A dictionary containing counts of objects categorized by class.
        queue_count (int): The count of objects in a queue or waiting area.
        workout_count (int): The count of workout repetitions.
        workout_angle (float): The angle calculated during a workout exercise.
        workout_stage (str): The current stage of the workout.
        pixels_distance (float): The calculated distance in pixels between two points or objects.
        available_slots (int): The number of available slots in a monitored area.
        filled_slots (int): The number of filled slots in a monitored area.
        email_sent (bool): A flag indicating whether an email notification was sent.
        total_tracks (int): The total number of tracked objects.
        region_counts (dict[str, int]): The count of objects within a specific region.
        speed_dict (dict[str, float]): A dictionary containing speed information for tracked objects.
        total_crop_objects (int): Total number of cropped objects using ObjectCropper class.
        speed (dict[str, float]): Performance timing information for tracking and solution processing.
    """

    def __init__(self, **kwargs):
        """Initialize a SolutionResults object with default or user-specified values.

        Args:
            **kwargs (Any): Optional arguments to override default attribute values.
        """
        self.plot_im = None
        self.in_count = 0
        self.out_count = 0
        self.classwise_count = {}
        self.queue_count = 0
        self.workout_count = 0
        self.workout_angle = 0.0
        self.workout_stage = None
        self.pixels_distance = 0.0
        self.available_slots = 0
        self.filled_slots = 0
        self.email_sent = False
        self.total_tracks = 0
        self.region_counts = {}
        self.speed_dict = {}  # for speed estimation
        self.total_crop_objects = 0
        self.speed = {}

        # Override with user-defined values
        self.__dict__.update(kwargs)
```
</details>

<br>

### Method `ultralytics.solutions.solutions.SolutionResults.__str__` {#ultralytics.solutions.solutions.SolutionResults.\_\_str\_\_}

```python
def __str__(self) -> str
```

Return a formatted string representation of the SolutionResults object.

**Returns**

| Type | Description |
| --- | --- |
| `str` | A string representation listing non-null attributes. |

<details>
<summary>Source code in <code>ultralytics/solutions/solutions.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/solutions.py#L817-L828"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __str__(self) -> str:
    """Return a formatted string representation of the SolutionResults object.

    Returns:
        (str): A string representation listing non-null attributes.
    """
    attrs = {
        k: v
        for k, v in self.__dict__.items()
        if k != "plot_im" and v not in [None, {}, 0, 0.0, False]  # Exclude `plot_im` explicitly
    }
    return ", ".join(f"{k}={v}" for k, v in attrs.items())
```
</details>

<br><br>
