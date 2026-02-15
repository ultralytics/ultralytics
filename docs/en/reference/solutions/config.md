---
description: Configure and customize Ultralytics Vision AI solutions using the SolutionConfig class. Define model paths, regions of interest, visualization options, tracking parameters, and keypoint analytics with a clean, type-safe dataclass structure for scalable development.
keywords: Ultralytics, SolutionConfig, vision AI configuration, YOLO models, Python dataclass, object detection, region of interest, tracking, keypoint analytics, computer vision, model inference, object counting, heatmaps, parking management, research
---

# Reference for `ultralytics/solutions/config.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/config.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/config.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`SolutionConfig`](#ultralytics.solutions.config.SolutionConfig)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`SolutionConfig.update`](#ultralytics.solutions.config.SolutionConfig.update)


## Class `ultralytics.solutions.config.SolutionConfig` {#ultralytics.solutions.config.SolutionConfig}

```python
SolutionConfig()
```

Manages configuration parameters for Ultralytics Vision AI solutions.

The SolutionConfig class serves as a centralized configuration container for all the Ultralytics solution modules: https://docs.ultralytics.com/solutions/#solutions. It leverages Python `dataclass` for clear, type-safe, and maintainable parameter definitions.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `source` | `str, optional` | Path to the input source (video, RTSP, etc.). Only usable with Solutions CLI. |
| `model` | `str, optional` | Path to the Ultralytics YOLO model to be used for inference. |
| `classes` | `list[int], optional` | List of class indices to filter detections. |
| `show_conf` | `bool` | Whether to show confidence scores on the visual output. |
| `show_labels` | `bool` | Whether to display class labels on visual output. |
| `region` | `list[tuple[int, int]], optional` | Polygonal region or line for object counting. |
| `colormap` | `int, optional` | OpenCV colormap constant for visual overlays (e.g., cv2.COLORMAP_JET). |
| `show_in` | `bool` | Whether to display count number for objects entering the region. |
| `show_out` | `bool` | Whether to display count number for objects leaving the region. |
| `up_angle` | `float` | Upper angle threshold used in pose-based workouts monitoring. |
| `down_angle` | `int` | Lower angle threshold used in pose-based workouts monitoring. |
| `kpts` | `list[int]` | Keypoint indices to monitor, e.g., for pose analytics. |
| `analytics_type` | `str` | Type of analytics to perform ("line", "area", "bar", "pie", etc.). |
| `figsize` | `tuple[float, float], optional` | Size of the matplotlib figure used for analytical plots (width,<br>    height). |
| `blur_ratio` | `float` | Ratio used to blur objects in the video frames (0.0 to 1.0). |
| `vision_point` | `tuple[int, int]` | Reference point for directional tracking or perspective drawing. |
| `crop_dir` | `str` | Directory path to save cropped detection images. |
| `json_file` | `str, optional` | Path to a JSON file containing data for parking areas. |
| `line_width` | `int` | Width for visual display, e.g. bounding boxes, keypoints, and counts. |
| `records` | `int` | Number of detection records to send email alerts. |
| `fps` | `float` | Frame rate (Frames Per Second) for speed estimation calculation. |
| `max_hist` | `int` | Maximum number of historical points or states stored per tracked object for speed estimation. |
| `meter_per_pixel` | `float` | Scale for real-world measurement, used in speed or distance calculations. |
| `max_speed` | `int` | Maximum speed limit (e.g., km/h or mph) used in visual alerts or constraints. |
| `show` | `bool` | Whether to display the visual output on screen. |
| `iou` | `float` | Intersection-over-Union threshold for detection filtering. |
| `conf` | `float` | Confidence threshold for keeping predictions. |
| `device` | `str, optional` | Device to run inference on (e.g., 'cpu', '0' for CUDA GPU). |
| `max_det` | `int` | Maximum number of detections allowed per video frame. |
| `half` | `bool` | Whether to use FP16 precision (requires a supported CUDA device). |
| `tracker` | `str` | Path to tracking configuration YAML file (e.g., 'botsort.yaml'). |
| `verbose` | `bool` | Enable verbose logging output for debugging or diagnostics. |
| `data` | `str` | Path to image directory used for similarity search. |

**Methods**

| Name | Description |
| --- | --- |
| [`update`](#ultralytics.solutions.config.SolutionConfig.update) | Update configuration parameters with new values provided as keyword arguments. |

**Examples**

```python
>>> from ultralytics.solutions.config import SolutionConfig
>>> cfg = SolutionConfig(model="yolo26n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
>>> cfg.update(show=False, conf=0.3)
>>> print(cfg.model)
```

<details>
<summary>Source code in <code>ultralytics/solutions/config.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/config.py#L12-L108"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@dataclass
class SolutionConfig:
```
</details>

<br>

### Method `ultralytics.solutions.config.SolutionConfig.update` {#ultralytics.solutions.config.SolutionConfig.update}

```python
def update(self, **kwargs: Any)
```

Update configuration parameters with new values provided as keyword arguments.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `**kwargs` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/solutions/config.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/config.py#L99-L108"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, **kwargs: Any):
    """Update configuration parameters with new values provided as keyword arguments."""
    for key, value in kwargs.items():
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            url = "https://docs.ultralytics.com/solutions/#solutions-arguments"
            raise ValueError(f"{key} is not a valid solution argument, see {url}")

    return self
```
</details>

<br><br>
