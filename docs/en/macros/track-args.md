{% macro param_table(params=None) %}
| Argument | Type | Default | Description |
|------|------|---------|-------------|
{%- if not params -%}
| `source` | `str` | `None` | Specifies the source directory for images or videos. Supports file paths and URLs. |
| `persist` | `bool` | `False` | Enables persistent tracking of objects between frames, maintaining IDs across video sequences. |
| `tracker` | `str` | `'botsort.yaml'` | Specifies the tracking algorithm to use, e.g., `bytetrack.yaml` or `botsort.yaml`. |
| `conf` | `float` | `0.3` | Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives. |
| `iou` | `float` | `0.5` | Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections. |
| `classes` | `list` | `None` | Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes. |
| `verbose` | `bool` | `True` | Controls the display of tracking results, providing a visual output of tracked objects. |
| `device` | `str` | `None` | Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution. |
{%- else -%}
{%- for param in params -%}
{%- if param == "source" %}
| `source` | `str` | `None` | Specifies the source directory for images or videos. Supports file paths and URLs. |
{%- endif %}
{%- if param == "persist" %}
| `persist` | `bool` | `False` | Enables persistent tracking of objects between frames, maintaining IDs across video sequences. |
{%- endif %}
{%- if param == "tracker" %}
| `tracker` | `str` | `'botsort.yaml'` | Specifies the tracking algorithm to use, e.g., `bytetrack.yaml` or `botsort.yaml`. |
{%- endif %}
{%- if param == "conf" %}
| `conf` | `float` | `0.3` | Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives. |
{%- endif %}
{%- if param == "iou" %}
| `iou` | `float` | `0.5` | Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections. |
{%- endif %}
{%- if param == "classes" %}
| `classes` | `list` | `None` | Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes. |
{%- endif %}
{%- if param == "verbose" %}
| `verbose` | `bool` | `True` | Controls the display of tracking results, providing a visual output of tracked objects. |
{%- endif %}
{%- if param == "device" %}
| `device` | `str` | `None` | Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution. |
{%- endif %}
{%- endfor %}
{%- endif %}
{% endmacro %}