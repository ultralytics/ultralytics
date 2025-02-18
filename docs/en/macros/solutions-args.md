{% macro param_table(params=None) %}

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |

{%- if not params %}
| `model` | `str` | `None` | Path to Ultralytics YOLO Model File. |
| `region` | `list` | `[(20, 400), (1260, 400)]` | List of points defining the counting region. |
| `show_in` | `bool` | `True` | Flag to control whether to display the in counts on the video stream. |
| `show_out` | `bool` | `True` | Flag to control whether to display the out counts on the video stream. |
| `analytics_type` | `str` | `line` | Type of graph, i.e., `line`, `bar`, `area`, or `pie`. |
| `colormap` | `int` | `cv2.COLORMAP_JET` | Colormap to use for the heatmap. |
| `json_file` | `str` | `None` | Path to the JSON file that contains all parking coordinates data. |
| `up_angle` | `float` | `145.0` | Angle threshold for the 'up' pose. |
| `kpts` | `list[int, int, int]` | `[6, 8, 10]` | List of keypoints used for monitoring workouts. These keypoints correspond to body joints or parts, such as shoulders, elbows, and wrists, for exercises like push-ups, pull-ups, squats, ab-workouts. |  
| `down_angle` | `float` | `90.0` | Angle threshold for the 'down' pose. |
| `blur_ratio` | `float` | `0.5` | Adjusts percentage of blur intensity, with values in range `0.1 - 1.0`. |
| `crop_dir` | `str` | `"cropped-detections"` | Directory name for storing cropped detections. |
| `records` | `int` | `5` | Total detections count to trigger an email with security alarm system. |
| `vision_point` | `tuple[int, int]` | `(50, 50)` | The point where vision will track objects and draw paths using VisionEye Solution. |
{%- else -%}
{%- for param in params %}
{%- if param == "model" %}
| `model` | `str` | `None` | Path to Ultralytics YOLO Model File. |
{%- endif %}
{%- if param == "region" %}
| `region` | `list` | `[(20, 400), (1260, 400)]` | List of points defining the counting region. |
{%- endif %}
{%- if param == "show_in" %}
| `show_in` | `bool` | `True` | Flag to control whether to display the in counts on the video stream. |
{%- endif %}
{%- if param == "show_out" %}
| `show_out` | `bool` | `True` | Flag to control whether to display the out counts on the video stream. |
{%- endif %}
{%- if param == "analytics_type" %}
| `analytics_type` | `str` | `line` | Type of graph, i.e., `line`, `bar`, `area`, or `pie`. |
{%- endif %}
{%- if param == "colormap" %}
| `colormap` | `int` | `cv2.COLORMAP_JET` | Colormap to use for the heatmap. |
{%- endif %}
{%- if param == "json_file" %}
| `json_file` | `str` | `None` | Path to the JSON file that contains all parking coordinates data. |
{%- endif %}
{%- if param == "up_angle" %}
| `up_angle` | `float` | `145.0` | Angle threshold for the 'up' pose. |
{%- endif %}
{%- if param == "up_angle" %}
| `kpts` | `list[int, int, int]` | `[6, 8, 10]` | List of keypoints used for monitoring workouts. These keypoints correspond to body joints or parts, such as shoulders, elbows, and wrists, for exercises like push-ups, pull-ups, squats, ab-workouts. |
{%- endif %}
{%- if param == "down_angle" %}
| `down_angle` | `float` | `90.0` | Angle threshold for the 'down' pose. |
{%- endif %}
{%- if param == "blur_ratio" %}
| `blur_ratio` | `float` | `0.5` | Adjusts percentage of blur intensity, with values in range `0.1 - 1.0`. |
{%- endif %}
{%- if param == "crop_dir" %}
| `crop_dir` | `str` | `"cropped-detections"` | Directory name for storing cropped detections. |
{%- endif %}
{%- if param == "records" %}
| `records` | `int` | `5` | Total detections count to trigger an email with security alarm system. |
{%- endif %}
{%- if param == "vision_point" %}
| `vision_point` | `tuple[int, int]` | `(50, 50)` | The point where vision will track objects and draw paths using VisionEye Solution. |
{%- endif %}
{%- endfor %}
{%- endif %}

{% endmacro %}
