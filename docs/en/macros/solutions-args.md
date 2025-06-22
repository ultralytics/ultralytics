{% macro param_table(params=None) %}

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |

{%- set default_params = {
    "model": ["str", "None", "Path to Ultralytics YOLO Model File."],
    "region": ["list", "'[(20, 400), (1260, 400)]'", "List of points defining the counting region."],
    "show_in": ["bool", "True", "Flag to control whether to display the in counts on the video stream."],
    "show_out": ["bool", "True", "Flag to control whether to display the out counts on the video stream."],
    "analytics_type": ["str", "line", "Type of graph, i.e., `line`, `bar`, `area`, or `pie`."],
    "colormap": ["int", "cv2.COLORMAP_JET", "Colormap to use for the heatmap."],
    "json_file": ["str", "None", "Path to the JSON file that contains all parking coordinates data."],
    "up_angle": ["float", "145.0", "Angle threshold for the 'up' pose."],
    "kpts": ["list[int, int, int]", "'[6, 8, 10]'", "List of keypoints used for monitoring workouts. These keypoints correspond to body joints or parts, such as shoulders, elbows, and wrists, for exercises like push-ups, pull-ups, squats, ab-workouts."],
    "down_angle": ["float", "90.0", "Angle threshold for the 'down' pose."],
    "blur_ratio": ["float", "0.5", "Adjusts percentage of blur intensity, with values in range `0.1 - 1.0`."],
    "crop_dir": ["str", "'cropped-detections'", "Directory name for storing cropped detections."],
    "records": ["int", "5", "Total detections count to trigger an email with security alarm system."],
    "vision_point": ["tuple[int, int]", "(20, 20)", "The point where vision will track objects and draw paths using VisionEye Solution."],
    "source": ["str", "None", "Path to the input source (video, RTSP, etc.). Only usable with Solutions command line interface (CLI)."],
    "figsize": ["tuple[int, int]", "(12.8, 7.2)", "Figure size for analytics charts such as heatmaps or graphs."],
    "fps": ["float", "30.0", "Frames per second used for speed calculations."],
    "max_hist": ["int", "5", "Maximum historical points to track per object for speed/direction calculations."],
    "meter_per_pixel": ["float", "0.05", "Scaling factor used for converting pixel distance to real-world units."],
    "max_speed": ["int", "120", "Maximum speed limit in visual overlays (used in alerts)."],
    "data": ["str", "'images'", "Path to image directory used for similarity search."],
} %}
{%- if not params %}
{%- for param, details in default_params.items() %}
| `{{ param }}` | `{{ details[0] }}` | `{{ details[1] }}` | {{ details[2] }} |
{%- endfor %}
{%- else %}
{%- for param in params %}
{%- if param in default_params %}
| `{{ param }}` | `{{ default_params[param][0] }}` | `{{ default_params[param][1] }}` | {{ default_params[param][2] }} |
{%- endif %}
{%- endfor %}
{%- endif %}

{% endmacro %}
