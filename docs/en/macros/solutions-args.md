{% macro param_table(params=None) %}

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |

{%- set default_params = {
    "model": ["str", "None", "Path to Ultralytics YOLO Model File."],
    "region": ["list", "[(20, 400), (1260, 400)]", "List of points defining the counting region."],
    "show_in": ["bool", "True", "Flag to control whether to display the in counts on the video stream."],
    "show_out": ["bool", "True", "Flag to control whether to display the out counts on the video stream."],
    "analytics_type": ["str", "line", "Type of graph, i.e., `line`, `bar`, `area`, or `pie`."],
    "colormap": ["int", "cv2.COLORMAP_JET", "Colormap to use for the heatmap."],
    "json_file": ["str", "None", "Path to the JSON file that contains all parking coordinates data."],
    "up_angle": ["float", "145.0", "Angle threshold for the 'up' pose."],
    "kpts": ["list[int, int, int]", "[6, 8, 10]", "List of keypoints used for monitoring workouts. These keypoints correspond to body joints or parts, such as shoulders, elbows, and wrists, for exercises like push-ups, pull-ups, squats, ab-workouts."],
    "down_angle": ["float", "90.0", "Angle threshold for the 'down' pose."],
    "blur_ratio": ["float", "0.5", "Adjusts percentage of blur intensity, with values in range `0.1 - 1.0`."],
    "crop_dir": ["str", "\"cropped-detections\"", "Directory name for storing cropped detections."],
    "records": ["int", "5", "Total detections count to trigger an email with security alarm system."],
    "vision_point": ["tuple[int, int]", "(50, 50)", "The point where vision will track objects and draw paths using VisionEye Solution."],
    "tracker": ["str", "'botsort.yaml'", "Specifies the tracking algorithm to use, e.g., `bytetrack.yaml` or `botsort.yaml`."],
    "conf": ["float", "0.3", "Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives."],
    "iou": ["float", "0.5", "Sets the Intersection over Union (IoU) threshold for filtering overlapping detections."],
    "classes": ["list", "None", "Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes."],
    "verbose": ["bool", "True", "Controls the display of tracking results, providing a visual output of tracked objects."],
    "device": ["str", "None", "Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution."],
    "show": ["bool", "False", "If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing."],
    "line_width": ["None or int", "None", "Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity."]
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
