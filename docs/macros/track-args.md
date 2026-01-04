{% macro param_table(params=None) -%}
| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
{% set default_params = {
    "source": ["str", "None", "Specifies the source directory for images or videos. Supports file paths, URLs, and video streams."],
    "persist": ["bool", "False", "Enables persistent tracking of objects between frames, maintaining IDs across video sequences."],
    "stream": ["bool", "False", "Treats the input source as a continuous video stream for real-time processing."],
    "tracker": ["str", "'botsort.yaml'", "Specifies the tracking algorithm to use, e.g., `bytetrack.yaml` or `botsort.yaml`."],
    "conf": ["float", "0.3", "Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives."],
    "iou": ["float", "0.5", "Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections."],
    "classes": ["list", "None", "Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes."],
    "verbose": ["bool", "True", "Controls the display of tracking results, providing a visual output of tracked objects."],
    "device": ["str", "None", "Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution."],
    "show": ["bool", "False", "If `True`, displays the annotated images or videos in a window for immediate visual feedback."],
    "line_width": ["int or None", "None", "Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size."]
} %}
{% if not params %}
{% for param, details in default_params.items() %}
| `{{ param }}` | `{{ details[0] }}` | `{{ details[1] }}` | {{ details[2] }} |
{% endfor %}
{% else %}
{% for param in params %}
{% if param in default_params %}
| `{{ param }}` | `{{ default_params[param][0] }}` | `{{ default_params[param][1] }}` | {{ default_params[param][2] }} |
{% endif %}
{% endfor %}
{% endif %}
{%- endmacro -%}
