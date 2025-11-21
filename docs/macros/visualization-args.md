{% macro param_table(params=None) %}

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |

{%- set default_params = {
    "show": ["bool", "False", "If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing."],
    "save": ["bool", "False or True", "Enables saving of the annotated images or videos to files. Useful for documentation, further analysis, or sharing results. Defaults to True when using CLI & False when used in Python."],
    "save_frames": ["bool", "False", "When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis."],
    "save_txt": ["bool", "False", "Saves detection results in a text file, following the format `[class] [x_center] [y_center] [width] [height] [confidence]`. Useful for integration with other analysis tools."],
    "save_conf": ["bool", "False", "Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis."],
    "save_crop": ["bool", "False", "Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects."],
    "show_labels": ["bool", "True", "Displays labels for each detection in the visual output. Provides immediate understanding of detected objects."],
    "show_conf": ["bool", "True", "Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection."],
    "show_boxes": ["bool", "True", "Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames."],
    "line_width": ["None or int", "None", "Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity."],
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
