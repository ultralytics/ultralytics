=== "BoundingBox"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> bounding_box_annotator = sv.BoundingBoxAnnotator()
    >>> annotated_frame = bounding_box_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![bounding-box-annotator-example](https://media.roboflow.com/supervision-annotator-examples/bounding-box-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "BoxCorner"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> corner_annotator = sv.BoxCornerAnnotator()
    >>> annotated_frame = corner_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![box-corner-annotator-example](https://media.roboflow.com/supervision-annotator-examples/box-corner-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "BoxMask"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> box_mask_annotator = sv.BoxMaskAnnotator()
    >>> annotated_frame = box_mask_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![box-mask-annotator-example](https://media.roboflow.com/supervision-annotator-examples/box-mask-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Circle"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> circle_annotator = sv.CircleAnnotator()
    >>> annotated_frame = circle_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![circle-annotator-example](https://media.roboflow.com/supervision-annotator-examples/circle-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Dot"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> dot_annotator = sv.DotAnnotator()
    >>> annotated_frame = dot_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![circle-annotator-example](https://media.roboflow.com/supervision-annotator-examples/dot-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Ellipse"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> ellipse_annotator = sv.EllipseAnnotator()
    >>> annotated_frame = ellipse_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![ellipse-annotator-example](https://media.roboflow.com/supervision-annotator-examples/ellipse-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Halo"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> halo_annotator = sv.HaloAnnotator()
    >>> annotated_frame = halo_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![ellipse-annotator-example](https://media.roboflow.com/supervision-annotator-examples/halo-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Mask"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> mask_annotator = sv.MaskAnnotator()
    >>> annotated_frame = mask_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![mask-annotator-example](https://media.roboflow.com/supervision-annotator-examples/mask-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Label"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    >>> annotated_frame = label_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![label-annotator-example](https://media.roboflow.com/supervision-annotator-examples/label-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Blur"

    ```python
    >>> import supervision as sv

    >>> image = ...
    >>> detections = sv.Detections(...)

    >>> blur_annotator = sv.BlurAnnotator()
    >>> annotated_frame = blur_annotator.annotate(
    ...     scene=image.copy(),
    ...     detections=detections
    ... )
    ```

    <div class="result" markdown>

    ![blur-annotator-example](https://media.roboflow.com/supervision-annotator-examples/blur-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "Trace"

    ```python
    >>> import supervision as sv
    >>> from ultralytics import YOLO

    >>> model = YOLO('yolov8x.pt')

    >>> trace_annotator = sv.TraceAnnotator()

    >>> video_info = sv.VideoInfo.from_video_path(video_path='...')
    >>> frames_generator = get_video_frames_generator(source_path='...')
    >>> tracker = sv.ByteTrack()

    >>> with sv.VideoSink(target_path='...', video_info=video_info) as sink:
    ...    for frame in frames_generator:
    ...        result = model(frame)[0]
    ...        detections = sv.Detections.from_ultralytics(result)
    ...        detections = tracker.update_with_detections(detections)
    ...        annotated_frame = trace_annotator.annotate(
    ...            scene=frame.copy(),
    ...            detections=detections)
    ...        sink.write_frame(frame=annotated_frame)
    ```

    <div class="result" markdown>

    ![trace-annotator-example](https://media.roboflow.com/supervision-annotator-examples/trace-annotator-example-purple.png){ align=center width="800" }

    </div>

=== "HeatMap"

    ```python
    >>> import supervision as sv
    >>> from ultralytics import YOLO

    >>> model = YOLO('yolov8x.pt')

    >>> heat_map_annotator = sv.HeatMapAnnotator()

    >>> video_info = sv.VideoInfo.from_video_path(video_path='...')
    >>> frames_generator = get_video_frames_generator(source_path='...')

    >>> with sv.VideoSink(target_path='...', video_info=video_info) as sink:
    ...    for frame in frames_generator:
    ...        result = model(frame)[0]
    ...        detections = sv.Detections.from_ultralytics(result)
    ...        annotated_frame = heat_map_annotator.annotate(
    ...            scene=frame.copy(),
    ...            detections=detections)
    ...        sink.write_frame(frame=annotated_frame)
    ```

    <div class="result" markdown>

    ![trace-annotator-example](https://media.roboflow.com/supervision-annotator-examples/heat-map-annotator-example-purple.png){ align=center width="800" }

    </div>

## BoundingBoxAnnotator

:::supervision.annotators.core.BoundingBoxAnnotator

## BoxCornerAnnotator

:::supervision.annotators.core.BoxCornerAnnotator

## BoxMaskAnnotator

:::supervision.annotators.core.BoxMaskAnnotator

## CircleAnnotator

:::supervision.annotators.core.CircleAnnotator

## DotAnnotator

:::supervision.annotators.core.DotAnnotator

## EllipseAnnotator

:::supervision.annotators.core.EllipseAnnotator

## HaloAnnotator

:::supervision.annotators.core.HaloAnnotator

## HeatMapAnnotator

:::supervision.annotators.core.HeatMapAnnotator

## MaskAnnotator

:::supervision.annotators.core.MaskAnnotator

## LabelAnnotator

:::supervision.annotators.core.LabelAnnotator

## BlurAnnotator

:::supervision.annotators.core.BlurAnnotator

## TraceAnnotator

:::supervision.annotators.core.TraceAnnotator

## ColorLookup

:::supervision.annotators.utils.ColorLookup
