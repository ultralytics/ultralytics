The advanced filtering capabilities of the `Detections` class offer users a versatile and efficient way to narrow down
and refine object detections. This section outlines various filtering methods, including filtering by specific class
or a set of classes, confidence, object area, bounding box area, relative area, box dimensions, and designated zones.
Each method is demonstrated with concise code examples to provide users with a clear understanding of how to implement
the filters in their applications.

### by specific class

Allows you to select detections that belong only to one selected class.

=== "After"

    ```python
    import supervision as sv

    detections = sv.Detections(...)
    detections = detections[detections.class_id == 0]
    ```

    <div class="result" markdown>

    ![by-specific-class](https://media.roboflow.com/open-source/supervision/supervision-detection-by-specific-class.png){ align=center width="800" }

    </div>

=== "Before"

    ```python
    import supervision as sv

    detections = sv.Detections(...)
    detections = detections[detections.class_id == 0]
    ```

    <div class="result" markdown>

    ![original](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png){ align=center width="800" }

    </div>


### by set of classes

Allows you to select detections that belong only to selected set of classes.

=== "After"

    ```python
    import numpy as np
    import supervision as sv

    selected_classes = [0, 2, 3]
    detections = sv.Detections(...)
    detections = detections[np.isin(detections.class_id, selected_classes)]
    ```

    <div class="result" markdown>

    ![by-set-of-classes](https://media.roboflow.com/open-source/supervision/supervision-detection-by-set-of-classes.png){ align=center width="800" }

    </div>

=== "Before"

    ```python
    import numpy as np
    import supervision as sv

    class_id = [0, 2, 3]
    detections = sv.Detections(...)
    detections = detections[np.isin(detections.class_id, class_id)]
    ```

    <div class="result" markdown>

    ![original](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png){ align=center width="800" }

    </div>

### by confidence

Allows you to select detections with specific confidence value, for example higher than selected threshold.

=== "After"

    ```python
    import supervision as sv

    detections = sv.Detections(...)
    detections = detections[detections.confidence > 0.5]
    ```

    <div class="result" markdown>

    ![by-set-of-classes](https://media.roboflow.com/open-source/supervision/supervision-detection-by-confidence.png){ align=center width="800" }

    </div>

=== "Before"

    ```python
    import supervision as sv

    detections = sv.Detections(...)
    detections = detections[detections.confidence > 0.5]
    ```

    <div class="result" markdown>

    ![original](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png){ align=center width="800" }

    </div>

### by area

Allows you to select detections based on their size. We define the area as the number of pixels occupied by the
detection in the image. In the example below, we have sifted out the detections that are too small.

=== "After"

    ```python
    import supervision as sv

    detections = sv.Detections(...)
    detections = detections[detections.area > 1000]
    ```

    <div class="result" markdown>

    ![by-area](https://media.roboflow.com/open-source/supervision/supervision-detection-by-area.png){ align=center width="800" }

    </div>

=== "Before"

    ```python
    import supervision as sv

    detections = sv.Detections(...)
    detections = detections[detections.area > 1000]
    ```

    <div class="result" markdown>

    ![original](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png){ align=center width="800" }

    </div>

### by relative area

Allows you to select detections based on their size in relation to the size of whole image. Sometimes the concept of
detection size changes depending on the image. Detection occupying 10000 square px can be large on a 1280x720 image
but small on a 3840x2160 image. In such cases, we can filter out detections based on the percentage of the image area
occupied by them. In the example below, we remove too large detections.

=== "After"

    ```python
    import supervision as sv

    image = ...
    height, width, channels = image.shape
    image_area = height * width

    detections = sv.Detections(...)
    detections = detections[(detections.area / image_area) < 0.8]
    ```

    <div class="result" markdown>

    ![by-relative-area](https://media.roboflow.com/open-source/supervision/supervision-detection-by-relative-area.png?updatedAt=1683207183434){ align=center width="800" }

    </div>

=== "Before"

    ```python
    import supervision as sv

    image = ...
    height, width, channels = image.shape
    image_area = height * width

    detections = sv.Detections(...)
    detections = detections[(detections.area / image_area) < 0.8]
    ```

    <div class="result" markdown>

    ![original](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png){ align=center width="800" }

    </div>

### by box dimensions

Allows you to select detections based on their dimensions. The size of the bounding box, as well as its coordinates,
can be criteria for rejecting detection. Implementing such filtering requires a bit of custom code but is relatively
simple and fast.

=== "After"

    ```python
    import supervision as sv

    detections = sv.Detections(...)
    w = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    h = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    detections = detections[(w > 200) & (h > 200)]
    ```

    <div class="result" markdown>

    ![by-box-dimensions](https://media.roboflow.com/open-source/supervision/supervision-detection-by-box-dimensions.png){ align=center width="800" }

    </div>

=== "Before"

    ```python
    import supervision as sv

    detections = sv.Detections(...)
    w = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    h = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    detections = detections[(w > 200) & (h > 200)]
    ```

    <div class="result" markdown>

    ![original](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png){ align=center width="800" }

    </div>

### by `PolygonZone`

Allows you to use `Detections` in combination with `PolygonZone` to weed out bounding boxes that are in and out of the
zone. In the example below you can see how to filter out all detections located in the lower part of the image.

=== "After"

    ```python
    import supervision as sv

    zone = sv.PolygonZone(...)
    detections = sv.Detections(...)
    mask = zone.trigger(detections=detections)
    detections = detections[mask]
    ```

    <div class="result" markdown>

    ![by-polygon-zone](https://media.roboflow.com/open-source/supervision/supervision-detection-by-polygon-zone.png?updatedAt=1683211380445){ align=center width="800" }

    </div>

=== "Before"

    ```python
    import supervision as sv

    zone = sv.PolygonZone(...)
    detections = sv.Detections(...)
    mask = zone.trigger(detections=detections)
    detections = detections[mask]
    ```

    <div class="result" markdown>

    ![original](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png){ align=center width="800" }

    </div>

### by mixed conditions

`Detections`' greatest strength, however, is that you can build arbitrarily complex logical conditions by simply combining separate conditions using `&` or `|`.

=== "After"

    ```python
    import supervision as sv

    zone = sv.PolygonZone(...)
    detections = sv.Detections(...)
    mask = zone.trigger(detections=detections)
    detections = detections[(detections.confidence > 0.7) & mask]
    ```

    <div class="result" markdown>

    ![by-mixed-conditions](https://media.roboflow.com/open-source/supervision/supervision-detection-by-mixed-conditions.png){ align=center width="800" }

    </div>

=== "Before"

    ```python
    import supervision as sv

    zone = sv.PolygonZone(...)
    detections = sv.Detections(...)
    mask = zone.trigger(detections=detections)
    detections = detections[mask]
    ```

    <div class="result" markdown>

    ![original](https://media.roboflow.com/open-source/supervision/supervision-detection-original.png){ align=center width="800" }

    </div>
