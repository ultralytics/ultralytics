### 0.16.0 <small>October 19, 2023</small>

- Added [#422](https://github.com/roboflow/supervision/pull/422): [`sv.BoxMaskAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.BoxMaskAnnotator) allowing to annotate images and videos with mox masks.

- Added [#433](https://github.com/roboflow/supervision/pull/433): [`sv.HaloAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.HaloAnnotator) allowing to annotate images and videos with halo effect.

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

- Added [#466](https://github.com/roboflow/supervision/pull/466): [`sv.HeatMapAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.HeatMapAnnotator) allowing to annotate videos with heat maps.

- Added [#492](https://github.com/roboflow/supervision/pull/492): [`sv.DotAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.DotAnnotator) allowing to annotate images and videos with dots.

- Added [#449](https://github.com/roboflow/supervision/pull/449): [`sv.draw_image`](https://supervision.roboflow.com/draw/utils/#supervision.draw.utils.draw_image) allowing to draw an image onto a given scene with specified opacity and dimensions.

- Added [#280](https://github.com/roboflow/supervision/pull/280): [`sv.FPSMonitor`](https://supervision.roboflow.com/utils/video/#supervision.utils.video.FPSMonitor) for monitoring frames per second (FPS) to benchmark latency.

- Added [#454](https://github.com/roboflow/supervision/pull/454): 🤗 Hugging Face Annotators [space](https://huggingface.co/spaces/Roboflow/Annotators).

- Changed [#482](https://github.com/roboflow/supervision/pull/482): [`sv.LineZone.tigger`](https://supervision.roboflow.com/detection/tools/line_zone/#supervision.detection.line_counter.LineZone.trigger) now return `Tuple[np.ndarray, np.ndarray]`. The first array indicates which detections have crossed the line from outside to inside. The second array indicates which detections have crossed the line from inside to outside.

- Changed [#465](https://github.com/roboflow/supervision/pull/465): Annotator argument name from `color_map: str` to `color_lookup: ColorLookup` enum to increase type safety.

- Changed [#426](https://github.com/roboflow/supervision/pull/426): [`sv.MaskAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.MaskAnnotator) allowing 2x faster annotation.

- Fixed [#477](https://github.com/roboflow/supervision/pull/477): Poetry env definition allowing proper local installation.

- Fixed [#430](https://github.com/roboflow/supervision/pull/430):  [`sv.ByteTrack`](https://supervision.roboflow.com/trackers/#supervision.tracker.byte_tracker.core.ByteTrack) to return `np.array([], dtype=int)` when `svDetections` is empty.

!!! warning

    `sv.Detections.from_yolov8` and `sv.Classifications.from_yolov8` as those are now replaced by [`sv.Detections.from_ultralytics`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_ultralytics) and [`sv.Classifications.from_ultralytics`](https://supervision.roboflow.com/classification/core/#supervision.classification.core.Classifications.from_ultralytics).


### 0.15.0 <small>October 5, 2023</small>

- Added [#170](https://github.com/roboflow/supervision/pull/170): [`sv.BoundingBoxAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.BoundingBoxAnnotator) allowing to annotate images and videos with bounding boxes.

- Added [#170](https://github.com/roboflow/supervision/pull/170): [`sv.BoxCornerAnnotator `](https://supervision.roboflow.com/annotators/#supervision.annotators.core.BoxCornerAnnotator) allowing to annotate images and videos with just bounding box corners.

- Added [#170](https://github.com/roboflow/supervision/pull/170): [`sv.MaskAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.MaskAnnotator) allowing to annotate images and videos with segmentation masks.

- Added [#170](https://github.com/roboflow/supervision/pull/170): [`sv.EllipseAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.EllipseAnnotator) allowing to annotate images and videos with ellipses (sports game style).

- Added [#386](https://github.com/roboflow/supervision/pull/386): [`sv.CircleAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.CircleAnnotator) allowing to annotate images and videos with circles.

- Added [#354](https://github.com/roboflow/supervision/pull/354): [`sv.TraceAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.TraceAnnotator) allowing to draw path of moving objects on videos.

- Added [#405](https://github.com/roboflow/supervision/pull/405): [`sv.BlurAnnotator`](https://supervision.roboflow.com/annotators/#supervision.annotators.core.BlurAnnotator) allowing to blur objects on images and videos.

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

- Added [#354](https://github.com/roboflow/supervision/pull/354): Supervision usage [example](https://github.com/roboflow/supervision/tree/develop/examples/traffic_analysis). You can now learn how to perform traffic flow analysis with Supervision.

- Changed [#399](https://github.com/roboflow/supervision/pull/399): [`sv.Detections.from_roboflow`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_roboflow) now does not require `class_list` to be specified. The `class_id` value can be extracted directly from the [inference](https://github.com/roboflow/inference) response.

- Changed [#381](https://github.com/roboflow/supervision/pull/381): [`sv.VideoSink`](https://supervision.roboflow.com/utils/video/#videosink) now allows to customize the output codec.

- Changed [#361](https://github.com/roboflow/supervision/pull/361): [`sv.InferenceSlicer`](https://supervision.roboflow.com/detection/tools/inference_slicer/#supervision.detection.tools.inference_slicer.InferenceSlicer) can now operate in multithreading mode.

- Fixed [#348](https://github.com/roboflow/supervision/pull/348): [`sv.Detections.from_deepsparse`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_deepsparse) to allow processing empty [deepsparse](https://github.com/neuralmagic/deepsparse) result object.

### 0.14.0 <small>August 31, 2023</small>

- Added [#282](https://github.com/roboflow/supervision/pull/282): support for SAHI inference technique with [`sv.InferenceSlicer`](https://supervision.roboflow.com/detection/tools/inference_slicer).

```python
>>> import cv2
>>> import supervision as sv
>>> from ultralytics import YOLO

>>> image = cv2.imread(SOURCE_IMAGE_PATH)
>>> model = YOLO(...)

>>> def callback(image_slice: np.ndarray) -> sv.Detections:
...     result = model(image_slice)[0]
...     return sv.Detections.from_ultralytics(result)

>>> slicer = sv.InferenceSlicer(callback = callback)

>>> detections = slicer(image)
```

- Added [#297](https://github.com/roboflow/supervision/pull/297): [`Detections.from_deepsparse`](https://roboflow.github.io/supervision/detection/core/#supervision.detection.core.Detections.from_deepsparse) to enable seamless integration with [DeepSparse](https://github.com/neuralmagic/deepsparse) framework.

- Added [#281](https://github.com/roboflow/supervision/pull/281): [`sv.Classifications.from_ultralytics`](https://supervision.roboflow.com/classification/core/#supervision.classification.core.Classifications.from_ultralytics) to enable seamless integration with [Ultralytics](https://github.com/ultralytics/ultralytics) framework. This will enable you to use supervision with all [models](https://docs.ultralytics.com/models/) that Ultralytics supports.

!!! warning

    [sv.Detections.from_yolov8](https://roboflow.github.io/supervision/detection/core/#supervision.detection.core.Detections.from_yolov8) and [sv.Classifications.from_yolov8](https://supervision.roboflow.com/classification/core/#supervision.classification.core.Classifications.from_yolov8) are now deprecated and will be removed with supervision-0.16.0 release.

- Added [#341](https://github.com/roboflow/supervision/pull/341): First supervision usage example script showing how to detect and track objects on video using YOLOv8 + Supervision.

- Changed [#296](https://github.com/roboflow/supervision/pull/296): [`sv.ClassificationDataset`](https://supervision.roboflow.com/dataset/core/#supervision.dataset.core.ClassificationDataset) and [`sv.DetectionDataset`](https://supervision.roboflow.com/dataset/core/#supervision.dataset.core.DetectionDataset) now use image path (not image name) as dataset keys.

- Fixed [#300](https://github.com/roboflow/supervision/pull/300): [`Detections.from_roboflow`](https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections.from_roboflow) to filter out polygons with less than 3 points.

### 0.13.0 <small>August 8, 2023</small>

- Added [#236](https://github.com/roboflow/supervision/pull/236): support for mean average precision (mAP) for object detection models with [`sv.MeanAveragePrecision`](https://roboflow.github.io/supervision/metrics/detection/#meanaverageprecision).

```python
>>> import supervision as sv
>>> from ultralytics import YOLO

>>> dataset = sv.DetectionDataset.from_yolo(...)

>>> model = YOLO(...)
>>> def callback(image: np.ndarray) -> sv.Detections:
...     result = model(image)[0]
...     return sv.Detections.from_yolov8(result)

>>> mean_average_precision = sv.MeanAveragePrecision.benchmark(
...     dataset = dataset,
...     callback = callback
... )

>>> mean_average_precision.map50_95
0.433
```

- Added [#256](https://github.com/roboflow/supervision/pull/256): support for ByteTrack for object tracking with [`sv.ByteTrack`](https://roboflow.github.io/supervision/tracker/core/#bytetrack).

- Added [#222](https://github.com/roboflow/supervision/pull/222): [`sv.Detections.from_ultralytics`](https://roboflow.github.io/supervision/detection/core/#supervision.detection.core.Detections.from_ultralytics) to enable seamless integration with [Ultralytics](https://github.com/ultralytics/ultralytics) framework. This will enable you to use `supervision` with all [models](https://docs.ultralytics.com/models/) that Ultralytics supports.

!!! warning

    [`sv.Detections.from_yolov8`](https://roboflow.github.io/supervision/detection/core/#supervision.detection.core.Detections.from_yolov8) is now deprecated and will be removed with `supervision-0.15.0` release.

- Added [#191](https://github.com/roboflow/supervision/pull/191): [`sv.Detections.from_paddledet`](https://roboflow.github.io/supervision/detection/core/#supervision.detection.core.Detections.from_paddledet) to enable seamless integration with [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) framework.

- Added [#245](https://github.com/roboflow/supervision/pull/245): support for loading PASCAL VOC segmentation datasets with [`sv.DetectionDataset.`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.DetectionDataset.from_pascal_voc).

### 0.12.0 <small>July 24, 2023</small>

!!! warning

    With the `supervision-0.12.0` release, we are terminating official support for Python 3.7.

- Added [#177](https://github.com/roboflow/supervision/pull/177): initial support for object detection model benchmarking with [`sv.ConfusionMatrix`](https://roboflow.github.io/supervision/metrics/detection/#confusionmatrix).

```python
>>> import supervision as sv
>>> from ultralytics import YOLO

>>> dataset = sv.DetectionDataset.from_yolo(...)

>>> model = YOLO(...)
>>> def callback(image: np.ndarray) -> sv.Detections:
...     result = model(image)[0]
...     return sv.Detections.from_yolov8(result)

>>> confusion_matrix = sv.ConfusionMatrix.benchmark(
...     dataset = dataset,
...     callback = callback
... )

>>> confusion_matrix.matrix
array([
    [0., 0., 0., 0.],
    [0., 1., 0., 1.],
    [0., 1., 1., 0.],
    [1., 1., 0., 0.]
])
```

- Added [#173](https://github.com/roboflow/supervision/pull/173): [`Detections.from_mmdetection`](https://roboflow.github.io/supervision/detection/core/#supervision.detection.core.Detections.from_mmdetection) to enable seamless integration with [MMDetection](https://github.com/open-mmlab/mmdetection) framework.

- Added [#130](https://github.com/roboflow/supervision/issues/130): ability to [install](https://roboflow.github.io/supervision/) package in `headless` or `desktop` mode.

- Changed [#180](https://github.com/roboflow/supervision/pull/180): packing method from `setup.py` to `pyproject.toml`.

- Fixed [#188](https://github.com/roboflow/supervision/issues/188): [`sv.DetectionDataset.from_cooc`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.DetectionDataset.from_coco) can't be loaded when there are images without annotations.

- Fixed [#226](https://github.com/roboflow/supervision/issues/226): [`sv.DetectionDataset.from_yolo`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.DetectionDataset.from_yolo) can't load background instances.

### 0.11.1 <small>June 29, 2023</small>

- Fix [#165](https://github.com/roboflow/supervision/pull/165): [`as_folder_structure`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.ClassificationDataset.as_folder_structure) fails to save [`sv.ClassificationDataset`](https://roboflow.github.io/supervision/dataset/core/#classificationdataset) when it is result of inference.

### 0.11.0 <small>June 28, 2023</small>

- Added [#150](https://github.com/roboflow/supervision/pull/150): ability to load and save [`sv.DetectionDataset`](https://roboflow.github.io/supervision/dataset/core/#detectiondataset) in COCO format using [`as_coco`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.DetectionDataset.as_coco) and [`from_coco`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.DetectionDataset.from_coco) methods.

```python
>>> import supervision as sv

>>> ds = sv.DetectionDataset.from_coco(
...     images_directory_path='...',
...     annotations_path='...'
... )

>>> ds.as_coco(
...     images_directory_path='...',
...     annotations_path='...'
... )
```

- Added [#158](https://github.com/roboflow/supervision/pull/158): ability to marge multiple [`sv.DetectionDataset`](https://roboflow.github.io/supervision/dataset/core/#detectiondataset) together using [`merge`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.DetectionDataset.merge) method.

```python
>>> import supervision as sv

>>> ds_1 = sv.DetectionDataset(...)
>>> len(ds_1)
100
>>> ds_1.classes
['dog', 'person']

>>> ds_2 = sv.DetectionDataset(...)
>>> len(ds_2)
200
>>> ds_2.classes
['cat']

>>> ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
>>> len(ds_merged)
300
>>> ds_merged.classes
['cat', 'dog', 'person']
```

- Added [#162](https://github.com/roboflow/supervision/pull/162): additional `start` and `end` arguments to [`sv.get_video_frames_generator`](https://roboflow.github.io/supervision/utils/video/#get_video_frames_generator) allowing to generate frames only for a selected part of the video.

- Fix [#157](https://github.com/roboflow/supervision/pull/157): incorrect loading of YOLO dataset class names from `data.yaml`.

### 0.10.0 <small>June 14, 2023</small>

- Added [#125](https://github.com/roboflow/supervision/pull/125): ability to load and save [`sv.ClassificationDataset`](https://roboflow.github.io/supervision/dataset/core/#classificationdataset) in a folder structure format.

```python
>>> import supervision as sv

>>> cs = sv.ClassificationDataset.from_folder_structure(
...     root_directory_path='...'
... )

>>> cs.as_folder_structure(
...     root_directory_path='...'
... )
```

- Added [#125](https://github.com/roboflow/supervision/pull/125): support for [`sv.ClassificationDataset.split`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.ClassificationDataset.split) allowing to divide `sv.ClassificationDataset` into two parts.

- Added [#110](https://github.com/roboflow/supervision/pull/110): ability to extract masks from Roboflow API results using [`sv.Detections.from_roboflow`](https://roboflow.github.io/supervision/detection/core/#supervision.detection.core.Detections.from_roboflow).

- Added [commit hash](https://github.com/roboflow/supervision/commit/d000292eb2f2342544e0947b65528082e60fb8d6): Supervision Quickstart [notebook](https://colab.research.google.com/github/roboflow/supervision/blob/main/demo.ipynb) where you can learn more about Detection, Dataset and Video APIs.

- Changed [#135](https://github.com/roboflow/supervision/pull/135): `sv.get_video_frames_generator` documentation to better describe actual behavior.

### 0.9.0 <small>June 7, 2023</small>

- Added [#118](https://github.com/roboflow/supervision/pull/118): ability to select [`sv.Detections`](https://roboflow.github.io/supervision/detection/core/#supervision.detection.core.Detections.__getitem__) by index, list of indexes or slice. Here is an example illustrating the new selection methods.

```python
>>> import supervision as sv

>>> detections = sv.Detections(...)
>>> len(detections[0])
1
>>> len(detections[[0, 1]])
2
>>> len(detections[0:2])
2
```

- Added [#101](https://github.com/roboflow/supervision/pull/101): ability to extract masks from YOLOv8 result using [`sv.Detections.from_yolov8`](https://roboflow.github.io/supervision/detection/core/#supervision.detection.core.Detections.from_yolov8). Here is an example illustrating how to extract boolean masks from the result of the YOLOv8 model inference.

- Added [#122](https://github.com/roboflow/supervision/pull/122): ability to crop image using [`sv.crop`](https://roboflow.github.io/supervision/utils/image/#crop). Here is an example showing how to get a separate crop for each detection in `sv.Detections`.

- Added [#120](https://github.com/roboflow/supervision/pull/120): ability to conveniently save multiple images into directory using [`sv.ImageSink`](https://roboflow.github.io/supervision/utils/image/#imagesink). Here is an example showing how to save every tenth video frame as a separate image.

```python
>>> import supervision as sv

>>> with sv.ImageSink(target_dir_path='target/directory/path') as sink:
...     for image in sv.get_video_frames_generator(source_path='source_video.mp4', stride=10):
...         sink.save_image(image=image)
```

- Fixed [#106](https://github.com/roboflow/supervision/issues/106): inconvenient handling of [`sv.PolygonZone`](https://roboflow.github.io/supervision/detection/tools/polygon_zone/#polygonzone) coordinates. Now `sv.PolygonZone` accepts coordinates in the form of `[[x1, y1], [x2, y2], ...]` that can be both integers and floats.

### 0.8.0 <small>May 17, 2023</small>

- Added [#100](https://github.com/roboflow/supervision/pull/100): support for dataset inheritance. The current `Dataset` got renamed to `DetectionDataset`. Now [`DetectionDataset`](https://roboflow.github.io/supervision/dataset/core/#detectiondataset) inherits from `BaseDataset`. This change was made to enforce the future consistency of APIs of different types of computer vision datasets.
- Added [#100](https://github.com/roboflow/supervision/pull/100): ability to save datasets in YOLO format using [`DetectionDataset.as_yolo`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.DetectionDataset.as_yolo).

```python
>>> import roboflow
>>> from roboflow import Roboflow
>>> import supervision as sv

>>> roboflow.login()

>>> rf = Roboflow()

>>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
>>> dataset = project.version(PROJECT_VERSION).download("yolov5")

>>> ds = sv.DetectionDataset.from_yolo(
...     images_directory_path=f"{dataset.location}/train/images",
...     annotations_directory_path=f"{dataset.location}/train/labels",
...     data_yaml_path=f"{dataset.location}/data.yaml"
... )

>>> ds.classes
['dog', 'person']
```

- Added [#102](https://github.com/roboflow/supervision/pull/103): support for [`DetectionDataset.split`](https://roboflow.github.io/supervision/dataset/core/#supervision.dataset.core.DetectionDataset.split) allowing to divide `DetectionDataset` into two parts.

```python
>>> import supervision as sv

>>> ds = sv.DetectionDataset(...)
>>> train_ds, test_ds = ds.split(split_ratio=0.7, random_state=42, shuffle=True)

>>> len(train_ds), len(test_ds)
(700, 300)
```

- Changed [#100](https://github.com/roboflow/supervision/pull/100): default value of `approximation_percentage` parameter from `0.75` to `0.0` in `DetectionDataset.as_yolo` and `DetectionDataset.as_pascal_voc`.

### 0.7.0 <small>May 11, 2023</small>

- Added [#91](https://github.com/roboflow/supervision/pull/91): `Detections.from_yolo_nas` to enable seamless integration with [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) model.
- Added [#86](https://github.com/roboflow/supervision/pull/86): ability to load datasets in YOLO format using `Dataset.from_yolo`.
- Added [#84](https://github.com/roboflow/supervision/pull/84): `Detections.merge` to merge multiple `Detections` objects together.
- Fixed [#81](https://github.com/roboflow/supervision/pull/81): `LineZoneAnnotator.annotate` does not return annotated frame.
- Changed [#44](https://github.com/roboflow/supervision/pull/44): `LineZoneAnnotator.annotate` to allow for custom text for the in and out tags.

### 0.6.0 <small>April 19, 2023</small>

- Added [#71](https://github.com/roboflow/supervision/pull/71): initial `Dataset` support and ability to save `Detections` in Pascal VOC XML format.
- Added [#71](https://github.com/roboflow/supervision/pull/71): new `mask_to_polygons`, `filter_polygons_by_area`, `polygon_to_xyxy` and `approximate_polygon` utilities.
- Added [#72](https://github.com/roboflow/supervision/pull/72): ability to load Pascal VOC XML **object detections** dataset as `Dataset`.
- Changed [#70](https://github.com/roboflow/supervision/pull/70): order of `Detections` attributes to make it consistent with order of objects in `__iter__` tuple.
- Changed [#71](https://github.com/roboflow/supervision/pull/71): `generate_2d_mask` to `polygon_to_mask`.

### 0.5.2 <small>April 13, 2023</small>

- Fixed [#63](https://github.com/roboflow/supervision/pull/63): `LineZone.trigger` function expects 4 values instead of 5.

### 0.5.1 <small>April 12, 2023</small>

- Fixed `Detections.__getitem__` method did not return mask for selected item.
- Fixed `Detections.area` crashed for mask detections.

### 0.5.0 <small>April 10, 2023</small>

- Added [#58](https://github.com/roboflow/supervision/pull/58): `Detections.mask` to enable segmentation support.
- Added [#58](https://github.com/roboflow/supervision/pull/58): `MaskAnnotator` to allow easy `Detections.mask` annotation.
- Added [#58](https://github.com/roboflow/supervision/pull/58): `Detections.from_sam` to enable native Segment Anything Model (SAM) support.
- Changed [#58](https://github.com/roboflow/supervision/pull/58): `Detections.area` behaviour to work not only with boxes but also with masks.

### 0.4.0 <small>April 5, 2023</small>

- Added [#46](https://github.com/roboflow/supervision/discussions/48): `Detections.empty` to allow easy creation of empty `Detections` objects.
- Added [#56](https://github.com/roboflow/supervision/pull/56): `Detections.from_roboflow` to allow easy creation of `Detections` objects from Roboflow API inference results.
- Added [#56](https://github.com/roboflow/supervision/pull/56): `plot_images_grid` to allow easy plotting of multiple images on single plot.
- Added [#56](https://github.com/roboflow/supervision/pull/56): initial support for Pascal VOC XML format with `detections_to_voc_xml` method.
- Changed [#56](https://github.com/roboflow/supervision/pull/56): `show_frame_in_notebook` refactored and renamed to `plot_image`.

### 0.3.2 <small>March 23, 2023</small>

- Changed [#50](https://github.com/roboflow/supervision/issues/50): Allow `Detections.class_id` to be `None`.

### 0.3.1 <small>March 6, 2023</small>

- Fixed [#41](https://github.com/roboflow/supervision/issues/41): `PolygonZone` throws an exception when the object touches the bottom edge of the image.
- Fixed [#42](https://github.com/roboflow/supervision/issues/42): `Detections.wth_nms` method throws an exception when `Detections` is empty.
- Changed [#36](https://github.com/roboflow/supervision/pull/36): `Detections.wth_nms` support class agnostic and non-class agnostic case.

### 0.3.0 <small>March 6, 2023</small>

- Changed: Allow `Detections.confidence` to be `None`.
- Added: `Detections.from_transformers` and `Detections.from_detectron2` to enable seamless integration with Transformers and Detectron2 models.
- Added: `Detections.area` to dynamically calculate bounding box area.
- Added: `Detections.wth_nms` to filter out double detections with NMS. Initial - only class agnostic - implementation.

### 0.2.0 <small>February 2, 2023</small>

- Added: Advanced `Detections` filtering with pandas-like API.
- Added: `Detections.from_yolov5` and `Detections.from_yolov8` to enable seamless integration with YOLOv5 and YOLOv8 models.

### 0.1.0 <small>January 19, 2023</small>

Say hello to Supervision 👋
