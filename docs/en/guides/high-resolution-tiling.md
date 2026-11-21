---
title: YOLO26 High Resolution Tiling for Tiny Objects
comments: true
description: Detect small objects in large images with the Ultralytics YOLO tile argument. Train, validate, and predict on overlapping tiles without installing SAHI.
keywords: Ultralytics, YOLO26, high resolution tiling, tile argument, tiled inference, sliced inference, SAHI alternative, small object detection, aerial imagery, satellite imagery, drone imagery
---

# High Resolution Tiling with Ultralytics YOLO

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-ultralytics-yolo-high-resolution-tiling.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open High Resolution Tiling In Colab"></a>

Use `tile=True` to process high-resolution images in overlapping windows before resizing them to the model input. This can preserve small-object detail that would be lost by resizing the whole image. No additional package is required, and `tile` replaces the deprecated [SAHI integration](sahi-tiled-inference.md) for Ultralytics models.

|                                                         Whole image, `imgsz=640`: 35 detections                                                          |                                                                                `tile=640`: 161 detections                                                                                |
| :------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![YOLO26s whole-image prediction on a drone view of a city intersection, with 35 detections](https://cdn.ul.run/i/197756a7765c387e6b3ecef2da707ef2.avif) | ![YOLO26s tiled prediction on the same intersection, with 161 detections, including distant pedestrians, cyclists, and cars](https://cdn.ul.run/i/c9e7956b41324d0d2d3f6949d80e5d0e.avif) |

YOLO26s on a 1920 × 1080 [VisDrone](../datasets/detect/visdrone.md) frame at the default confidence threshold. With `tile=640`, the model runs eight overlapping 640-pixel windows at native resolution instead of one pass over the frame shrunk to 640 pixels.

Tiling uses the normal training, validation, and prediction pipelines. Prediction returns one [`Results`](../reference/engine/results.md) object per original image. Validation merges predictions before evaluating them against the original annotations; training loss is computed on the tiles.

## Choose Between `imgsz` and `tile`

Raising `imgsz` and tiling both give small objects more pixels. A larger `imgsz` keeps the whole image in one model pass, while `tile=True` keeps every pass at `imgsz`, which matches the training resolution when `imgsz` is the size the model was trained at. The [benchmarks](#benchmarks) below cover two workloads, and their best settings are candidates to validate on your own data:

| Workload tested                                           | Setting                              | Measured with YOLO26s and YOLO26s-obb                                                                           |
| --------------------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| VisDrone drone images, COCO weights not trained on tiles  | A larger `imgsz`, without `tile`     | `imgsz=1920` reached 23.7 mAP against 21.7 mAP with `tile=640`, at 5.8 ms against 13.5 ms per 1920 × 1080 frame |
| DOTAv1 aerial images, weights trained on 1024-pixel crops | `tile` equal to the training `imgsz` | `tile=1024` reached 64.4 mAP in 40.8 ms and 2.2 GB, against 61.1 mAP in 60.9 ms and 9.0 GB with `imgsz=4096`    |

## Usage

Pass `tile` to [train](../modes/train.md), [val](../modes/val.md), [predict](../modes/predict.md), or [track](../modes/track.md):

!!! example "Train, validate, and predict with tiling"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.train(data="aerial.yaml", tile=True, imgsz=640)
        metrics = model.val(data="aerial.yaml", tile=True, imgsz=640)
        results = model.predict("large-image.tif", tile=640, imgsz=640, batch=8)
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.pt source=large-image.tif tile=640 imgsz=640 batch=8
        ```

`tile=True` uses `imgsz` as the window size. A positive integer, such as `tile=1024`, sets the window size independently of `imgsz`, and `tile=[768, 1280]` sets a rectangular `[h, w]` window. `tile=False` is the default. Neighboring windows share the `tile_overlap` fraction of the tile, 20% by default, and the last row and column are aligned to the image boundary, so they can overlap more. A 1920 × 1080 image at `tile=640` uses a 4 × 2 grid of windows. Smaller images use a single full-image window.

`batch` controls how many tiles the model processes together during prediction. The existing backend handles fixed export batch sizes, including the last partial batch. [Exported models](../modes/export.md) retain their normal input graph; tiling runs in the Python prediction pipeline.

For video, stream results to keep memory bounded and use the existing prediction options to save or display them:

!!! example "Predict on video with tiling"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        for result in model.predict("video.mp4", tile=640, stream=True, save=True):
            print(result.boxes)
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.pt source=video.mp4 tile=640 save=True
        ```

## Tasks and Models

| Task     | Full-image output                                                                                      |
| -------- | ------------------------------------------------------------------------------------------------------ |
| Detect   | Restore boxes to source coordinates and suppress duplicates with `iou`, `agnostic_nms`, and `max_det`. |
| Segment  | Apply the same box selection and place the selected masks in the source image.                         |
| Semantic | Stitch class maps, choosing the most central tile in overlapping regions.                              |
| Depth    | Blend overlapping depth maps with weights increasing toward each tile's center.                        |
| Classify | Average tile probabilities.                                                                            |
| Pose     | Apply box selection and restore the corresponding keypoints.                                           |
| OBB      | Restore rotated boxes and use rotated NMS.                                                             |

The shared pipeline supports YOLO task models, YOLO-World, YOLOE, RT-DETR, and FastSAM prediction. YOLOE visual prompts are embedded from the full source image and reused by its tiles. FastSAM selects prompted masks after merging them into source coordinates. SAM predictors retain their own prompt and crop-generation pipeline; they do not accept `tile`.

During training, boxes and polygons are cropped to their windows, OBB clipping preserves orientation, and out-of-window keypoints lose visibility. Instances with less than 10% of their original bounding-box area remaining are removed. Empty windows remain as negative examples. Semantic and depth targets use the same windows as their images. Each window is one training sample, so `batch` and the epoch length count tiles: the 4 training images of [COCO8](../datasets/detect/coco8.md) become 24 samples at `tile=320`.

## Performance and Quality

- Tiling processes more windows than whole-image inference. Choose the largest tile and batch sizes that preserve the detail you need within available memory. The number of windows per image grows quickly as the tile shrinks, here at the default `tile_overlap=0.2`:

    | Image size  | `tile=320` | `tile=640` | `tile=1024` | `tile=1280` |
    | ----------- | ---------- | ---------- | ----------- | ----------- |
    | 1280 × 720  | 15         | 6          | 2           | 1           |
    | 1920 × 1080 | 32         | 8          | 6           | 2           |
    | 2560 × 1440 | 60         | 15         | 6           | 6           |
    | 3840 × 2160 | 135        | 32         | 15          | 8           |
    | 8000 × 6000 | 744        | 192        | 80          | 48          |

- Prediction runs one window at a time unless you set `batch`. On 1920 × 1080 frames with `tile=640`, `batch=16` cut the time per frame from 26.6 ms to 10.6 ms for YOLO26n and from 56.0 ms to 35.5 ms for YOLO26x, and raised peak GPU memory for YOLO26x from 0.44 GB to 1.57 GB.
- Consecutive tiles reuse the decoded source. With `cache="ram"`, training stores source images once; `cache="disk"` uses the existing image cache. Training shuffles tiles, so caching is especially useful for large compressed source images.
- Source-sized masks and dense maps still require memory proportional to the original image. `max_det` limits the number of retained instance masks.
- Tiling removes global context. Validate accuracy on your own data, especially for classification and depth, whose tile outputs may not agree globally.
- A box cut by an interior window edge is dropped when another window sees the object clear of its edges and a kept box covers it, so seams keep the more complete prediction. A cut box returns to the final NMS pass when no kept box of the same class, or of any class with `agnostic_nms=True`, covers more than half of it. Objects larger than the overlap can still be split, because no window sees them whole, and merging does not join separate fragments. An object smaller than the overlap always fits whole in some window, so raise `tile_overlap` or the tile size for objects you need whole.
- `max_det` caps each window and then the merged image, so a tiled image never returns more than `max_det` detections. Raise it above the default of 300 for dense scenes.
- Tensor sources, `embed`, and `visualize` are unsupported. Tracking runs on the merged predictions once per source frame, including ReID features.

## Benchmarks

Both benchmarks ran Ultralytics 8.4.216 in PyTorch FP32 on one **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPU. Accuracy is mAP<sup>val</sup> 50-95 from `model.val()`, and latency is the full `model.predict()` call for one already decoded image, with tiles batched 16 at a time unless noted.

### Large Aerial Images

The pretrained [YOLO26 OBB](../tasks/obb.md) models, trained at `imgsz=1024`, were validated on the 458 original images of the [DOTAv1](../datasets/obb/dota-v2.md) val split without cropping them first. Their long side has a median of about 2140 pixels and a maximum of 13,383 pixels. Every `tile` column uses `imgsz=1024`.

=== "Accuracy"

    mAP<sup>val</sup> 50-95, higher is better. The best value for each model is in bold.

    | Model       | `imgsz=1024` | `imgsz=2048` | `imgsz=4096` | `tile=2048` | `tile=1024` | `tile=768` |
    | ----------- | ------------ | ------------ | ------------ | ----------- | ----------- | ---------- |
    | YOLO26n-obb | 46.5         | 57.3         | 60.0         | 58.4        | 62.7        | **63.1**   |
    | YOLO26s-obb | 50.6         | 60.2         | 61.1         | 61.0        | **64.4**    | 64.1       |
    | YOLO26m-obb | 53.8         | 62.7         | 62.2         | 63.7        | **64.9**    | 64.8       |
    | YOLO26l-obb | 54.2         | 62.6         | 62.1         | 63.6        | **65.4**    | 64.7       |
    | YOLO26x-obb | 55.8         | 63.5         | 62.7         | 64.7        | **65.5**    | 64.9       |

=== "Latency"

    Mean milliseconds per image, lower is better, on every twelfth image of the split: 39 images with a long side of 567 to 5441 pixels. `tile=2048` ran 3.7 windows per image on average, `tile=1024` ran 11.7, and `tile=768` ran 20.9.

    | Model       | `imgsz=1024` | `imgsz=2048` | `imgsz=4096` | `tile=2048` | `tile=1024` | `tile=768` |
    | ----------- | ------------ | ------------ | ------------ | ----------- | ----------- | ---------- |
    | YOLO26n-obb | 6.1          | 9.4          | 37.5         | 12.6        | 25.7        | 58.6       |
    | YOLO26s-obb | 5.9          | 13.2         | 60.9         | 16.6        | 40.8        | 86.4       |
    | YOLO26m-obb | 7.8          | 20.9         | 97.1         | 26.4        | 74.6        | 145.8      |
    | YOLO26l-obb | 9.0          | 25.3         | 120.8        | 30.7        | 87.8        | 169.0      |
    | YOLO26x-obb | 12.5         | 41.7         | 203.6        | 46.4        | 144.2       | 273.9      |

=== "Memory"

    Peak GPU memory in GB during prediction on the same 39 images, lower is better. Tiled values are for `batch=16`; with `batch=1`, `tile=1024` peaked at 0.10 to 0.79 GB.

    | Model       | `imgsz=1024` | `imgsz=2048` | `imgsz=4096` | `tile=2048` | `tile=1024` | `tile=768` |
    | ----------- | ------------ | ------------ | ------------ | ----------- | ----------- | ---------- |
    | YOLO26n-obb | 0.09         | 0.40         | 4.56         | 0.54        | 1.11        | 1.11       |
    | YOLO26s-obb | 0.17         | 0.76         | 8.95         | 1.07        | 2.15        | 2.15       |
    | YOLO26m-obb | 0.33         | 1.06         | 9.57         | 2.10        | 4.13        | 4.13       |
    | YOLO26l-obb | 0.42         | 1.27         | 9.58         | 2.73        | 5.24        | 5.24       |
    | YOLO26x-obb | 0.75         | 2.33         | 14.36        | 4.97        | 9.32        | 9.32       |

- **`tile=1024`** was the most accurate setting for four of the five models and within 0.4 mAP of the best for YOLO26n-obb. Against `imgsz=4096` it scored 2.7 to 3.3 mAP higher, ran 1.3 to 1.5 times faster, and used 1.5 to 4.2 times less GPU memory.
- **`imgsz=4096`** took 4.0 to 4.9 times longer than `imgsz=2048` and scored lower on YOLO26m-obb, YOLO26l-obb, and YOLO26x-obb.
- **`tile=2048`** with `imgsz=1024` scored 0.8 to 1.2 mAP higher than `imgsz=2048` and ran 1.1 to 1.3 times slower.
- **`tile=768`**, a window smaller than the training size, scored 0.4 mAP higher than `tile=1024` on YOLO26n-obb and 0.1 to 0.7 lower on the other four models, and took 1.9 to 2.3 times longer.

### Drone Images

The COCO-pretrained YOLO26 detection models, trained at `imgsz=640`, were validated on the 548 images of the [VisDrone](../datasets/detect/visdrone.md) val split without fine-tuning. Scoring uses the six classes VisDrone shares with COCO: person, bicycle, car, motorcycle, bus, and truck, with the VisDrone pedestrian and people labels counted as person, van as car, and motor as motorcycle. Every `tile` column uses `imgsz=640`.

=== "Accuracy"

    mAP<sup>val</sup> 50-95, higher is better. The best value for each model is in bold.

    | Model   | `imgsz=640` | `imgsz=1280` | `imgsz=1920` | `tile=640` | `tile=480` | `tile=320` |
    | ------- | ----------- | ------------ | ------------ | ---------- | ---------- | ---------- |
    | YOLO26n | 8.8         | 15.6         | **17.3**     | 15.0       | 16.5       | 15.9       |
    | YOLO26s | 13.5        | 21.9         | **23.7**     | 21.7       | 22.5       | 21.3       |
    | YOLO26m | 17.6        | 25.7         | **26.9**     | 25.3       | 25.9       | 23.8       |
    | YOLO26l | 19.0        | 27.2         | **27.5**     | 26.5       | **27.5**   | 25.5       |
    | YOLO26x | 20.9        | 30.2         | **30.8**     | 29.1       | 29.9       | 27.3       |

=== "Latency"

    Median milliseconds per image, lower is better, on the 19 val images that are exactly 1920 × 1080. `tile=640` runs 8 windows per frame, `tile=480` runs 15, and `tile=320` runs 32 with `batch=32`. YOLO26n measured faster at `imgsz=1920` than at `imgsz=1280` in two separate runs.

    | Model   | `imgsz=640` | `imgsz=1280` | `imgsz=1920` | `tile=640` | `tile=480` | `tile=320` |
    | ------- | ----------- | ------------ | ------------ | ---------- | ---------- | ---------- |
    | YOLO26n | 3.7         | 4.7          | 3.9          | 10.6       | 22.7       | 44.3       |
    | YOLO26s | 3.9         | 5.0          | 5.8          | 13.5       | 28.8       | 58.5       |
    | YOLO26m | 4.3         | 6.9          | 10.1         | 22.1       | 43.6       | 93.9       |
    | YOLO26l | 5.5         | 8.4          | 12.4         | 26.1       | 49.1       | 108.6      |
    | YOLO26x | 6.4         | 12.0         | 22.5         | 35.5       | 67.7       | 157.6      |

- **`tile=640`** scored 6.2 to 8.2 mAP higher than `imgsz=640`, the gain that tiling gives over shrinking the frame to the model input.
- **`imgsz=1280`** scored 0.2 to 1.1 mAP higher than `tile=640` and ran 2.3 to 3.2 times faster. **`imgsz=1920`** matched or beat every tiled setting on all five models.
- **`tile=480`** scored 0.6 to 1.5 mAP higher than `tile=640` on all five models and took 1.9 to 2.1 times longer.
- **`tile=320`** raised recall over `tile=640` by 2.6 to 6.0 points on all five models, lowered mAP on four of them, and took 4.2 to 4.4 times longer, so compare window sizes on your own validation set.

## Migrate from SAHI

| SAHI argument                                 | `tile` equivalent                                                                                          |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `slice_height`, `slice_width`                 | `tile=[h, w]`, `tile=<int>` for a square window, or `tile=True` to use `imgsz`                             |
| `overlap_height_ratio`, `overlap_width_ratio` | `tile_overlap`, one fraction for both axes                                                                 |
| `postprocess_match_threshold`                 | `iou`: IoU-based NMS that suppresses boxes instead of merging them, prediction default 0.5 for YOLO models |
| `postprocess_class_agnostic`                  | `agnostic_nms`                                                                                             |
| `confidence_threshold`, `image_size`          | `conf`, `imgsz`                                                                                            |
| `perform_standard_pred`                       | No equivalent: `tile` does not add a full-image pass                                                       |
| Sliced dataset export                         | Not needed: `tile` crops images and labels on the fly during training                                      |

## FAQ

### What does the `tile` argument do in Ultralytics YOLO?

`tile` splits each image into overlapping windows, runs the model on every window, and merges the outputs back into one result per original image. It works in training, validation, prediction, and tracking. With `tile=True`, windows match `imgsz`, so small objects keep the pixels they would lose if the whole image were resized to `imgsz`.

### How do I choose the tile size?

Start with a tile equal to the `imgsz` the model was trained at: `tile=True` uses `imgsz` as the window size, and an integer such as `tile=1024` sets it independently. On DOTAv1 aerial images, `tile=1024` was the most accurate setting for four of the five YOLO26 OBB models, which were trained on 1024-pixel crops. On VisDrone with COCO weights, `tile=480` scored 0.6 to 1.5 mAP higher than `tile=640`. At the default `tile_overlap=0.2`, a 1920 × 1080 image at `tile=640` runs eight windows and a 3840 × 2160 image at `tile=1280` also runs eight. A window smaller than `imgsz`, such as `tile=400` with `imgsz=640`, is enlarged to the model input, which helps when objects are only a few pixels wide even at native resolution. Smaller windows keep more detail but add windows to process, remove more global context, and can enlarge background clutter into false positives, so compare a few sizes on your own validation set.

### Should I increase `imgsz` or use `tile` for small objects?

It depends on the workload, so validate both. In the [benchmarks](#benchmarks), COCO-pretrained YOLO26 models on VisDrone drone images scored highest with `imgsz=1920` and no tiling, while YOLO26 OBB models trained on 1024-pixel crops scored higher on DOTAv1 aerial images with `tile=1024` than with `imgsz=4096`, in less time and GPU memory. See [Choose Between `imgsz` and `tile`](#choose-between-imgsz-and-tile).

### Do I still need SAHI for sliced inference with YOLO?

No. The `tile` argument handles sliced inference without an extra package, and it also tiles training and validation. Prediction returns standard [`Results`](../reference/engine/results.md) objects, and validation scores the merged predictions against the original annotations. The [SAHI guide](sahi-tiled-inference.md) is deprecated, and [Migrate from SAHI](#migrate-from-sahi) maps its arguments to `tile`.

### Should I train with tiling as well as predict with it?

Use the same `tile` setting for training and prediction when your images are much larger than `imgsz`. Training on tiles shows the model objects at the scale it sees during tiled prediction, clips labels to each window, and keeps empty windows as negative examples. Validate with `tile` too, since validation merges tile predictions before computing metrics on the full images.

### Does tiling work with exported models such as ONNX or TensorRT?

Yes. Tiling runs in the Python prediction pipeline, so [exported models](../modes/export.md) keep their normal input graph. Load the exported file with `YOLO("yolo26n.onnx")` and pass `tile` to `predict` as usual. `batch` sets how many tiles run together, and fixed export batch sizes are handled, including the last partial batch.

### Which tasks and models support tiling?

Detection, segmentation, semantic segmentation, depth, classification, pose, and OBB all support `tile`, as do YOLO-World, YOLOE, RT-DETR, and FastSAM prediction. SAM keeps its own prompt and crop pipeline and does not accept `tile`. See [Tasks and Models](#tasks-and-models) for how each task merges tile outputs.
