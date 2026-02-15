---
description: Explore Ultralytics' annotator script for automatic image annotation using YOLO and SAM models. Contribute to improve it on GitHub.
keywords: Ultralytics, image annotation, YOLO, SAM, Python script, GitHub, object detection, segmentation
---

# Reference for `ultralytics/data/annotator.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/annotator.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/annotator.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`auto_annotate`](#ultralytics.data.annotator.auto_annotate)


## Function `ultralytics.data.annotator.auto_annotate` {#ultralytics.data.annotator.auto\_annotate}

```python
def auto_annotate(
    data: str | Path,
    det_model: str = "yolo26x.pt",
    sam_model: str = "sam_b.pt",
    device: str = "",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    classes: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> None
```

Automatically annotate images using a YOLO object detection model and a SAM segmentation model.

This function processes images in a specified directory, detects objects using a YOLO model, and then generates segmentation masks using a SAM model. The resulting annotations are saved as text files in YOLO format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data` | `str | Path` | Path to a folder containing images to be annotated. | *required* |
| `det_model` | `str` | Path or name of the pre-trained YOLO detection model. | `"yolo26x.pt"` |
| `sam_model` | `str` | Path or name of the pre-trained SAM segmentation model. | `"sam_b.pt"` |
| `device` | `str` | Device to run the models on (e.g., 'cpu', 'cuda', '0'). Empty string for auto-selection. | `""` |
| `conf` | `float` | Confidence threshold for detection model. | `0.25` |
| `iou` | `float` | IoU threshold for filtering overlapping boxes in detection results. | `0.45` |
| `imgsz` | `int` | Input image resize dimension. | `640` |
| `max_det` | `int` | Maximum number of detections per image. | `300` |
| `classes` | `list[int], optional` | Filter predictions to specified class IDs, returning only relevant detections. | `None` |
| `output_dir` | `str | Path, optional` | Directory to save the annotated results. If None, creates a default directory<br>    based on the input data path. | `None` |

**Examples**

```python
>>> from ultralytics.data.annotator import auto_annotate
>>> auto_annotate(data="ultralytics/assets", det_model="yolo26n.pt", sam_model="mobile_sam.pt")
```

<details>
<summary>Source code in <code>ultralytics/data/annotator.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/annotator.py#L10-L66"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def auto_annotate(
    data: str | Path,
    det_model: str = "yolo26x.pt",
    sam_model: str = "sam_b.pt",
    device: str = "",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    classes: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """Automatically annotate images using a YOLO object detection model and a SAM segmentation model.

    This function processes images in a specified directory, detects objects using a YOLO model, and then generates
    segmentation masks using a SAM model. The resulting annotations are saved as text files in YOLO format.

    Args:
        data (str | Path): Path to a folder containing images to be annotated.
        det_model (str): Path or name of the pre-trained YOLO detection model.
        sam_model (str): Path or name of the pre-trained SAM segmentation model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0'). Empty string for auto-selection.
        conf (float): Confidence threshold for detection model.
        iou (float): IoU threshold for filtering overlapping boxes in detection results.
        imgsz (int): Input image resize dimension.
        max_det (int): Maximum number of detections per image.
        classes (list[int], optional): Filter predictions to specified class IDs, returning only relevant detections.
        output_dir (str | Path, optional): Directory to save the annotated results. If None, creates a default directory
            based on the input data path.

    Examples:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolo26n.pt", sam_model="mobile_sam.pt")
    """
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(
        data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes
    )

    for result in det_results:
        if class_ids := result.boxes.cls.int().tolist():  # Extract class IDs from detection results
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w", encoding="utf-8") as f:
                for i, s in enumerate(segments):
                    if s.any():
                        segment = map(str, s.reshape(-1).tolist())
                        f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
```
</details>

<br><br>
