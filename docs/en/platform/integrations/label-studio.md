---
plans: [free, pro, enterprise]
coming_soon: true
comments: true
description: Move a Label Studio project into Ultralytics Platform with the YOLO export, then edit annotations, train YOLO models, and deploy from one workspace.
keywords: Ultralytics Platform, Label Studio, Label Studio export, YOLO export, dataset import, annotation, integrations, YOLO, computer vision
title: Label Studio Dataset Import - Ultralytics Platform
---

# Label Studio Integration

Direct [Label Studio](https://labelstud.io/) imports are coming to [Ultralytics Platform](https://platform.ultralytics.com), so that a raw image detection or segmentation export uploads as-is with no format to choose.

Until then there is a short path that works today, because Label Studio's YOLO with Images export is a layout Platform reads, class list included.

## Import from Label Studio Today

1. **Export from Label Studio.** Open your project and click **Export**.
2. **Pick a format that includes the images.** Choose **[YOLO with Images](https://labelstud.io/guide/export)**, or **YOLOv8 OBB with Images** for rotated boxes, and click **Export**.
3. **Upload to Platform.** [Create a new dataset](../data/datasets.md) from the ZIP.
4. **Train.** [Edit the annotations](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) without leaving the workspace.

![Ultralytics Platform Label Studio Dataset Import](https://cdn.ul.run/i/c64a0b325b244a5c633a474ac19bd643.avif)<!-- screenshot -->

!!! warning "Plain YOLO and COCO export annotations only"

    Label Studio's `YOLO` and `COCO` options write label files without the images, because your images normally live behind the URLs Label Studio was pointed at. Uploading one of those archives to Platform fails with no images found. Pick the **with Images** variant instead.

### Export with the API

Label Studio's [export API](https://labelstud.io/guide/export) returns the same archive, with the format name in `exportType`:

```bash
curl -X GET "https://<your-label-studio>/api/projects/<project-id>/export?exportType=YOLO_WITH_IMAGES&download_all_tasks=true" \
  -H "Authorization: Token <your-legacy-token>" \
  -o dataset.zip
```

`COCO_WITH_IMAGES` and `YOLO_OBB_WITH_IMAGES` work the same way. Newer [personal access tokens](https://labelstud.io/guide/access_tokens) are exchanged for a short-lived bearer token before use, so check which token type your instance issues. Large projects should use the [snapshot endpoints](https://labelstud.io/guide/export) instead, which create the export as a background job and download it by ID; the SDK's [`export_yolo_with_images.py`](https://github.com/HumanSignal/label-studio-sdk/blob/master/examples/export_yolo_with_images.py) example does exactly that.

A Label Studio YOLO export carries its class list alongside the labels, and Platform reads it:

```text
archive.zip/
├── classes.txt        # class names Platform reads first
├── notes.json         # fallback, used only when classes.txt is missing or empty
├── images/
└── labels/
```

Platform reads the shallowest `classes.txt` (or `notes.json`) in the archive, so an export you unzipped and re-zipped inside a folder keeps its label names too.

## Choosing an Export Format

Label Studio offers [several export formats](https://labelstud.io/guide/export). For image detection and segmentation:

| Label Studio Format        | Works | Notes                                                                  |
| -------------------------- | ----- | ---------------------------------------------------------------------- |
| **YOLO with Images**       | Best  | Ships `classes.txt` and the images, so a single upload is enough       |
| **YOLOv8 OBB with Images** | Yes   | Keeps box rotation and imports as an OBB dataset                       |
| **COCO with Images**       | Yes   | Read too; the better choice if each annotation is a four-point polygon |
| **YOLO** / **COCO**        | No    | Annotation files only — the upload fails with no images found          |
| **Pascal VOC XML**         | No    | XML label files cannot be read                                         |

In a COCO export, every annotation must carry a `bbox` to be read, crowd regions (`"iscrowd": 1`) are skipped, and the
category names become your class names — so a COCO archive does not need `classes.txt` to keep its labels.

!!! warning "Pascal VOC imports without annotations"

    Platform does not read Pascal VOC XML labels, and a VOC export fails quietly rather than loudly: the images import, the boxes do not, and an export of five or more images also picks up a single class named `images`. Choose YOLO with Images or COCO with Images instead.

## What Carries Over

Platform picks the [YOLO task](../data/index.md#supported-tasks) from the labels in the export:

| Label Studio labels                 | Platform dataset                                                                    |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| `RectangleLabels`                   | Detect, or OBB from a YOLOv8 OBB export                                             |
| `PolygonLabels`                     | Segment, or OBB from a YOLO export when every polygon has four points               |
| `RectangleLabels` + `PolygonLabels` | Segment — the boxes stay editable, but only the polygons are used for training      |
| `KeyPointLabels` inside a rectangle | Pose                                                                                |
| `Choices`                           | Not exported by Label Studio's YOLO or COCO formats, so the images import unlabeled |

- **Rotated boxes** — the YOLO and COCO exports drop the rotation; use **YOLOv8 OBB with Images** to keep it.
- **Keypoints** — Label Studio's YOLO export includes a keypoint only when its `<Label>` in `KeyPointLabels` has a `model_index` attribute and the point was drawn inside a rectangle; otherwise the export holds the boxes alone and the dataset imports as detect.
- **Four-point polygons** — in YOLO files a polygon with exactly four points looks the same as an oriented box, so a YOLO export in which every label is a four-point polygon imports as OBB. Use **COCO with Images** for those projects.

## What the Integration Will Add

Picking the right export format is the step the integration removes. Once it ships, you will export any of Label Studio's image detection and segmentation formats, upload it, and Platform will map the annotations to the matching [YOLO task](../data/index.md#supported-tasks) itself.

- **No format to choose** — Label Studio's image detection and segmentation exports map to a YOLO dataset with label names preserved
- **One workspace** — labeling, training, and deployment stop spanning separate tools and a conversion script
- **Keep annotating** — imported datasets open in Platform's [annotation editor](../data/annotation.md), including SAM-powered smart annotation
- **Train immediately** — datasets are ready for [cloud training](../train/cloud-training.md) as soon as they finish processing

!!! tip "Available now"

    The [Labelbox](labelbox.md) and [Roboflow](roboflow.md) integrations work today, and Platform imports YOLO, COCO, [LabelMe](labelme.md) JSON, and Ultralytics NDJSON datasets directly.

## FAQ

### Which Label Studio export format should I choose?

Choose **YOLO with Images** so the archive contains `classes.txt` and the images. **COCO with Images** also imports. The plain YOLO and COCO options write label files only, and Pascal VOC XML is not read.

### Why did my upload fail with no images found?

The plain `YOLO` or `COCO` export was used. Those archives omit the images because Label Studio normally serves them from URLs. Re-export with the **with Images** variant.

### Why are my classes named `class0`, `class1`, and so on?

The archive has no `classes.txt` or `notes.json`, usually because it was rebuilt from the `images/` and `labels/` folders alone. Upload the ZIP Label Studio exported, which ships both files.

### Can I export from the API instead of the UI?

Yes. Call the export endpoint with `exportType=YOLO_WITH_IMAGES`, or `COCO_WITH_IMAGES` and `YOLO_OBB_WITH_IMAGES`. Large projects should use the snapshot endpoints, which build the export as a background job.
