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
2. **Pick a format that includes the images.** Choose **[YOLO with Images](https://labelstud.io/guide/export)** and click **Export**.
3. **Upload to Platform.** [Create a new dataset](../data/datasets.md) from the ZIP.
4. **Train.** [Edit the annotations](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) without leaving the workspace.

![Ultralytics Platform Label Studio Dataset Import](https://cdn.ul.run/i/730d485fb7b6856bd0fd91de876c67e1.avif)<!-- screenshot -->

!!! warning "Plain YOLO and COCO export annotations only"

    Label Studio's `YOLO` and `COCO` options write label files without the images, because your images normally live behind the URLs Label Studio was pointed at. Uploading one of those archives to Platform gives you a dataset with no images. Pick the **with Images** variant instead.

### Export with the API

Label Studio's [export API](https://labelstud.io/guide/export) returns the same archive, with the format name in `exportType`:

```bash
curl -X GET "https://<your-label-studio>/api/projects/<project-id>/export?exportType=YOLO_WITH_IMAGES&download_all_tasks=true" \
  -H "Authorization: Token <your-legacy-token>" \
  -o dataset.zip
```

`COCO_WITH_IMAGES` and `YOLO_OBB_WITH_IMAGES` work the same way. Newer [personal access tokens](https://labelstud.io/guide/access_tokens) are exchanged for a short-lived bearer token before use, so check which token type your instance issues. Large projects should use the [snapshot endpoints](https://labelstud.io/guide/export) instead, which create the export as a background job and download it by ID; the SDK's [`export_yolo_with_images.py`](https://github.com/HumanSignal/label-studio-sdk/blob/master/examples/export_yolo_with_images.py) example does exactly that.

A Label Studio YOLO export carries its class list alongside the labels, and Platform reads it:

```
archive.zip/
├── classes.txt        # class names Platform reads first
├── notes.json         # fallback, used only when classes.txt is missing or empty
├── images/
└── labels/
```

Upload the archive exactly as Label Studio produced it. Platform looks for `classes.txt` and `notes.json` at the root of the archive, so re-zipping the export inside another folder loses your label names and the classes import as `class0`, `class1`, and so on.

## Choosing an Export Format

Label Studio offers [several export formats](https://labelstud.io/guide/export). For image detection and segmentation:

| Label Studio Format  | Works | Notes                                                                                |
| -------------------- | ----- | ------------------------------------------------------------------------------------ |
| **YOLO with Images** | Best  | Ships `classes.txt` and the images, so a single upload is enough                     |
| **COCO with Images** | Yes   | Read too; a mix of polygons and boxes imports as segment, dropping the box-only ones |
| **YOLO** / **COCO**  | No    | Annotation files only — the dataset imports with no images                           |
| **Pascal VOC XML**   | No    | XML label files cannot be read                                                       |

In a COCO export, every annotation must carry a `bbox` to be read, crowd regions (`"iscrowd": 1`) are skipped, and the
category names become your class names — so a COCO archive does not need `classes.txt` to keep its labels.

!!! warning "Pascal VOC imports without annotations"

    Platform does not read Pascal VOC XML labels, and a VOC export fails quietly rather than loudly: the images import and the annotations do not. Choose YOLO with Images or COCO with Images instead.

## What the Integration Will Add

Picking the right export format is the step the integration removes. Once it ships, you will export any of Label Studio's image detection and segmentation formats, upload it, and Platform will map the annotations to the matching [YOLO task](../data/index.md#supported-tasks) itself.

- **No format to choose** — Label Studio's image detection and segmentation exports map to a YOLO dataset with label names preserved
- **One workspace** — labeling, training, and deployment stop spanning separate tools and a conversion script
- **Keep annotating** — imported datasets open in Platform's [annotation editor](../data/annotation.md), including SAM-powered smart annotation
- **Train immediately** — datasets are ready for [cloud training](../train/cloud-training.md) as soon as they finish processing

!!! tip "Available now"

    The [Labelbox](labelbox.md) and [Roboflow](roboflow.md) integrations work today, and Platform imports YOLO, COCO, and Ultralytics NDJSON datasets directly.
