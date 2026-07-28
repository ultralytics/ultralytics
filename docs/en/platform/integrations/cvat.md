---
plans: [free, pro, enterprise]
coming_soon: true
comments: true
description: Move a CVAT task or project into Ultralytics Platform with the Ultralytics YOLO export, then edit annotations, train YOLO models, and deploy from one workspace.
keywords: Ultralytics Platform, CVAT, CVAT export, Ultralytics YOLO format, dataset import, annotation, integrations, YOLO, computer vision
title: CVAT Dataset Import - Ultralytics Platform
---

# CVAT Integration

Direct [CVAT](https://www.cvat.ai/) imports are coming to [Ultralytics Platform](https://platform.ultralytics.com), so that a CVAT image detection or segmentation export uploads as-is with no format to choose.

Until then there is a short path that works today, because CVAT already exports in the Ultralytics YOLO layout Platform reads.

## Import from CVAT Today

1. **Export from CVAT.** Open your task and choose **Actions > Export task dataset** (from a job it is **Menu > Export job dataset**).
2. **Pick the format.** Choose the **[Ultralytics YOLO](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)** entry matching your task — CVAT lists `Ultralytics YOLO Detection 1.0`, `Ultralytics YOLO Segmentation 1.0`, `Ultralytics YOLO Oriented Bounding Boxes 1.0`, and `Ultralytics YOLO Pose 1.0` separately.
3. **Turn on Save images**, name the `.zip`, and click **OK**. Without the images the archive holds annotations only, and Platform has nothing to import.
4. **Download the archive.** The export runs in the background — collect it from CVAT's [Requests](https://docs.cvat.ai/docs/workspace/requests-page/) page when it finishes.
5. **Upload to Platform.** [Create a new dataset](../data/datasets.md) from the ZIP.
6. **Train.** [Edit the annotations](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) without leaving the workspace.

### Export with the CLI

[CVAT's CLI](https://docs.cvat.ai/docs/api_sdk/cli/) exports the same archive from a terminal:

```bash
pip install cvat-cli
cvat-cli project export-dataset --format "Ultralytics YOLO Detection 1.0" --with-images yes 104 dataset.zip
```

Replace `104` with your project ID and the format string with the variant matching your task. `--with-images yes` is the CLI equivalent of the **Save images** switch; without it the archive holds annotations only.

CVAT's Ultralytics YOLO export produces the layout Platform expects, so nothing needs converting:

```
archive.zip/
├── data.yaml          # class names Platform reads
├── images/train/
└── labels/train/
```

## Choosing an Export Format

CVAT offers [many export formats](https://docs.cvat.ai/docs/dataset_management/formats/). Three matter here:

| CVAT Format          | Works  | Notes                                                                                        |
| -------------------- | ------ | -------------------------------------------------------------------------------------------- |
| **Ultralytics YOLO** | Best   | Ships `data.yaml`, so your label names come across intact                                    |
| **COCO 1.0**         | Yes    | Read too; a mix of polygons and boxes imports as segment, and the box-only ones are dropped  |
| **YOLO 1.1**         | Partly | Boxes import, but its `obj.names` file is not read — classes arrive as `class0`, `class1`, … |

!!! warning "Pascal VOC imports without annotations"

    Platform does not read Pascal VOC XML labels, and a VOC export fails quietly rather than loudly: the images import, the annotations do not, and a VOC export of five or more images also picks up a single class named after its image folder. Choose Ultralytics YOLO or COCO instead.

## What the Integration Will Add

Picking the right export format is the step the integration removes. Once it ships, you will export any of CVAT's image detection and segmentation formats, upload it, and Platform will map the annotations to the matching [YOLO task](../data/index.md#supported-tasks) itself.

- **No format to choose** — CVAT's image detection and segmentation exports map to a YOLO dataset with label names preserved
- **One workspace** — labeling, training, and deployment stop spanning separate tools and a conversion script
- **Keep annotating** — imported datasets open in Platform's [annotation editor](../data/annotation.md), including SAM-powered smart annotation
- **Train immediately** — datasets are ready for [cloud training](../train/cloud-training.md) as soon as they finish processing

!!! tip "Available now"

    The [Labelbox](labelbox.md), [Roboflow](roboflow.md), and [Ultralytics HUB](ultralytics-hub.md) integrations work today, and Platform imports YOLO, COCO, and Ultralytics NDJSON datasets directly.
