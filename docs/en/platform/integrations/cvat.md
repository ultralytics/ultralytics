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
2. **Pick the format.** Choose the **[Ultralytics YOLO](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)** entry matching your task — CVAT lists `Ultralytics YOLO Detection 1.0`, `Ultralytics YOLO Detection Track 1.0`, `Ultralytics YOLO Segmentation 1.0`, `Ultralytics YOLO Oriented Bounding Boxes 1.0`, `Ultralytics YOLO Pose 1.0`, and `Ultralytics YOLO Classification 1.0` separately.
3. **Turn on Save images**, name the `.zip`, and click **OK**. Save images is a paid feature on CVAT Online; without it the archive holds only labels, so add your images in an `images/` folder that mirrors `labels/` before uploading.
4. **Download the archive.** The export runs in the background — collect it from CVAT's [Requests](https://docs.cvat.ai/docs/workspace/requests-page/) page when it finishes.
5. **Upload to Platform.** [Create a new dataset](../data/datasets.md) from the ZIP.
6. **Train.** [Edit the annotations](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) without leaving the workspace.

![Ultralytics Platform CVAT Dataset Import](https://cdn.ul.run/i/bdc1bd479481181ca5088f467ea945c6.avif)<!-- screenshot -->

### Export with the CLI

[CVAT's CLI](https://docs.cvat.ai/docs/api_sdk/cli/) exports the same archive from a terminal:

```bash
pip install cvat-cli
cvat-cli project export-dataset --format "Ultralytics YOLO Detection 1.0" --with-images yes 104 dataset.zip
```

Replace `104` with your project ID and the format string with the variant matching your task. `--with-images yes` is the CLI equivalent of the **Save images** switch; without it the archive holds only labels, so add an `images/` folder that mirrors `labels/` before uploading.

CVAT's Ultralytics YOLO export produces the layout Platform expects, so nothing needs converting:

```text
archive.zip/
├── data.yaml          # class names Platform reads
├── images/train/
└── labels/train/
```

### What Each Export Keeps

Each Ultralytics YOLO format writes only the CVAT shapes its task can hold, so pick the one matching how you labeled:

- **Detection** keeps rectangles, but skips rotated ones — export those with **Oriented Bounding Boxes**
- **Detection Track** keeps rectangles plus a track ID; Platform imports them as boxes and ignores the track ID
- **Segmentation** keeps polygons and masks, converting masks to polygons, and drops rectangles
- **Pose** keeps skeletons only; standalone points are not exported
- **Classification** keeps tags, and images without a tag land in a `no_label` class

Polylines are not written by any of them.

## Choosing an Export Format

CVAT offers [many export formats](https://docs.cvat.ai/docs/dataset_management/formats/). These matter here:

| CVAT Format            | Works  | Notes                                                                                        |
| ---------------------- | ------ | -------------------------------------------------------------------------------------------- |
| **Ultralytics YOLO**   | Best   | Ships `data.yaml`, so your label names come across intact                                    |
| **COCO 1.0**           | Yes    | Read too; a mix of polygons and boxes imports as segment, and the box-only ones are dropped  |
| **COCO Keypoints 1.0** | Yes    | Imports as a pose dataset, with the keypoint count taken from the most common shape          |
| **YOLO 1.1**           | Partly | Boxes import, but its `obj.names` file is not read — classes arrive as `class0`, `class1`, … |
| **CVAT for images**    | No     | The XML annotations are not read yet; the same applies to **CVAT for video**                 |

Every COCO annotation must carry a `bbox` to be read, and crowd regions (`"iscrowd": 1`) are skipped. CVAT writes masks
to COCO as crowd regions, so export masks with Ultralytics YOLO Segmentation instead. Category names
become the class names, and category IDs are unified across all the JSON files in the archive, so per-split exports keep
consistent class IDs. An object stored as several polygons is joined into one polygon.

!!! warning "CVAT XML and Pascal VOC import without annotations"

    Platform does not read CVAT for images, CVAT for video, or Pascal VOC XML labels, and these exports fail quietly rather than loudly: the images import, the annotations do not, and an export of five or more images also picks up classes named after its image folders. Choose Ultralytics YOLO or COCO instead.

## What the Integration Will Add

Picking the right export format is the step the integration removes. Once it ships, you will export any of CVAT's image detection and segmentation formats, upload it, and Platform will map the annotations to the matching [YOLO task](../data/index.md#supported-tasks) itself.

- **No format to choose** — CVAT's image detection and segmentation exports map to a YOLO dataset with label names preserved
- **One workspace** — labeling, training, and deployment stop spanning separate tools and a conversion script
- **Keep annotating** — imported datasets open in Platform's [annotation editor](../data/annotation.md), including SAM-powered smart annotation
- **Train immediately** — datasets are ready for [cloud training](../train/cloud-training.md) as soon as they finish processing

!!! tip "Available now"

    The [Labelbox](labelbox.md) and [Roboflow](roboflow.md) integrations work today, and Platform imports YOLO, COCO, and Ultralytics NDJSON datasets directly.

## FAQ

### Which CVAT export format should I choose?

Choose the **Ultralytics YOLO** entry that matches your task, with **Save images** turned on. It ships `data.yaml`, so class names arrive intact. COCO 1.0 and COCO Keypoints 1.0 also import, while YOLO 1.1 loses class names and CVAT XML and Pascal VOC import without annotations.

### Why did my dataset import without any images?

The **Save images** switch was off, or the CLI export ran without `--with-images yes`. The archive then holds only labels. Re-export with images included, or, on a CVAT Online plan without Save images, add an `images/` folder that mirrors `labels/` to the ZIP.

### Does the import support segmentation and pose?

Yes. Use `Ultralytics YOLO Segmentation 1.0` or `Ultralytics YOLO Pose 1.0`, or the COCO variants. Masks need the Ultralytics YOLO Segmentation export, which converts them to polygons, and pose needs CVAT skeletons. A COCO export mixing polygons and boxes imports as a segment dataset and drops the box-only annotations.

### What will change when the direct integration ships?

You will upload any CVAT image detection or segmentation export as-is and Platform will map it to the matching YOLO task itself. Datasets you import today with the Ultralytics YOLO export continue to work unchanged.
