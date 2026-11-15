---
plans: [free, pro, enterprise]
comments: true
description: Import hosted Label Studio projects into Ultralytics Platform with an access token as YOLO detect, OBB, segment, or classify datasets, or upload a self-hosted export.
keywords: Ultralytics Platform, Label Studio, Label Studio import, Label Studio to YOLO, HumanSignal, access token, Label Studio export, YOLO export, dataset import, annotation, integrations, YOLO, computer vision
title: Label Studio Dataset Import - Ultralytics Platform
---

# Label Studio Integration

The Label Studio integration imports the projects you select from hosted [Label Studio](https://labelstud.io/) (`app.humansignal.com`) into [Ultralytics Platform](https://platform.ultralytics.com), one dataset per project with each task's latest annotation, so you can edit the labels, train YOLO models, and deploy them without exporting a file. Self-hosted Label Studio projects come in through an [export upload](#import-from-self-hosted-label-studio) instead.

## Import from Label Studio

1. Go to **Settings > [Integrations](index.md)** and select **Label Studio** from the integration list.
2. Paste a Label Studio [personal access token or legacy token](https://docs.humansignal.com/guide/access_tokens) into **Label Studio API Key** and click **Connect**. Only the workspace owner can enter a token. If the workspace has a [saved Label Studio token](../account/api-keys.md#provider-keys-for-agents-and-annotation), the preview opens with it automatically.
3. Review the **Import from Label Studio** preview:
    - A table of the supported projects (name, task, images), with nothing selected by default; select the new projects you want to import, or click a name to open the project in Label Studio
    - Already-imported projects shown as dimmed, unselectable rows with an **Already imported** tooltip
    - A count of projects whose labeling setup isn't [supported](#supported-labeling-setups)
    - Your storage usage
4. Click **Import N** to import the N selected projects.

The preview stays open after an import. Started and already-imported projects remain visible but cannot be selected, while projects that could not be started stay selected for a retry. Click **Connect** again to refresh the preview.

![Ultralytics Platform Label Studio Dataset Import](https://cdn.ul.run/i/0d0530f043d408471579f849443179fd.avif)<!-- screenshot -->

Each project becomes a dataset named after the project, which takes the workspace's [default dataset visibility](../account/settings.md#default-visibility) and appears in your [Datasets](../data/datasets.md) list immediately with a `processing` status. Platform snapshots the project in Label Studio, downloads its images, and marks the dataset ready once its images and annotations finish importing.

Any workspace editor can run the import with the token the owner saved; without one, the panel says the workspace owner can add the Label Studio key. If some projects can't be started, the rest still import and the notification after you confirm names the first failure and its reason.

## Supported Labeling Setups

Platform reads each project's labeling config and maps it to a [YOLO task](../data/index.md#supported-tasks). The config must label one image per task:

| Label Studio labeling config                     | Platform task                                                                                                          |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `RectangleLabels`                                | [Detect](../../datasets/detect/index.md), or [OBB](../../datasets/obb/index.md) when any box in the project is rotated |
| `PolygonLabels`, alone or with `RectangleLabels` | [Segment](../../datasets/segment/index.md); only the polygons import                                                   |
| One `Choices` question                           | [Classify](../../datasets/classify/index.md)                                                                           |

Projects with anything else, such as keypoints, ellipses, brush masks, text answers, a `Choices` question beside box or polygon labels, or several labeled images per task, are left out of the preview and counted as unsupported.

- **Rotated boxes**: the preview shows a box project as Detect, because only the annotations record rotation. The import makes the whole project OBB when any box is rotated, and turns its other boxes into four-corner OBB labels.
- **Boxes and polygons in one project**: only the polygons import, so an image labeled only with boxes arrives unlabeled.
- **Which annotation imports**: each task's latest annotation that wasn't skipped. Predictions are ignored, and tasks without an annotation import as unlabeled images.
- **One label per object**: YOLO stores one class per object, so a region or classification with more than one label fails the import and names the task, rather than silently dropping labels. Multi-select `Choices` projects are listed as Classify, but no task may have more than one choice selected. Fix those tasks in Label Studio and import again.
- **Classes**: the labels in the labeling config become the class names in alphabetical order, as Label Studio's own YOLO export numbers them.
- **Splits**: Label Studio has no train and validation splits, so about 20% of the train images, or of the labeled ones for Classify, move to `val` automatically, as for [any upload without a validation split](../data/datasets.md#how-does-platform-split-my-images-into-train-val-and-test).

## Images

Images uploaded to Label Studio and images at public `http://` or `https://` URLs are imported. Some images are skipped while the rest of the project imports:

- Images in Label Studio cloud storage (`s3://`, `gs://`, or `azure-blob://` paths), which aren't supported yet
- Images that are missing, larger than 100 MB, blocked, or on another host that stays down

If Label Studio itself can't be reached, the import fails and is retried in place the next time you import the project. An image host that is down from the start can also fail the import: when 16 images are lost to connection errors, timeouts, or HTTP 5xx responses before any image has downloaded, the import stops with `Label Studio images couldn't be downloaded`, naming the host. Check that the host is online, then import the project again.

## Storage

Label Studio can't report image sizes before the images are downloaded, so the preview shows your storage usage without a size for the selection, and blocks the import only when your storage is full. Each project in one import gets an equal share of the free storage, capped by your plan's per-import size limit, and a project that outgrows its share fails with a `Label Studio project exceeds` message. Import a large project on its own to give it all the free space. Imported images count against your [storage](../account/billing.md).

## Re-running the Import

Run the import again any time to pick up projects you've added in Label Studio. Each project imports once per workspace: already-imported projects are skipped, so re-running never creates duplicates. An import is a one-time copy, and later changes in Label Studio don't sync to the dataset.

A project whose import failed is listed with the new projects, and importing it again retries it in the same dataset: images already imported stay as they are, and only missing images are added.

!!! warning "Trashed imports still count as imported"

    A dataset you moved to [Trash](../account/trash.md) still counts as imported. To import a project again from scratch, for example to pick up annotations changed in Label Studio, delete its dataset permanently first.

## Import from Self-Hosted Label Studio

The connected import reads hosted Label Studio only. For a self-hosted Label Studio, or an export you already have, use **Import an exported dataset** on the same page, because Label Studio's YOLO with Images export is a layout Platform reads, class list included:

1. **Export from Label Studio.** Open your project and click **Export**.
2. **Pick a format that includes the images.** Choose **[YOLO with Images](https://labelstud.io/guide/export)**, or **YOLOv8 OBB with Images** for rotated boxes, and click **Export**.
3. **Upload to Platform.** [Create a new dataset](../data/datasets.md) from the ZIP.
4. **Train.** [Edit the annotations](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) without leaving the workspace.

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

### Choosing an Export Format

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

### What an Export Carries Over

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

## FAQ

### Does Platform store my Label Studio token?

Yes. When the workspace owner connects, Platform saves the token in [**Settings > API Keys**](../account/api-keys.md#provider-keys-for-agents-and-annotation), and every import in the workspace runs with it. If Label Studio rejects the saved token, for example because it was revoked, the import fails until the owner connects a new one.

### Does the connected import work with self-hosted Label Studio?

No. It connects to hosted Label Studio at `app.humansignal.com`. From a self-hosted instance, [upload a YOLO with Images export](#import-from-self-hosted-label-studio).

### Which Label Studio projects can be imported?

Projects that label one image per task with `RectangleLabels`, `PolygonLabels`, or a single `Choices` question. They import as detect or OBB, segment, and classify datasets. Other projects are counted as unsupported in the preview.

### Why did my import fail with "has a region with several labels"?

A box, polygon, or classification in the named task carries more than one label, and a YOLO dataset stores one class per object. Keep one label per region or one selected choice per task in Label Studio, then run the import again; the failed project is retried in place.

### Why are some of my images missing?

Images in Label Studio cloud storage (`s3://`, `gs://`, or `azure-blob://`) aren't imported yet, and images that are missing, larger than 100 MB, blocked, or on a host that stays down are skipped. A host that is down from the start can [fail the import](#images) instead. To bring in images Platform couldn't fetch, export the project with **YOLO with Images** from a Label Studio that can reach them and [upload the ZIP](#import-from-self-hosted-label-studio) as a dataset.

### How do I pick up annotations I changed after importing?

The import is a one-time copy, so re-running it skips the project. Delete the dataset permanently, including from [Trash](../account/trash.md), and import the project again.

### Which Label Studio export format should I choose?

Choose **YOLO with Images** so the archive contains `classes.txt` and the images. **COCO with Images** also imports. The plain YOLO and COCO options write label files only, and Pascal VOC XML is not read.

### Why did my upload fail with no images found?

The plain `YOLO` or `COCO` export was used. Those archives omit the images because Label Studio normally serves them from URLs. Re-export with the **with Images** variant.

### Why are my classes named `class0`, `class1`, and so on?

The archive has no `classes.txt` or `notes.json`, usually because it was rebuilt from the `images/` and `labels/` folders alone. Upload the ZIP Label Studio exported, which ships both files.
