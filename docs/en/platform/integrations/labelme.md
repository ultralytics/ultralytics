---
plans: [free, pro, enterprise]
comments: true
description: Convert LabelMe JSON annotations to YOLO format, upload the dataset to Ultralytics Platform, and train a computer vision model.
keywords: Ultralytics Platform, LabelMe, LabelMe to YOLO, LabelMe JSON to YOLO, labelmetk, YOLO export, dataset import, offline annotation, computer vision
title: LabelMe to YOLO Dataset Export - Ultralytics Platform
---

# Export LabelMe Annotations to YOLO and Ultralytics Platform

[LabelMe](https://labelme.io/) is an offline image annotation tool with an
[open-source Python application](https://github.com/wkentaro/labelme). There is no live LabelMe connection or API key
to configure in [Ultralytics Platform](https://platform.ultralytics.com). The complete integration is a local workflow:
annotate in LabelMe, convert the LabelMe JSON annotations to YOLO format with the LabelMe Toolkit, and upload the
resulting ZIP as a Platform dataset.

## 1. Annotate the Images in LabelMe

Install LabelMe using the [desktop app](https://labelme.io/download) or the
[open-source Python package](https://labelme.io/docs/install-labelme-terminal), then open the directory
containing your images. LabelMe saves each image's annotations in a matching JSON file.

For the YOLO detection workflow in this guide, draw rectangles around each object and assign a class name. The
[LabelMe starter guide](https://labelme.io/docs/starter-guide) covers opening images, drawing shapes, and saving
annotations, while the [LabelMe dataset guide](https://labelme.io/docs/dataset-guide) covers reviewing and preparing a
complete annotated dataset.

Your source directory should contain the images and LabelMe JSON files:

```text
your_dataset/
├── image_001.jpg
├── image_001.json
├── image_002.jpg
└── image_002.json
```

## 2. Install the LabelMe Toolkit

Install the LabelMe Toolkit by following LabelMe's
[toolkit installation guide](https://labelme.io/docs/install-toolkit). The toolkit and its exports run locally.

!!! info "LabelMe Pro is required for the export"

    `export-to-yolo` is part of the LabelMe Pro Toolkit, and downloading its installer requires a LabelMe sign-in.
    This is a LabelMe product requirement; the resulting ZIP can be uploaded on any Platform plan.

Verify the installation, then list every label found in the source dataset:

```bash
labelmetk --version
labelmetk list-labels your_dataset/
```

Review the output before exporting. Labels omitted from `--class-names` are skipped, and the order you provide becomes
the YOLO class ID order.

## 3. Export to YOLO Format

Run [`export-to-yolo`](https://labelme.io/docs/export-to-yolo) with the source directory and a comma-separated list of
class names:

```bash
labelmetk export-to-yolo your_dataset/ --class-names crack,normal
```

Replace `crack,normal` with the labels returned by `list-labels`. LabelMe writes the result to
`your_dataset.export/`:

```text
your_dataset.export/
├── classes.txt
├── images/
│   ├── image_001.jpg
│   └── image_002.jpg
└── labels/
    ├── image_001.txt
    └── image_002.txt
```

`classes.txt` preserves the class names in the same order used by the YOLO label files. Keep it at the root of the
export.

!!! note "This workflow creates a detection dataset"

    LabelMe Toolkit exports rectangles as YOLO bounding boxes. It also reduces polygons and masks to their
    axis-aligned bounding boxes, so `export-to-yolo` does not preserve segmentation geometry. Draw rectangles when
    preparing a detection dataset with this workflow.

## 4. Create the ZIP Archive

Compress the **contents** of `your_dataset.export/`, not the directory around them. On macOS or Linux:

```bash
cd your_dataset.export
zip -r ../your_dataset.zip classes.txt images labels
```

On Windows PowerShell:

```powershell
Compress-Archive -Path .\your_dataset.export\* -DestinationPath .\your_dataset.zip
```

Open the ZIP before uploading and confirm that `classes.txt`, `images/`, and `labels/` are at its root. An extra
`your_dataset.export/` wrapper prevents Platform from finding the root class list.

## 5. Upload to Ultralytics Platform

1. Open [**Settings > Integrations > LabelMe**](https://platform.ultralytics.com/settings?tab=integrations&integration=labelme).
2. Click **Upload export**.
3. Upload `your_dataset.zip` and finish creating the dataset.
4. Wait for processing to complete, then review the images, classes, and annotations.
5. [Edit the annotations](../data/annotation.md), [train a model](../train/index.md), and
   [deploy it](../deploy/index.md) from the same workspace.

LabelMe and the YOLO export remain entirely offline. Only the ZIP file you select in the upload dialog is sent to
Platform.

## Troubleshooting

- **Classes are named `class0`, `class1`, and so on:** confirm that `classes.txt` is present at the root of the ZIP.
- **Some annotations are missing:** run `labelmetk list-labels your_dataset/` again and include every required label in
  `--class-names`.
- **Polygons became boxes:** this is the documented behavior of LabelMe's `export-to-yolo`; it converts non-rectangle
  shapes to bounding boxes.
- **The dataset has no images:** confirm that the exported `images/` directory is included in the ZIP.
- **Platform cannot find the classes:** remove any outer directory from the archive so `classes.txt`, `images/`, and
  `labels/` are the top-level entries.

For the complete local workflow, see LabelMe's
[YOLO training guide](https://labelme.io/blog/yolo-training-with-labelme) and
[`export-to-yolo` reference](https://labelme.io/docs/export-to-yolo).
