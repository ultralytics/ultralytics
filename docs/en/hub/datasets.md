---
comments: true
description: Effortlessly manage, upload, and share your custom datasets on Ultralytics HUB for seamless model training integration. Simplify your workflow today!.
keywords: Ultralytics HUB, datasets, custom datasets, dataset management, model training, upload datasets, share datasets, dataset workflow
---

# Ultralytics HUB Datasets

[Ultralytics HUB](https://www.ultralytics.com/hub) datasets are a practical solution for managing and leveraging your custom datasets.

Once uploaded, datasets can be immediately utilized for model training. This integrated approach facilitates a seamless transition from dataset management to model training, significantly simplifying the entire process.

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/R42s2zFtNIY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Watch: Upload Datasets to Ultralytics HUB | Complete Walkthrough of Dataset Upload Feature
</p>

## Upload Dataset

[Ultralytics HUB](https://www.ultralytics.com/hub) datasets are just like YOLOv5 and YOLOv8 🚀 datasets. They use the same structure and the same label formats to keep everything simple.

Before you upload a dataset to [Ultralytics HUB](https://www.ultralytics.com/hub), make sure to **place your dataset YAML file inside the dataset root directory** and that **your dataset YAML, directory and ZIP have the same name**, as shown in the example below, and then zip the dataset directory.

For example, if your dataset is called "coco8", as our [COCO8](https://docs.ultralytics.com/datasets/detect/coco8/) example dataset, then you should have a `coco8.yaml` inside your `coco8/` directory, which will create a `coco8.zip` when zipped:

```bash
zip -r coco8.zip coco8
```

You can download our [COCO8](https://github.com/ultralytics/hub/blob/main/example_datasets/coco8.zip) example dataset and unzip it to see exactly how to structure your dataset.

<p align="center">
  <img  src="https://github.com/ultralytics/docs/releases/download/0/coco8-dataset-structure.avif" alt="COCO8 Dataset Structure" width="80%">
</p>

The dataset YAML is the same standard YOLOv5 and YOLOv8 YAML format.

!!! example "coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

After zipping your dataset, you should [validate it](https://docs.ultralytics.com/reference/hub/__init__/#ultralytics.hub.check_dataset) before uploading it to [Ultralytics HUB](https://www.ultralytics.com/hub). [Ultralytics HUB](https://www.ultralytics.com/hub) conducts the dataset validation check post-upload, so by ensuring your dataset is correctly formatted and error-free ahead of time, you can forestall any setbacks due to dataset rejection.

```py
from ultralytics.hub import check_dataset

check_dataset("path/to/dataset.zip", task="detect")
```

Once your dataset ZIP is ready, navigate to the [Datasets](https://hub.ultralytics.com/datasets) page by clicking on the **Datasets** button in the sidebar and click on the **Upload Dataset** button on the top right of the page.

![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Datasets button in the sidebar and one to the Upload Dataset button](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-datasets-upload.avif)

??? tip

    You can upload a dataset directly from the [Home](https://hub.ultralytics.com/home) page.

    ![Ultralytics HUB screenshot of the Home page with an arrow pointing to the Upload Dataset card](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-upload-dataset-card.avif)

This action will trigger the **Upload Dataset** dialog.

Select the dataset task of your dataset and upload it in the _Dataset .zip file_ field.

You have the additional option to set a custom name and description for your [Ultralytics HUB](https://www.ultralytics.com/hub) dataset.

When you're happy with your dataset configuration, click **Upload**.

![Ultralytics HUB screenshot of the Upload Dataset dialog with arrows pointing to dataset task, dataset file and Upload button](https://github.com/ultralytics/docs/releases/download/0/hub-upload-dataset-dialog.avif)

After your dataset is uploaded and processed, you will be able to access it from the [Datasets](https://hub.ultralytics.com/datasets) page.

![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to one of the datasets](https://github.com/ultralytics/docs/releases/download/0/hub-datasets-page.avif)

You can view the images in your dataset grouped by splits (Train, Validation, Test).

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Images tab](https://github.com/ultralytics/docs/releases/download/0/hub-dataset-page-images-tab.avif)

??? tip

    Each image can be enlarged for better visualization.

    ![Ultralytics HUB screenshot of the Images tab inside the Dataset page with an arrow pointing to the expand icon](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-images-tab-expand-icon.avif)

    ![Ultralytics HUB screenshot of the Images tab inside the Dataset page with one of the images expanded](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dataset-page-expanded-image.avif)

Also, you can analyze your dataset by click on the **Overview** tab.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Overview tab](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-overview-tab.avif)

Next, [train a model](./models.md#train-model) on your dataset.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Train Model button](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dataset-page-train-model-button.avif)

## Download Dataset

Navigate to the Dataset page of the dataset you want to download, open the dataset actions dropdown and click on the **Download** option. This action will start downloading your dataset.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Download option](https://github.com/ultralytics/docs/releases/download/0/hub-download-dataset-1.avif)

??? tip

    You can download a dataset directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Download option of one of the datasets](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-datasets-download-option.avif)

## Share Dataset

!!! info

    [Ultralytics HUB](https://www.ultralytics.com/hub)'s sharing functionality provides a convenient way to share datasets with others. This feature is designed to accommodate both existing [Ultralytics HUB](https://www.ultralytics.com/hub) users and those who have yet to create an account.

!!! note

    You have control over the general access of your datasets.

    You can choose to set the general access to "Private", in which case, only you will have access to it. Alternatively, you can set the general access to "Unlisted" which grants viewing access to anyone who has the direct link to the dataset, regardless of whether they have an [Ultralytics HUB](https://www.ultralytics.com/hub) account or not.

Navigate to the Dataset page of the dataset you want to share, open the dataset actions dropdown and click on the **Share** option. This action will trigger the **Share Dataset** dialog.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Share option](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-share-dataset.avif)

??? tip

    You can share a dataset directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Share option of one of the datasets](https://github.com/ultralytics/docs/releases/download/0/hub-share-dataset-2.avif)

Set the general access to "Unlisted" and click **Save**.

![Ultralytics HUB screenshot of the Share Dataset dialog with an arrow pointing to the dropdown and one to the Save button](https://github.com/ultralytics/docs/releases/download/0/hub-share-dataset-dialog.avif)

Now, anyone who has the direct link to your dataset can view it.

??? tip

    You can easily click on the dataset's link shown in the **Share Dataset** dialog to copy it.

    ![Ultralytics HUB screenshot of the Share Dataset dialog with an arrow pointing to the dataset's link](https://github.com/ultralytics/docs/releases/download/0/hub-share-dataset-link.avif)

## Edit Dataset

Navigate to the Dataset page of the dataset you want to edit, open the dataset actions dropdown and click on the **Edit** option. This action will trigger the **Update Dataset** dialog.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Edit option](https://github.com/ultralytics/docs/releases/download/0/hub-edit-dataset-1.avif)

??? tip

    You can edit a dataset directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Edit option of one of the datasets](https://github.com/ultralytics/docs/releases/download/0/hub-edit-dataset-page.avif)

Apply the desired modifications to your dataset and then confirm the changes by clicking **Save**.

![Ultralytics HUB screenshot of the Update Dataset dialog with an arrow pointing to the Save button](https://github.com/ultralytics/docs/releases/download/0/hub-edit-dataset-save-button.avif)

## Delete Dataset

Navigate to the Dataset page of the dataset you want to delete, open the dataset actions dropdown and click on the **Delete** option. This action will delete the dataset.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Delete option](https://github.com/ultralytics/docs/releases/download/0/hub-delete-dataset-option.avif)

??? tip

    You can delete a dataset directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Delete option of one of the datasets](https://github.com/ultralytics/docs/releases/download/0/hub-delete-dataset-page.avif)

!!! note

    If you change your mind, you can restore the dataset from the [Trash](https://hub.ultralytics.com/trash) page.

    ![Ultralytics HUB screenshot of the Trash page with an arrow pointing to Trash button in the sidebar and one to the Restore option of one of the datasets](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-trash-restore.avif)
