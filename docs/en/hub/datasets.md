---
comments: true
description: Learn how Ultralytics HUB datasets streamline your ML workflow. Upload, format, validate, access, share, edit or delete datasets for Ultralytics YOLO model training.
keywords: Ultralytics, HUB datasets, YOLO model training, upload datasets, dataset validation, ML workflow, share datasets
---

# HUB Datasets

[Ultralytics HUB](https://hub.ultralytics.com/) datasets are a practical solution for managing and leveraging your custom datasets.

Once uploaded, datasets can be immediately utilized for model training. This integrated approach facilitates a seamless transition from dataset management to model training, significantly simplifying the entire process.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/R42s2zFtNIY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Watch: Upload Datasets to Ultralytics HUB | Complete Walkthrough of Dataset Upload Feature
</p>

## Upload Dataset

Ultralytics HUB datasets are just like YOLOv5 and YOLOv8 🚀 datasets. They use the same structure and the same label formats to keep everything simple.

Before you upload a dataset to Ultralytics HUB, make sure to **place your dataset YAML file inside the dataset root directory** and that **your dataset YAML, directory and ZIP have the same name**, as shown in the example below, and then zip the dataset directory.

For example, if your dataset is called "coco8", as our [COCO8](https://docs.ultralytics.com/datasets/detect/coco8) example dataset, then you should have a `coco8.yaml` inside your `coco8/` directory, which will create a `coco8.zip` when zipped:

```bash
zip -r coco8.zip coco8
```

You can download our [COCO8](https://github.com/ultralytics/hub/blob/main/example_datasets/coco8.zip) example dataset and unzip it to see exactly how to structure your dataset.

<p align="center">
  <img  src="https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_1.jpg" alt="COCO8 Dataset Structure" width="80%">
</p>

The dataset YAML is the same standard YOLOv5 and YOLOv8 YAML format.

!!! Example "coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

After zipping your dataset, you should validate it before uploading it to Ultralytics HUB. Ultralytics HUB conducts the dataset validation check post-upload, so by ensuring your dataset is correctly formatted and error-free ahead of time, you can forestall any setbacks due to dataset rejection.

```py
from ultralytics.hub import check_dataset

check_dataset('path/to/coco8.zip')
```

Once your dataset ZIP is ready, navigate to the [Datasets](https://hub.ultralytics.com/datasets) page by clicking on the **Datasets** button in the sidebar.

![Ultralytics HUB screenshot of the Home page with an arrow pointing to the Datasets button in the sidebar](https://github.com/ultralytics/ultralytics/assets/19519529/2d9f774c-100d-4ff4-a82b-2a38ced33c21)

Click on the **Upload Dataset** button on the top right of the page. This action will trigger the **Upload Dataset** dialog.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Upload Dataset button](https://github.com/ultralytics/ultralytics/assets/19519529/52ac10f5-ce42-483a-ac02-1d37d2cba3de)

Upload your dataset in the _Dataset .zip file_ field.

You have the additional option to set a custom name and description for your Ultralytics HUB dataset.

When you're happy with your dataset configuration, click **Upload**.

![Ultralytics HUB screenshot of the Upload Dataset dialog with an arrow pointing to the Upload button](https://github.com/ultralytics/ultralytics/assets/19519529/7d210ff6-bdb2-4535-a661-0470274bd7d6)

After your dataset is uploaded and processed, you will be able to access it from the Datasets page.

![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to one of the datasets](https://github.com/ultralytics/ultralytics/assets/19519529/a05d9b66-f8ba-4474-b8ac-ebe0dd143831)

You can view the images in your dataset grouped by splits (Train, Validation, Test).

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Images tab](https://github.com/ultralytics/ultralytics/assets/19519529/e07468e3-6284-4334-9783-84bfb11130f8)

!!! tip "Tip"

    Each image can be enlarged for better visualization.

    ![Ultralytics HUB screenshot of the Images tab inside the Dataset page with an arrow pointing to the expand icon](https://github.com/ultralytics/ultralytics/assets/19519529/26f411a0-5153-4805-a8c1-cbd379708e28)

    ![Ultralytics HUB screenshot of the Images tab inside the Dataset page with one of the images expanded](https://github.com/ultralytics/ultralytics/assets/19519529/7d5e0d50-85e5-4014-9f5b-464284e5b291)

Also, you can analyze your dataset by click on the **Overview** tab.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Overview tab](https://github.com/ultralytics/ultralytics/assets/19519529/5eaacd5d-fedf-4332-9091-1418c9f333cb)

Next, [train a model](https://docs.ultralytics.com/hub/models/#train-model) on your dataset.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Train Model button](https://github.com/ultralytics/ultralytics/assets/19519529/cb709e5f-a10b-478f-a81d-a48f61c193fe)

## Share Dataset

!!! Info "Info"

    Ultralytics HUB's sharing functionality provides a convenient way to share datasets with others. This feature is designed to accommodate both existing Ultralytics HUB users and those who have yet to create an account.

!!! note "Note"

    You have control over the general access of your datasets.

    You can choose to set the general access to "Private", in which case, only you will have access to it. Alternatively, you can set the general access to "Unlisted" which grants viewing access to anyone who has the direct link to the dataset, regardless of whether they have an Ultralytics HUB account or not.

Navigate to the Dataset page of the dataset you want to share, open the dataset actions dropdown and click on the **Share** option. This action will trigger the **Share Dataset** dialog.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Share option](https://github.com/ultralytics/ultralytics/assets/19519529/9a0e61e7-2838-42b3-8abe-a22980e6c680)

!!! tip "Tip"

    You can also share a dataset directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Share option of one of the datasets](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_share_dataset_2.jpg)

Set the general access to "Unlisted" and click **Save**.

![Ultralytics HUB screenshot of the Share Dataset dialog with an arrow pointing to the dropdown and one to the Save button](https://github.com/ultralytics/ultralytics/assets/19519529/5818b928-19a3-48a8-892d-27ac1dc684dd)

Now, anyone who has the direct link to your dataset can view it.

!!! tip "Tip"

    You can easily click on the dataset's link shown in the **Share Dataset** dialog to copy it.

    ![Ultralytics HUB screenshot of the Share Dataset dialog with an arrow pointing to the dataset's link](https://github.com/ultralytics/ultralytics/assets/19519529/8ede7d20-2a68-411d-9de5-3175b5ba7038)

## Edit / Delete Dataset

Navigate to the Dataset page of the dataset you want to edit, open the dataset actions dropdown and click on the **Edit** option. This action will trigger the **Update Dataset** dialog.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Edit and Delete option](https://github.com/ultralytics/ultralytics/assets/19519529/6c248c8c-29cd-4bd5-b33d-43e90aa1d000)

Apply the desired modifications to your dataset and then confirm the changes by clicking **Save**.

Navigate to the Dataset page of the dataset you want to delete, open the dataset actions dropdown and click on the **Delete** option. This action will delete the dataset.

!!! note "Note"

    If you change your mind, you can restore the dataset from the [Trash](https://hub.ultralytics.com/trash) page.

    ![Ultralytics HUB screenshot of the Trash page with an arrow pointing to the Restore option of one of the datasets](https://github.com/ultralytics/ultralytics/assets/19519529/56a9460c-0e06-4659-989d-715211b9d7ce)
