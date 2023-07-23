---
comments: true
description: Learn how Ultralytics HUB datasets streamline your ML workflow. Upload, format, validate, access, share, edit or delete datasets for Ultralytics YOLO model training.
keywords: Ultralytics, HUB datasets, YOLO model training, upload datasets, dataset validation, ML workflow, share datasets
---

# HUB Datasets

Ultralytics HUB datasets are a practical solution for managing and leveraging your custom datasets.

Once uploaded, datasets can be immediately utilized for model training. This integrated approach facilitates a seamless transition from dataset management to model training, significantly simplifying the entire process.

## Upload Dataset

Ultralytics HUB datasets are just like YOLOv5 and YOLOv8 🚀 datasets. They use the same structure and the same label formats to keep
everything simple.

Before you upload a dataset to Ultralytics HUB, make sure to **place your dataset YAML file inside the dataset root directory** and that **your dataset YAML, directory and ZIP have the same name**, as shown in the example below, and then zip the dataset directory.

For example, if your dataset is called "coco8", as our [COCO8](https://docs.ultralytics.com/datasets/detect/coco8) example dataset, then you should have a `coco8.yaml` inside your `coco8/` directory, which will create a `coco8.zip` when zipped:

```bash
zip -r coco8.zip coco8
```

You can download our [COCO8](https://github.com/ultralytics/hub/blob/master/example_datasets/coco8.zip) example dataset and unzip it to see exactly how to structure your dataset.

<p align="center">
  <img  src="https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_1.jpg" alt="COCO8 Dataset Structure" width="80%" />
</p>

The dataset YAML is the same standard YOLOv5 and YOLOv8 YAML format.

!!! example "coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

After zipping your dataset, you should validate it before uploading it to Ultralytics HUB. Ultralytics HUB conducts the dataset validation check post-upload, so by ensuring your dataset is correctly formatted and error-free ahead of time, you can forestall any setbacks due to dataset rejection.

```py
from ultralytics.hub import check_dataset
check_dataset('path/to/coco8.zip')
```

Once your dataset ZIP is ready, navigate to the [Datasets](https://hub.ultralytics.com/datasets) page by clicking on the **Datasets** button in the sidebar.

![Ultralytics HUB screenshot of the Home page with an arrow pointing to the Datasets button in the sidebar](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_2.jpg)

??? tip "Tip"

    You can also upload a dataset directly from the [Home](https://hub.ultralytics.com/home) page.

    ![Ultralytics HUB screenshot of the Home page with an arrow pointing to the Upload Dataset card](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_3.jpg)

Click on the **Upload Dataset** button on the top right of the page. This action will trigger the **Upload Dataset** dialog.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Upload Dataset button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_4.jpg)

Upload your dataset in the _Dataset .zip file_ field.

You have the additional option to set a custom name and description for your Ultralytics HUB dataset.

When you're happy with your dataset configuration, click **Upload**.

![Ultralytics HUB screenshot of the Upload Dataset dialog with an arrow pointing to the Upload button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_5.jpg)

After your dataset is uploaded and processed, you will be able to access it from the Datasets page.

![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to one of the datasets](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_6.jpg)

You can view the images in your dataset grouped by splits (Train, Validation, Test).

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Images tab](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_7.jpg)

??? tip "Tip"

    Each image can be enlarged for better visualization.

    ![Ultralytics HUB screenshot of the Images tab inside the Dataset page with an arrow pointing to the expand icon](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_8.jpg)

    ![Ultralytics HUB screenshot of the Images tab inside the Dataset page with one of the images expanded](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_9.jpg)

Also, you can analyze your dataset by click on the **Overview** tab.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Overview tab](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_10.jpg)

Next, [train a model](https://docs.ultralytics.com/hub/models/#train-model) on your dataset.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Train Model button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_upload_dataset_11.jpg)

## Share Dataset

!!! info "Info"

    Ultralytics HUB's sharing functionality provides a convenient way to share datasets with others. This feature is designed to accommodate both existing Ultralytics HUB users and those who have yet to create an account.

??? note "Note"

    You have control over the general access of your datasets.

    You can choose to set the general access to "Private", in which case, only you will have access to it. Alternatively, you can set the general access to "Unlisted" which grants viewing access to anyone who has the direct link to the dataset, regardless of whether they have an Ultralytics HUB account or not.

Navigate to the Dataset page of the dataset you want to share, open the dataset actions dropdown and click on the **Share** option. This action will trigger the **Share Dataset** dialog.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Share option](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_share_dataset_1.jpg)

??? tip "Tip"

    You can also share a dataset directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Share option of one of the datasets](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_share_dataset_2.jpg)

Set the general access to "Unlisted" and click **Save**.

![Ultralytics HUB screenshot of the Share Dataset dialog with an arrow pointing to the dropdown and one to the Save button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_share_dataset_3.jpg)

Now, anyone who has the direct link to your dataset can view it.

??? tip "Tip"

    You can easily click on the dataset's link shown in the **Share Dataset** dialog to copy it.

    ![Ultralytics HUB screenshot of the Share Dataset dialog with an arrow pointing to the dataset's link](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_share_dataset_4.jpg)

## Edit Dataset

Navigate to the Dataset page of the dataset you want to edit, open the dataset actions dropdown and click on the **Edit** option. This action will trigger the **Update Dataset** dialog.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Edit option](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_edit_dataset_1.jpg)

??? tip "Tip"

    You can also edit a dataset directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Edit option of one of the datasets](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_edit_dataset_2.jpg)

Apply the desired modifications to your dataset and then confirm the changes by clicking **Save**.

![Ultralytics HUB screenshot of the Update Dataset dialog with an arrow pointing to the Save button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_edit_dataset_3.jpg)

## Delete Dataset

Navigate to the Dataset page of the dataset you want to delete, open the dataset actions dropdown and click on the **Delete** option. This action will delete the dataset.

![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Delete option](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_delete_dataset_1.jpg)

??? tip "Tip"

    You can also delete a dataset directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Delete option of one of the datasets](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_delete_dataset_2.jpg)

??? note "Note"

    If you change your mind, you can restore the dataset from the [Trash](https://hub.ultralytics.com/trash) page.

    ![Ultralytics HUB screenshot of the Trash page with an arrow pointing to the Restore option of one of the datasets](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/datasets/hub_delete_dataset_3.jpg)
