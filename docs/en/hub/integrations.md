---
comments: true
description: Explore seamless integrations between Ultralytics HUB and platforms like Roboflow. Learn how to import datasets, train models, and more.
keywords: Ultralytics HUB, Roboflow integration, dataset import, model training, AI, machine learning
---

# Ultralytics HUB Integrations

Learn about [Ultralytics HUB](https://ultralytics.com/hub) integrations with various platforms and formats.

## Datasets

Seamlessly import your datasets in [Ultralytics HUB](https://ultralytics.com/hub) for [model training](./models.md#train-model).

After a dataset is imported in [Ultralytics HUB](https://ultralytics.com/hub), you can [train a model](./models.md#train-model) on your dataset just like you would using the [Ultralytics HUB](https://ultralytics.com/hub) datasets.

### Roboflow

You can easily filter the [Roboflow](https://roboflow.com/?ref=ultralytics) datasets on the [Ultralytics HUB](https://ultralytics.com/hub) [Datasets](https://hub.ultralytics.com/datasets) page.

![Ultralytics HUB screenshot of the Datasets page with Roboflow provider filter](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_1.jpg)

[Ultralytics HUB](https://ultralytics.com/hub) supports two types of integrations with [Roboflow](https://roboflow.com/?ref=ultralytics), [Universe](#universe) and [Workspace](#workspace).

#### Universe

The [Roboflow](https://roboflow.com/?ref=ultralytics) Universe integration allows you to import one dataset at a time into [Ultralytics HUB](https://ultralytics.com/hub) from [Roboflow](https://roboflow.com/?ref=ultralytics).

##### Import

When you export a [Roboflow](https://roboflow.com/?ref=ultralytics) dataset, select the [Ultralytics HUB](https://ultralytics.com/hub) format. This action will redirect you to [Ultralytics HUB](https://ultralytics.com/hub) and trigger the **Dataset Import** dialog.

You can import your [Roboflow](https://roboflow.com/?ref=ultralytics) dataset by clicking on the **Import** button.

![Ultralytics HUB screenshot of the Dataset Import dialog with an arrow pointing to the Import button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_universe_import_1.jpg)

Next, [train a model](./models.md#train-model) on your dataset.

![Ultralytics HUB screenshot of the Dataset page of a Roboflow Universe dataset with an arrow pointing to the Train Model button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_universe_import_2.jpg)

##### Remove

Navigate to the Dataset page of the [Roboflow](https://roboflow.com/?ref=ultralytics) dataset you want to remove, open the dataset actions dropdown and click on the **Remove** option.

![Ultralytics HUB screenshot of the Dataset page of a Roboflow Universe dataset with an arrow pointing to the Remove option](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_universe_remove_1.jpg)

??? tip "Tip"

    You can remove an imported [Roboflow](https://roboflow.com/?ref=ultralytics) dataset directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Remove option of one of the Roboflow Universe datasets](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_remove_1.jpg)

#### Workspace

The [Roboflow](https://roboflow.com/?ref=ultralytics) Workspace integration allows you to import an entire [Roboflow](https://roboflow.com/?ref=ultralytics) Workspace at once into [Ultralytics HUB](https://ultralytics.com/hub).

##### Import

Navigate to the [Integrations](https://hub.ultralytics.com/settings?tab=integrations) page by clicking on the **Integrations** button in the sidebar.

Type your [Roboflow](https://roboflow.com/?ref=ultralytics) Workspace private API key and click on the **Add** button.

??? tip "Tip"

    You can click on the **Get my API key** button which will redirect you to the settings of your [Roboflow](https://roboflow.com/?ref=ultralytics) Workspace from where you can obtain your private API key.

![Ultralytics HUB screenshot of the Integrations page with an arrow pointing to the Integrations button in the sidebar and one to the Add button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_workspace_import_1.jpg)

This will connect your [Ultralytics HUB](https://ultralytics.com/hub) account with your [Roboflow](https://roboflow.com/?ref=ultralytics) Workspace and make your [Roboflow](https://roboflow.com/?ref=ultralytics) datasets available in [Ultralytics HUB](https://ultralytics.com/hub).

![Ultralytics HUB screenshot of the Integrations page with an arrow pointing to one of the connected workspaces](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_workspace_import_2.jpg)

Next, [train a model](./models.md#train-model) on your dataset.

![Ultralytics HUB screenshot of the Dataset page of a Roboflow Workspace dataset with an arrow pointing to the Train Model button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_workspace_import_3.jpg)

##### Remove

Navigate to the [Integrations](https://hub.ultralytics.com/settings?tab=integrations) page by clicking on the **Integrations** button in the sidebar and click on the **Unlink** button of the [Roboflow](https://roboflow.com/?ref=ultralytics) Workspace you want to remove.

![Ultralytics HUB screenshot of the Integrations page  with an arrow pointing to the Integrations button in the sidebar and one to the Unlink button of one of the connected workspaces](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_workspace_remove_1.jpg)

??? tip "Tip"

    You can remove a connected [Roboflow](https://roboflow.com/?ref=ultralytics) Workspace directly from the Dataset page of one of the datasets from your [Roboflow](https://roboflow.com/?ref=ultralytics) Workspace.

    ![Ultralytics HUB screenshot of the Dataset page of a Roboflow Workspace dataset with an arrow pointing to the remove option](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_workspace_remove_2.jpg)

??? tip "Tip"

    You can remove a connected [Roboflow](https://roboflow.com/?ref=ultralytics) Workspace directly from the [Datasets](https://hub.ultralytics.com/datasets) page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Remove option of one of the Roboflow Workspace datasets](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/integrations/hub_roboflow_remove_1.jpg)

## Models

### Exports

After you [train a model](./models.md#train-model), you can [export it](./models.md#deploy-model) to 13 different formats, including ONNX, OpenVINO, CoreML, TensorFlow, Paddle and many others.

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with an arrow pointing to the Export card and all formats exported](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_deploy_model_1.jpg)

The available export formats are presented in the table below.

{% include "macros/export-table.md" %}

## Exciting New Features on the Way 🎉

- Additional Dataset Integrations
- Detailed Export Integration Guides
- Step-by-Step Tutorials for Each Integration

## Stay Updated 🚧

This integrations page is your first stop for upcoming developments. Keep an eye out with our:

- **Newsletter:** Subscribe [here](https://ultralytics.com/#newsletter) for the latest news.
- **Social Media:** Follow us [here](https://www.linkedin.com/company/ultralytics) for updates and teasers.
- **Blog:** Visit our [blog](https://ultralytics.com/blog) for detailed insights.

## We Value Your Input 🗣️

Your feedback shapes our future releases. Share your thoughts and suggestions [here](https://ultralytics.com/survey).

## Thank You, Community! 🌍

Your [contributions](https://docs.ultralytics.com/help/contributing) inspire our continuous [innovation](https://github.com/ultralytics/ultralytics). Stay tuned for the big reveal of what's next in AI and ML at Ultralytics!
