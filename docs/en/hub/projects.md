---
comments: true
description: Optimize your model management with Ultralytics HUB Projects. Easily create, share, edit, and compare models for efficient development.
keywords: Ultralytics HUB, model management, create project, share project, edit project, delete project, compare models, reorder models, transfer models
---

# Ultralytics HUB Projects

[Ultralytics HUB](https://www.ultralytics.com/hub) projects provide an effective solution for consolidating and managing your models. If you are working with several models that perform similar tasks or have related purposes, [Ultralytics HUB](https://www.ultralytics.com/hub) projects allow you to group these models together.

This creates a unified and organized workspace that facilitates easier model management, comparison and development. Having similar models or various iterations together can facilitate rapid benchmarking, as you can compare their effectiveness. This can lead to faster, more insightful iterative development and refinement of your models.

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gc6K5eKrTNQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Train YOLOv8 Pose Model on Tiger-Pose Dataset Using Ultralytics HUB
</p>

## Create Project

Navigate to the [Projects](https://hub.ultralytics.com/projects) page by clicking on the **Projects** button in the sidebar and click on the **Create Project** button on the top right of the page.

![Ultralytics HUB screenshot of the Projects page with an arrow pointing to the Projects button in the sidebar and one to the Create Project button](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-create-project-page.avif)

??? tip

    You can create a project directly from the [Home](https://hub.ultralytics.com/home) page.

    ![Ultralytics HUB screenshot of the Home page with an arrow pointing to the Create Project card](https://github.com/ultralytics/docs/releases/download/0/hub-create-project-card.avif)

This action will trigger the **Create Project** dialog, opening up a suite of options for tailoring your project to your needs.

Type the name of your project in the _Project name_ field or keep the default name and finalize the project creation with a single click.

You have the additional option to enrich your project with a description and a unique image, enhancing its recognizability on the [Projects](https://hub.ultralytics.com/projects) page.

When you're happy with your project configuration, click **Create**.

![Ultralytics HUB screenshot of the Create Project dialog with an arrow pointing to the Create button](https://github.com/ultralytics/docs/releases/download/0/hub-create-project-dialog.avif)

After your project is created, you will be able to access it from the [Projects](https://hub.ultralytics.com/projects) page.

![Ultralytics HUB screenshot of the Projects page with an arrow pointing to one of the projects](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-projects-page.avif)

Next, [train a model](./models.md#train-model) inside your project.

![Ultralytics HUB screenshot of the Project page with an arrow pointing to the Train Model button](https://github.com/ultralytics/docs/releases/download/0/hub-train-model-button.avif)

## Share Project

!!! info

    [Ultralytics HUB](https://www.ultralytics.com/hub)'s sharing functionality provides a convenient way to share projects with others. This feature is designed to accommodate both existing [Ultralytics HUB](https://www.ultralytics.com/hub) users and those who have yet to create an account.

??? note

    You have control over the general access of your projects.

    You can choose to set the general access to "Private", in which case, only you will have access to it. Alternatively, you can set the general access to "Unlisted" which grants viewing access to anyone who has the direct link to the project, regardless of whether they have an [Ultralytics HUB](https://www.ultralytics.com/hub) account or not.

Navigate to the Project page of the project you want to share, open the project actions dropdown and click on the **Share** option. This action will trigger the **Share Project** dialog.

![Ultralytics HUB screenshot of the Project page with an arrow pointing to the Share option](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-share-project-dialog.avif)

??? tip

    You can share a project directly from the [Projects](https://hub.ultralytics.com/projects) page.

    ![Ultralytics HUB screenshot of the Projects page with an arrow pointing to the Share option of one of the projects](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-share-project-option.avif)

Set the general access to "Unlisted" and click **Save**.

![Ultralytics HUB screenshot of the Share Project dialog with an arrow pointing to the dropdown and one to the Save button](https://github.com/ultralytics/docs/releases/download/0/hub-share-project-dialog.avif)

!!! warning

    When changing the general access of a project, the general access of the models inside the project will be changed as well.

Now, anyone who has the direct link to your project can view it.

??? tip

    You can easily click on the project's link shown in the **Share Project** dialog to copy it.

    ![Ultralytics HUB screenshot of the Share Project dialog with an arrow pointing to the project's link](https://github.com/ultralytics/docs/releases/download/0/hub-share-project-dialog-arrow.avif)

## Edit Project

Navigate to the Project page of the project you want to edit, open the project actions dropdown and click on the **Edit** option. This action will trigger the **Update Project** dialog.

![Ultralytics HUB screenshot of the Project page with an arrow pointing to the Edit option](https://github.com/ultralytics/docs/releases/download/0/hub-edit-project-1.avif)

??? tip

    You can edit a project directly from the [Projects](https://hub.ultralytics.com/projects) page.

    ![Ultralytics HUB screenshot of the Projects page with an arrow pointing to the Edit option of one of the projects](https://github.com/ultralytics/docs/releases/download/0/hub-edit-project-2.avif)

Apply the desired modifications to your project and then confirm the changes by clicking **Save**.

![Ultralytics HUB screenshot of the Update Project dialog with an arrow pointing to the Save button](https://github.com/ultralytics/docs/releases/download/0/hub-edit-project-save-button.avif)

## Delete Project

Navigate to the Project page of the project you want to delete, open the project actions dropdown and click on the **Delete** option. This action will delete the project.

![Ultralytics HUB screenshot of the Project page with an arrow pointing to the Delete option](https://github.com/ultralytics/docs/releases/download/0/hub-delete-project-option.avif)

??? tip

    You can delete a project directly from the [Projects](https://hub.ultralytics.com/projects) page.

    ![Ultralytics HUB screenshot of the Projects page with an arrow pointing to the Delete option of one of the projects](https://github.com/ultralytics/docs/releases/download/0/hub-delete-project-option-1.avif)

!!! warning

    When deleting a project, the models inside the project will be deleted as well.

!!! note

    If you change your mind, you can restore the project from the [Trash](https://hub.ultralytics.com/trash) page.

    ![Ultralytics HUB screenshot of the Trash page with an arrow pointing to Trash button in the sidebar and one to the Restore option of one of the projects](https://github.com/ultralytics/docs/releases/download/0/hub-delete-project-restore-option.avif)

## Compare Models

Navigate to the Project page of the project where the models you want to compare are located. To use the model comparison feature, click on the **Charts** tab.

![Ultralytics HUB screenshot of the Project page with an arrow pointing to the Charts tab](https://github.com/ultralytics/docs/releases/download/0/hub-compare-models-1.avif)

This will display all the relevant charts. Each chart corresponds to a different metric and contains the performance of each model for that metric. The models are represented by different colors, and you can hover over each data point to get more information.

![Ultralytics HUB screenshot of the Charts tab inside the Project page](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-charts-tab.avif)

??? tip

    Each chart can be enlarged for better visualization.

    ![Ultralytics HUB screenshot of the Charts tab inside the Project page with an arrow pointing to the expand icon](https://github.com/ultralytics/docs/releases/download/0/hub-compare-models-expand-icon.avif)

    ![Ultralytics HUB screenshot of the Charts tab inside the Project page with one of the charts expanded](https://github.com/ultralytics/docs/releases/download/0/hub-compare-models-expanded-chart.avif)

    Furthermore, to properly analyze the data, you can utilize the zoom feature.

    ![Ultralytics HUB screenshot of the Charts tab inside the Project page with one of the charts expanded and zoomed](https://github.com/ultralytics/docs/releases/download/0/hub-charts-tab-expanded-zoomed.avif)

??? tip

    You have the flexibility to customize your view by selectively hiding certain models. This feature allows you to concentrate on the models of interest.

    ![Ultralytics HUB screenshot of the Charts tab inside the Project page with an arrow pointing to the hide/unhide icon of one of the model](https://github.com/ultralytics/docs/releases/download/0/hub-compare-models-hide-icon.avif)

## Reorder Models

??? note

    Ultralytics HUB's reordering functionality works only inside projects you own.

Navigate to the Project page of the project where the models you want to reorder are located. Click on the designated reorder icon of the model you want to move and drag it to the desired location.

![Ultralytics HUB screenshot of the Project page with an arrow pointing to the reorder icon](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-reorder-models.avif)

## Transfer Models

Navigate to the Project page of the project where the model you want to mode is located, open the project actions dropdown and click on the **Transfer** option. This action will trigger the **Transfer Model** dialog.

![Ultralytics HUB screenshot of the Project page with an arrow pointing to the Transfer option of one of the models](https://github.com/ultralytics/docs/releases/download/0/hub-transfer-models-1.avif)

??? tip

    You can also transfer a model directly from the [Models](https://hub.ultralytics.com/models) page.

    ![Ultralytics HUB screenshot of the Models page with an arrow pointing to the Transfer option of one of the models](https://github.com/ultralytics/docs/releases/download/0/hub-transfer-models-2.avif)

Select the project you want to transfer the model to and click **Save**.

![Ultralytics HUB screenshot of the Transfer Model dialog with an arrow pointing to the dropdown and one to the Save button](https://github.com/ultralytics/docs/releases/download/0/hub-transfer-models-dialog.avif)
