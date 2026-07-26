---
comments: true
description: Learn how to test computer vision models on unseen data, run YOLO26 validation and prediction on your test set, and catch overfitting and data leakage.
keywords: model testing, test computer vision model, model testing vs model evaluation, overfitting, underfitting, data leakage, validation mode, prediction mode, YOLO26, Ultralytics
---

# How to Test Computer Vision Models

## Introduction

Model testing checks how a [trained model](./model-training-tips.md) performs on previously unseen, real-world data — moving, poorly lit, or partially hidden objects rather than a curated benchmark. While [model evaluation](./model-evaluation-insights.md) measures metrics on a labeled dataset, testing verifies that the model's learned behavior matches the [goals of your application](./defining-project-goals.md) before deployment. This guide covers preparing test data, testing Ultralytics YOLO26 models, and catching [overfitting](https://www.ultralytics.com/glossary/overfitting), underfitting, and data leakage.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/SyyCUvxw9BM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Test Machine Learning Models | Avoid Data Leakage in Computer Vision 🚀
</p>

## Model Testing vs. Model Evaluation

Model testing and model evaluation are two distinct [steps in a computer vision project](./steps-of-a-cv-project.md). Evaluation measures performance with metrics on a labeled dataset; testing checks whether the model's learned behavior holds up in conditions that resemble deployment.

Suppose you have trained a [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) model to recognize cats and dogs, and you want to deploy this model at a pet store to monitor the animals. During the model evaluation phase, you use a labeled dataset to calculate metrics like [accuracy](https://www.ultralytics.com/glossary/accuracy), [precision](https://www.ultralytics.com/glossary/precision), and [recall](https://www.ultralytics.com/glossary/recall). For instance, the model might be 98% accurate in distinguishing between cats and dogs in a given dataset.

After evaluation, you test the model using images from a pet store to see how well it identifies cats and dogs in more varied and realistic conditions. You check if it can correctly label cats and dogs when they are moving, in different lighting conditions, or partially obscured by objects like toys or furniture. Model testing checks that the model behaves as expected outside the controlled evaluation environment.

## Preparing for Model Testing

Computer vision [datasets](./preprocessing-annotated-data.md) are usually divided into training and testing sets to simulate real-world conditions: [training data](https://www.ultralytics.com/glossary/training-data) teaches the model, while testing data verifies its behavior on examples it has never seen. The [Ultralytics Platform](https://platform.ultralytics.com) keeps dataset organization and annotation in one place, which helps when building a labeled test set.

!!! tip "Before you test"

    - **Realistic representation:** The previously unseen testing data should be similar to the data the model will handle when deployed. This gives a realistic picture of the model's capabilities.
    - **Sufficient size:** The testing dataset needs to be large enough to provide reliable insights into how well the model performs.

## How to Test a YOLO26 Model

Testing a trained YOLO26 model involves two complementary workflows: validating on a labeled test split to get quantitative metrics, and predicting on new images to inspect behavior qualitatively.

### Validate on a Labeled Test Split

[Validation mode](../modes/val.md) compares the model's predictions against ground-truth labels and reports precision, recall, mAP50, and mAP50-95 for detection models. It also saves visual aids like a confusion matrix and a precision-recall curve, which help you spot specific areas where the model might not be performing well.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model or your own trained checkpoint, e.g. "path/to/best.pt"
        model = YOLO("yolo26n.pt")

        # Validate; add split="test" if your dataset YAML defines a test split
        metrics = model.val(data="coco8.yaml")
        print(metrics.box.map)  # mAP50-95
        ```

    === "CLI"

        ```bash
        yolo val model=yolo26n.pt data=coco8.yaml
        ```

By default, validation runs on the dataset's `val` split. To measure performance on a held-out test set, define a `test:` split in your dataset YAML and pass `split="test"`.

### Predict on New Images

[Prediction mode](../modes/predict.md) runs the model on new, unseen data without requiring labels. It does not produce performance metrics, but saving the annotated outputs lets you review how the model behaves on real-world images — for example, an entire folder of test images in one go.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model or your own trained checkpoint, e.g. "path/to/best.pt"
        model = YOLO("yolo26n.pt")

        # Run predictions on a folder of test images and save annotated results
        results = model.predict(source="path/to/test_images", save=True)
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.pt source=path/to/test_images save=True
        ```

!!! tip "Testing a pretrained model before custom training"

    To check whether YOLO26 suits your application before investing in custom training, run prediction mode with a pretrained checkpoint on your own images. The models are pretrained on datasets like [COCO](../datasets/detect/coco.md), so the results give a quick sense of how well the model might perform in your specific context.

### Validation vs. Prediction Mode

| Mode                              | Purpose                                       | Requires labels | Output                                                          |
| --------------------------------- | --------------------------------------------- | --------------- | --------------------------------------------------------------- |
| [Validation](../modes/val.md)     | Quantify performance against ground truth     | Yes             | Precision, recall, mAP50, mAP50-95, confusion matrix, PR curves |
| [Prediction](../modes/predict.md) | Inspect model behavior on new, unlabeled data | No              | Annotated images and prediction results, no metrics             |

## How to Analyze Test Results

Once predictions and metrics are in hand, dig into where and why the model fails:

- **Misclassified images:** Identify and review images that the model misclassified to understand where it is going wrong.
- **Error analysis:** Perform a thorough error analysis to understand the types of errors (e.g., false positives vs. false negatives) and their potential causes.
- **Bias and fairness:** Check for any biases in the model's predictions. Ensure that the model performs equally well across different subsets of the data, especially if it includes sensitive attributes like race, gender, or age.

## Overfitting and Underfitting in Machine Learning

When testing a [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) model, especially in computer vision, it's important to watch out for overfitting and [underfitting](https://www.ultralytics.com/glossary/underfitting). These issues can significantly affect how well your model works with new data.

| Issue            | Common signs                                                                                                         | How to address it                                                                                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Overfitting**  | High training accuracy but low validation accuracy; oversensitivity to minor changes or irrelevant details in images | Apply [regularization](https://www.ultralytics.com/glossary/regularization) such as dropout, increase the size of the training dataset, simplify the model architecture |
| **Underfitting** | Low accuracy even on the training set; consistent failure to recognize obvious features or objects                   | Use a more complex model, provide more relevant features, increase training [epochs](https://www.ultralytics.com/glossary/epoch)                                        |

The key is to find a balance so the model performs well on both training and validation datasets. Regularly monitoring metrics and visually inspecting predictions during testing helps you catch a drift toward either extreme.

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/overfitting-underfitting-appropriate-fitting.avif" alt="Comparison of underfitting, appropriate fitting, and overfitting on the same dataset">
</p>

## Data Leakage in Computer Vision and How to Avoid It

Data leakage happens when information from outside the training dataset accidentally gets used to train the model. The model may seem very accurate during training, but it won't perform well on new, unseen data when data leakage occurs.

Leakage can be tricky to spot and often comes from hidden biases in the training data:

| Bias type                 | What it looks like                                                                                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Camera bias**           | Different angles, lighting, shadows, and camera movements introduce unwanted patterns                                                                      |
| **Overlay bias**          | Logos, timestamps, or other overlays in images mislead the model                                                                                           |
| **Font and object bias**  | Specific fonts or objects that frequently appear in certain classes skew the model's learning                                                              |
| **Spatial bias**          | Imbalances in foreground-background, [bounding box](https://www.ultralytics.com/glossary/bounding-box) distributions, and object locations affect training |
| **Label and domain bias** | Incorrect labels or shifts in data types lead to leakage                                                                                                   |

### How to Detect and Avoid Data Leakage

To find data leakage, check whether the model's results are surprisingly good, look at whether one feature is much more important than others, double-check that the model's decisions make sense intuitively, and verify that data was divided correctly before any processing.

To prevent it, use a diverse dataset with images or videos from different cameras and environments, and carefully review your data for hidden biases — such as all positive samples being taken at a specific time of day. Avoiding data leakage makes your computer vision models more reliable in real-world situations.

## What Comes After Model Testing

After testing your model, the next steps depend on the results. If your model performs well, you can deploy it into a real-world environment. If the results aren't satisfactory, you'll need to make improvements. This might involve analyzing errors, [gathering more data](./data-collection-and-annotation.md), improving data quality, [adjusting hyperparameters](./hyperparameter-tuning.md), and retraining the model.

## Conclusion

Rigorous model testing — validating on a held-out test split, predicting on real-world images, and checking for overfitting and data leakage — is what turns a well-evaluated model into a dependable one. Address the issues testing surfaces before deployment, and your model is far more likely to perform as intended in production. If questions come up along the way, ask the community on the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) or the [Ultralytics Discord server](https://discord.com/invite/ultralytics).

## FAQ

### What are the key differences between model evaluation and model testing in computer vision?

Model evaluation measures performance with metrics on a labeled dataset, while model testing checks how the model behaves on new, unseen data that resembles deployment conditions. Evaluation produces numbers like precision and mAP from a controlled dataset; testing reveals whether the learned behavior holds up with varied lighting, motion, or occlusion. See [Model Testing vs. Model Evaluation](#model-testing-vs-model-evaluation) for a worked example.

### How can I test my Ultralytics YOLO26 model on multiple images?

Use [prediction mode](../modes/predict.md) and pass a folder path as the `source` — YOLO26 runs on every image in the folder and can save the annotated results for review. Prediction mode does not compute metrics; to quantify performance on a labeled set, use [validation mode](../modes/val.md) instead. Both workflows are shown in [How to Test a YOLO26 Model](#how-to-test-a-yolo26-model).

### What metrics does YOLO26 validation report on a test set?

For detection models, validation reports precision, recall, mAP50, and mAP50-95, and saves plots including a confusion matrix and a precision-recall curve. To validate on a dedicated test split rather than the default `val` split, define `test:` in your dataset YAML and pass `split="test"`. See the [performance metrics guide](./yolo-performance-metrics.md) for how to interpret each metric.

### What should I do if my computer vision model shows signs of overfitting or underfitting?

For overfitting, apply regularization techniques such as dropout, increase the size of the training dataset, or simplify the model architecture. For underfitting, use a more complex model, provide more relevant features, or train for more epochs. The signs of each issue and the corresponding fixes are summarized in [Overfitting and Underfitting in Machine Learning](#overfitting-and-underfitting-in-machine-learning).

### How can I detect and avoid data leakage in computer vision?

Suspect data leakage when test performance looks surprisingly good, a single feature dominates predictions, or the model's decisions don't make intuitive sense. Prevent it by using diverse datasets from different cameras and environments, reviewing data for hidden biases, and verifying that the train/test split happened before any processing. See [Data Leakage in Computer Vision](#data-leakage-in-computer-vision-and-how-to-avoid-it) for the common bias types.

### What steps should I take after testing my computer vision model?

If the results meet your project goals, deploy the model; if not, improve it before deployment. That can mean analyzing errors, [collecting more diverse data](./data-collection-and-annotation.md), improving data quality, [tuning hyperparameters](./hyperparameter-tuning.md), and retraining. Repeat testing after each round of changes to confirm the fixes worked.
