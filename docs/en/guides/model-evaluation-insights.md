---
title: YOLO Model Evaluation and Fine-Tuning
comments: true
description: Learn how to evaluate YOLO26 models with metrics like mAP and IoU, then fine-tune them in Python to boost detection accuracy on your own dataset.
keywords: model evaluation, fine-tuning YOLO, mAP, IoU, confidence score, model validation, YOLO26 metrics, fine-tune YOLO26, improve detection accuracy
---

# Insights on Model Evaluation and Fine-Tuning

After [training](./model-training-tips.md) a YOLO model, the next step is to measure how well it performs and fine-tune it to close the gaps. Evaluation uses metrics like [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and [IoU](https://www.ultralytics.com/glossary/intersection-over-union-iou) to quantify accuracy, while fine-tuning adjusts training parameters to strengthen weak spots so the model meets your [project's objective](./defining-project-goals.md). This guide explains the key evaluation metrics, how to read them, and the fine-tuning techniques that elevate your model's capabilities.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/-aYO-6VaDrw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Insights into Model Evaluation and Fine-Tuning | Tips for Improving Mean Average Precision
</p>

Evaluation and fine-tuning sit near the end of the [computer vision project workflow](./steps-of-a-cv-project.md), once training is underway and you need to verify the model is accurate, efficient, and ready for deployment.

## Key Evaluation Metrics

Various metrics measure how effectively a model performs. These [performance metrics](./yolo-performance-metrics.md) provide clear, numerical insights that guide improvements toward making sure the model meets its intended goals.

### Confidence Score

The confidence score represents the model's certainty that a detected object belongs to a particular class. It ranges from 0 to 1, with higher scores indicating greater confidence. The confidence score helps filter predictions; only detections with confidence scores above a specified threshold are considered valid.

!!! tip "Seeing no predictions?"

    When running inference, if you aren't seeing any predictions and you've checked everything else, try lowering the confidence threshold. Sometimes the threshold is too high, causing the model to ignore valid predictions. Lowering it lets the model consider more possibilities. This might not meet your final project goals, but it's a good way to see what the model can do and decide how to fine-tune it.

### Intersection over Union

[Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) is a metric in [object detection](https://www.ultralytics.com/glossary/object-detection) that measures how well the predicted [bounding box](https://www.ultralytics.com/glossary/bounding-box) overlaps with the ground truth bounding box. IoU values range from 0 to 1, where one stands for a perfect match. IoU is essential because it measures how closely the predicted boundaries match the actual object boundaries.

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/intersection-over-union-overview.avif" alt="Intersection over Union Overview">
</p>

### Mean Average Precision

[Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) measures how well an object detection model performs overall. It looks at the precision of detecting each object class, averages these scores, and gives a single number that shows how accurately the model can identify and classify objects.

Two mAP metrics are most commonly reported:

- *mAP@.5:* Measures the average precision at a single IoU threshold of 0.5. This metric checks if the model can correctly find objects with a looser accuracy requirement. It focuses on whether the object is roughly in the right place, not needing perfect placement, and helps see if the model is generally good at spotting objects.
- *mAP@.5:.95:* Averages the mAP values calculated at multiple IoU thresholds, from 0.5 to 0.95 in 0.05 increments. This metric is more detailed and strict. It gives a fuller picture of how accurately the model can find objects at different levels of strictness and is especially useful for applications that need precise object detection.

Other mAP metrics include mAP@0.75, which uses a stricter IoU threshold of 0.75, and mAP@small, medium, and large, which evaluate precision across objects of different sizes.

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mean-average-precision-overview.avif" alt="Mean average precision mAP metric">
</p>

## Evaluating a YOLO26 Model

You can evaluate a trained YOLO26 model with the [validation mode](../modes/val.md). For a deeper look at how each metric is computed and interpreted, see the [YOLO26 performance metrics](./yolo-performance-metrics.md) guide.

### Handling Variable Image Sizes

Evaluating your model on images of different sizes helps you understand its performance on diverse datasets. The `rect=true` validation parameter groups images by aspect ratio and pads each batch to the smallest shape that fits, so rectangular images are evaluated without being forced into a square.

The `imgsz` parameter sets the image size used during validation (640 by default, applied as a square). With `rect=true`, YOLO26 constrains the longer side to `imgsz` and pads the shorter side to a stride multiple, preserving the aspect ratio. Adjust `imgsz` based on your dataset's dimensions and the GPU memory available.

### Accessing YOLO26 Metrics

To understand your model's performance in detail, you can access specific evaluation metrics with a few lines of Python. The snippet below loads a model, runs validation, and prints the most useful metrics.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")

        # Run validation on your dataset
        results = model.val(data="coco8.yaml")

        # Overall metrics
        print("mAP50-95:", results.box.map)  # mAP at IoU 0.50:0.95
        print("mAP50:", results.box.map50)  # mAP at IoU 0.50
        print("mAP75:", results.box.map75)  # mAP at IoU 0.75
        print("Mean precision:", results.box.mp)
        print("Mean recall:", results.box.mr)
        print("Fitness:", results.box.fitness())  # weighted score used for model selection

        # Per-class metrics
        print("Class indices evaluated:", results.box.ap_class_index)
        print("Per-class mAP50-95:", results.box.maps)

        # Per-image precision, recall, F1, TP, FP, and FN
        print("Per-image metrics:", results.box.image_metrics)

        # Per-stage timing breakdown in milliseconds per image
        print("Timing breakdown (ms/image):", results.speed)
        ```

Note that `fitness()` is a method and must be called with parentheses, while metrics like `map`, `map50`, and `mp` are properties accessed directly.

The `results.box.image_metrics` attribute is a per-image dictionary keyed by image filename, holding `precision`, `recall`, `f1`, `tp`, `fp`, and `fn` for each image. Preprocessing, inference, and postprocessing times are reported separately in the `results.speed` dictionary. Together, these let you pinpoint which images the model struggles with and fine-tune accordingly.

## Fine-Tuning Your Model

Fine-tuning takes a pretrained model and adjusts its parameters to improve performance on a specific task or dataset. Also known as model retraining, it lets the model better understand and predict outcomes for the data it will encounter in real-world applications. Based on your evaluation results, you retrain the model to achieve optimal results by paying close attention to a few key parameters and techniques.

### Starting With a Higher Learning Rate

During normal training, the [learning rate](https://www.ultralytics.com/glossary/learning-rate) starts low and gradually increases over the first few epochs to stabilize early updates. When fine-tuning, the model already carries useful features from pretraining, so you can skip this warmup and start adapting to your new data right away.

Set the `warmup_epochs` training argument to `0` in `model.train()` to disable the warmup phase. Training then continues from the pretrained weights at the configured base learning rate (`lr0`) instead of ramping up to it, adjusting to the nuances of your new data.

!!! example "Fine-tune without learning-rate warmup"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n.pt")

        # Fine-tune with the warmup phase disabled
        model.train(data="coco8.yaml", epochs=10, warmup_epochs=0)
        ```

    === "CLI"

        ```bash
        yolo detect train model=yolo26n.pt data=coco8.yaml epochs=10 warmup_epochs=0
        ```

### Image Tiling for Small Objects

Image tiling can improve detection accuracy for small objects. By dividing larger images into smaller segments, such as splitting 1280x1280 images into multiple 640x640 segments, you preserve the original resolution and let the model learn from high-resolution fragments. Ultralytics supports this at inference time through [SAHI tiled inference](./sahi-tiled-inference.md). When training on tiled images, make sure to adjust your labels for each new segment correctly.

## Conclusion

Evaluating and fine-tuning are what turn a trained model into a dependable, deployable one: metrics like mAP and IoU expose weaknesses, and targeted parameter changes address them. Start with the [validation mode](../modes/val.md) to benchmark your model, then apply the fine-tuning techniques above and keep iterating with new parameters, techniques, and datasets. If questions come up along the way, ask the community on the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/issues) or the [Ultralytics Discord server](https://discord.com/invite/ultralytics).

## FAQ

### What are the key metrics for evaluating YOLO26 model performance?

To evaluate YOLO26 model performance, important metrics include Confidence Score, Intersection over Union (IoU), and Mean Average Precision (mAP). Confidence Score measures the model's certainty for each detected object class. IoU evaluates how well the predicted bounding box overlaps with the ground truth. Mean Average Precision (mAP) aggregates precision scores across classes, with mAP@.5 and mAP@.5:.95 being two common types for varying IoU thresholds. Learn more about these metrics in our [YOLO26 performance metrics guide](./yolo-performance-metrics.md).

### How can I fine-tune a pretrained YOLO26 model for my specific dataset?

Fine-tuning a pretrained YOLO26 model involves adjusting its parameters to improve performance on a specific task or dataset. Start by evaluating your model with metrics, then set the `warmup_epochs` training argument to `0` in `model.train()` so the learning rate starts at the configured base value immediately instead of ramping up. During evaluation, parameters like `rect=true` help handle varied image sizes effectively. For more detailed guidance, refer to our section on [fine-tuning your model](#fine-tuning-your-model).

### How can I handle variable image sizes when evaluating my YOLO26 model?

To handle variable image sizes during evaluation, use the `rect=true` parameter in YOLO26, which groups images by aspect ratio and pads each batch instead of forcing every image to a square. The `imgsz` parameter sets the image size for validation, defaulting to 640. Adjust `imgsz` to suit your dataset and GPU memory. For more details, visit our [section on handling variable image sizes](#handling-variable-image-sizes).

### What practical steps can I take to improve mean average precision for my YOLO26 model?

Improving mean average precision (mAP) for a YOLO26 model involves several steps:

1. **Tuning Hyperparameters**: Experiment with different learning rates, [batch sizes](https://www.ultralytics.com/glossary/batch-size), and image augmentations.
2. **[Data Augmentation](https://www.ultralytics.com/glossary/data-augmentation)**: Use techniques like Mosaic and MixUp to create diverse training samples.
3. **Image Tiling**: Split larger images into smaller tiles to improve detection accuracy for small objects.

Refer to our detailed section on [fine-tuning your model](#fine-tuning-your-model) for specific strategies.

### How do I access YOLO26 model evaluation metrics in Python?

You can access YOLO26 model evaluation metrics using Python after running validation:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")

        # Run validation
        results = model.val(data="coco8.yaml")

        # Access key metrics
        print("Mean average precision at IoU=0.50:", results.box.map50)
        print("Mean average precision at IoU=0.50:0.95:", results.box.map)
        print("Mean recall:", results.box.mr)
        print("Class indices evaluated:", results.box.ap_class_index)
        ```

Analyzing these metrics helps you fine-tune and optimize your YOLO26 model. For a deeper dive, check out our guide on [YOLO performance metrics](./yolo-performance-metrics.md).
