# XAI Utilities

Ultralytics XAI utilities provide Grad-CAM feature visualization and quantitative
faithfulness evaluation for YOLO models.

## YOLO XAI Extractor

The `YOLO_XAI_Extractor` class extracts feature activations and gradients from a
configurable YOLO model layer using PyTorch forward and backward hooks.

## Grad-CAM Heatmap

`generate_gradcam_heatmap` converts the extracted activations and gradients into
a normalized 2D Grad-CAM heatmap.

## Heatmap Faithfulness Evaluation

`validate_heatmap` evaluates an XAI heatmap using Deletion and Insertion metrics
and returns the corresponding Area Under the Deletion Curve (AUDC) and Area Under
the Insertion Curve (AUIC).

## Relationship to `class_activation_map`

The existing `class_activation_map` utility provides an inference-oriented
visualization workflow for generating class activation heatmaps during
prediction.

The XAI utilities complement this functionality by providing a YOLO-specific
Grad-CAM extraction workflow with configurable internal target-layer hooks,
targeted prediction backpropagation, and quantitative Deletion/Insertion
faithfulness evaluation.

## API Reference

::: ultralytics.utils.xai.YOLO_XAI_Extractor

::: ultralytics.utils.xai.generate_gradcam_heatmap

::: ultralytics.utils.xai.validate_heatmap
