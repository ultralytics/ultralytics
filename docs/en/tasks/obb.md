---
comments: true
description: Discover how to detect objects with rotation for higher precision using YOLO26 OBB models. Learn, train, validate, and export OBB models effortlessly.
keywords: Oriented Bounding Boxes, OBB, Object Detection, YOLO26, Ultralytics, DOTAv1, Model Training, Model Export, AI, Machine Learning
model_name: yolo26n-obb
---

# Oriented Bounding Boxes [Object Detection](https://www.ultralytics.com/glossary/object-detection)

<!-- obb task poster -->

Oriented object detection goes a step further than standard object detection by introducing an extra angle to locate objects more accurately in an image.

The output of an oriented object detector is a set of rotated bounding boxes that precisely enclose the objects in the image, along with class labels and confidence scores for each box. Oriented bounding boxes are particularly useful when objects appear at various angles, such as in aerial imagery, where traditional axis-aligned bounding boxes may include unnecessary background.

<!-- youtube video link for obb task -->

!!! tip

    YOLO26 OBB models use the `-obb` suffix, i.e., `yolo26n-obb.pt`, and are pretrained on [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/128JhhR2DlM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Detect & Track Objects with Ultralytics YOLO26 Oriented Bounding Boxes (OBB) | Ship Tracking 🚢
</p>

## Visual Samples

|                                               Ships Detection using OBB                                               |                                                Vehicle Detection using OBB                                                |
| :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
| ![Ships Detection using OBB](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ships-detection-using-obb.avif) | ![Vehicle Detection using OBB](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vehicle-detection-using-obb.avif) |

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 pretrained OBB models are shown here, which are pretrained on the [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                          | size<br><sup>(pixels)</sup> | mAP<sup>test<br>50-95(e2e)</sup> | mAP<sup>test<br>50(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------------------------------------------------------------------------ | --------------------------- | -------------------------------- | ----------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-obb) | 1024                        | 52.4                             | 78.9                          | 97.7 ± 0.9                           | 2.8 ± 0.0                                 | 2.5                      | 14.0                    |
| [YOLO26s-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-obb) | 1024                        | 54.8                             | 80.9                          | 218.0 ± 1.4                          | 4.9 ± 0.1                                 | 9.8                      | 55.1                    |
| [YOLO26m-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-obb) | 1024                        | 55.3                             | 81.0                          | 579.2 ± 3.8                          | 10.2 ± 0.3                                | 21.2                     | 183.3                   |
| [YOLO26l-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-obb) | 1024                        | 56.2                             | 81.6                          | 735.6 ± 3.1                          | 13.0 ± 0.2                                | 25.6                     | 230.0                   |
| [YOLO26x-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-obb) | 1024                        | 56.7                             | 81.7                          | 1485.7 ± 11.5                        | 30.5 ± 0.9                                | 57.6                     | 516.5                   |

- **mAP<sup>test</sup>** values are for single-model multiscale on [DOTAv1](https://captain-whu.github.io/DOTA/index.html) dataset. <br>Reproduce by `yolo val obb data=DOTAv1.yaml device=0 split=test` and submit merged results to [DOTA evaluation](https://captain-whu.github.io/DOTA/evaluation.html).
- **Speed** averaged over DOTAv1 val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu`
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers and, for end2end models, removes the auxiliary one-to-many detection head. Pretrained checkpoints retain the full training architecture and may show higher counts.

## Train

Train YOLO26n-obb on the DOTA8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! note

    An OBB and its 180° rotation are identical, so rotation is defined modulo 180° and the box has no direction. Training labels are canonicalized to a long-edge form: the box width `w` is the longer side, the angle is measured clockwise from the positive x-axis to the orientation of the `w` edge, and it is stored in radians in **`[-π/4, 3π/4)`** (`[-45°, 135°)`). Predictions are not canonicalized, so a predicted box may report the short edge as `w`; see [Results Output](#results-output). The `[0°, 90°)` form is the regularized DOTA-style convention and is not applied at training or inference.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-obb.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-obb.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-obb.yaml").load("yolo26n-obb.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo obb train data=dota8.yaml model=yolo26n-obb.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo obb train data=dota8.yaml model=yolo26n-obb.yaml pretrained=yolo26n-obb.pt epochs=100 imgsz=640
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uZ7SymQfqKI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO-OBB (Oriented Bounding Boxes) Models on DOTA Dataset using Ultralytics Platform
</p>

### Dataset format

OBB dataset format can be found in detail in the [Dataset Guide](../datasets/obb/index.md). The YOLO OBB format designates bounding boxes by their four corner points with coordinates normalized between 0 and 1, following this structure. [Ultralytics Platform annotation](../platform/data/annotation.md) supports OBB labels with a dedicated oriented bounding box drawing tool:

```text
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

Internally, YOLO processes losses and outputs in the `xywhr` format, which represents the [bounding box](https://www.ultralytics.com/glossary/bounding-box)'s center point (xy), width, height, and rotation.

## Val

Validate trained YOLO26n-obb model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the DOTA8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-obb.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val(data="dota8.yaml")  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # a list containing mAP50-95(B) for each category
        metrics.box.image_metrics  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN
        ```

    === "CLI"

        ```bash
        yolo obb val model=yolo26n-obb.pt data=dota8.yaml         # val official model
        yolo obb val model=path/to/best.pt data=path/to/data.yaml # val custom model
        ```

## Predict

Use a trained YOLO26n-obb model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-obb.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/boats.jpg")  # predict on an image

        # Access the results
        for result in results:
            xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
            xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
            names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
            confs = result.obb.conf  # confidence score of each box
        ```

    === "CLI"

        ```bash
        yolo obb predict model=yolo26n-obb.pt source='https://ultralytics.com/images/boats.jpg'  # predict with official model
        yolo obb predict model=path/to/best.pt source='https://ultralytics.com/images/boats.jpg' # predict with custom model
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5XYdm5CYODA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Detect and Track Storage Tanks using Ultralytics YOLO-OBB | Oriented Bounding Boxes | DOTA
</p>

See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Results Output

Oriented bounding box detection returns one `Results` object per image. The primary prediction field is `result.obb`,
which contains rotated boxes, class IDs, and confidence scores for each detected object.

| Attribute             | Type            | Shape     | Description                              |
| --------------------- | --------------- | --------- | ---------------------------------------- |
| `result.obb`          | `OBB`           | `(N)`     | Oriented boxes.                          |
| `result.obb.data`     | `torch.float32` | `(N,7/8)` | Raw rotated boxes with confidence/class. |
| `result.obb.xywhr`    | `torch.float32` | `(N,5)`   | `xywhr` rotated boxes.                   |
| `result.obb.xyxyxyxy` | `torch.float32` | `(N,4,2)` | Four corner points.                      |
| `result.obb.conf`     | `torch.float32` | `(N,)`    | Confidence scores.                       |

Labels are canonicalized to the long-edge convention but predictions are not, so `result.obb.xywhr` may report `w` smaller than `h` with the rotation measured from the short edge. Both forms describe the same rectangle, so `result.obb.xyxyxyxy` is correct either way. Convert the corners back when downstream code needs the long-edge form, keeping in mind that rewriting the box parameters may cyclically shift the corner order:

```python
from ultralytics.utils.ops import xyxyxyxy2xywhr

xywhr = xyxyxyxy2xywhr(result.obb.xyxyxyxy)  # w >= h, rotation in [-pi/4, 3pi/4)
```

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

## Export

Export a YOLO26n-obb model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-obb.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-obb.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

Available YOLO26-obb export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-obb.onnx`. Usage examples are shown for your model after export completes.

| Format | `format` Argument | Model | Metadata | Arguments |
| ---------------------------------------------------------- | ----------------- | ------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| [PyTorch](https://pytorch.org/) | - | `yolo26n-obb.pt` | ✅ | - |
| [TorchScript](../integrations/torchscript.md) | `torchscript` | `yolo26n-obb.torchscript` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [ONNX](../integrations/onnx.md) | `onnx` | `yolo26n-obb.onnx` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [OpenVINO](../integrations/openvino.md) | `openvino` | `yolo26n-obb_openvino_model/` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TensorRT](../integrations/tensorrt.md) | `engine` | `yolo26n-obb.engine` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `workspace`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [CoreML](../integrations/coreml.md) | `coreml` | `yolo26n-obb.mlpackage` | ✅ | `imgsz`, `dynamic`, `quantize`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model` | `yolo26n-obb_saved_model/` | ✅ | `imgsz`, `keras`, `quantize`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TF GraphDef](../integrations/tf-graphdef.md) | `pb` | `yolo26n-obb.pb` | ❌ | `imgsz`, `opset`, `batch`, `device` |
| [TF Edge TPU](../integrations/edge-tpu.md) | `edgetpu` | `yolo26n-obb_edgetpu.tflite` | ✅ | `imgsz`, `quantize`, `opset`, `data`, `fraction`, `device` |
| [PaddlePaddle](../integrations/paddlepaddle.md) | `paddle` | `yolo26n-obb_paddle_model/` | ✅ | `imgsz`, `batch`, `device` |
| [MNN](../integrations/mnn.md) | `mnn` | `yolo26n-obb.mnn` | ✅ | `imgsz`, `batch`, `dynamic`, `quantize`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [NCNN](../integrations/ncnn.md) | `ncnn` | `yolo26n-obb_ncnn_model/` | ✅ | `imgsz`, `quantize`, `batch`, `device` |
| [IMX500](../integrations/sony-imx500.md):material-information-outline:{ title="IMX format is currently only supported for YOLOv8n, YOLO11n models" } | `imx` | `yolo26n-obb_imx_model/` | ✅ | `imgsz`, `quantize`, `data`, `fraction`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [RKNN](../integrations/rockchip-rknn.md) | `rknn` | `yolo26n-obb_rknn_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [ExecuTorch](../integrations/executorch.md) | `executorch` | `yolo26n-obb_executorch_model/` | ✅ | `imgsz`, `batch`, `device` |
| [Axelera](../integrations/axelera.md) | `axelera` | `yolo26n-obb_axelera_model/` | ✅ | `imgsz`, `batch`, `quantize`, `data`, `fraction`, `device` |
| [DEEPX](../integrations/deepx.md) | `deepx` | `yolo26n-obb_deepx_model/` | ✅ | `imgsz`, `quantize`, `simplify`, `opset`, `data`, `optimize`, `device` |
| [Qualcomm QNN](../integrations/qnn.md) | `qnn` | `yolo26n-obb_qnn.onnx` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [LiteRT](../integrations/litert.md) | `litert` | `yolo26n-obb.tflite` | ✅ | `imgsz`, `quantize`, `batch`, `data`, `fraction`, `device` |
| [Hailo](../integrations/hailo.md) | `hailo` | `yolo26n-obb_hailo_model/` | ✅ | `imgsz`, `name`, `quantize`, `data`, `fraction`, `simplify`, `conf`, `iou` |
| [Huawei Ascend](../integrations/ascend.md) | `ascend` | `yolo26n-obb_ascend_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `opset`, `simplify`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" } |

See full `export` details in the [Export](../modes/export.md) page.

## Real-World Applications

OBB detection with YOLO26 has numerous practical applications across various industries:

- **Maritime and Port Management**: Detecting ships and vessels at various angles for [fleet management](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-obb-object-detection) and monitoring.
- **Urban Planning**: Analyzing buildings and infrastructure from aerial imagery.
- **Agriculture**: Monitoring crops and agricultural equipment from drone footage.
- **Energy Sector**: Inspecting solar panels and wind turbines at different orientations.
- **Transportation**: Tracking vehicles on roads and in parking lots from various perspectives.

These applications benefit from OBB's ability to precisely fit objects at any angle, providing more accurate detection than traditional bounding boxes.

## FAQ

### What are Oriented Bounding Boxes (OBB) and how do they differ from regular bounding boxes?

Oriented Bounding Boxes (OBB) include an additional angle to enhance object localization accuracy in images. Unlike regular bounding boxes, which are axis-aligned rectangles, OBBs can rotate to fit the orientation of the object better. This is particularly useful for applications requiring precise object placement, such as aerial or satellite imagery ([Dataset Guide](../datasets/obb/index.md)).

### How do I train a YOLO26n-obb model using a custom dataset?

To train a YOLO26n-obb model with a custom dataset, follow the example below using Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-obb.pt")

        # Train the model
        results = model.train(data="path/to/custom_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo obb train data=path/to/custom_dataset.yaml model=yolo26n-obb.pt epochs=100 imgsz=640
        ```

For more training arguments, check the [Configuration](../usage/cfg.md) section.

### What datasets can I use for training YOLO26-OBB models?

YOLO26-OBB models are pretrained on datasets like [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) but you can use any dataset formatted for OBB. Detailed information on OBB dataset formats can be found in the [Dataset Guide](../datasets/obb/index.md).

### How can I export a YOLO26-OBB model to ONNX format?

Exporting a YOLO26-OBB model to ONNX format is straightforward using either Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-obb.pt")

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-obb.pt format=onnx
        ```

For more export formats and details, refer to the [Export](../modes/export.md) page.

### How do I validate the accuracy of a YOLO26n-obb model?

To validate a YOLO26n-obb model, you can use Python or CLI commands as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-obb.pt")

        # Validate the model
        metrics = model.val(data="dota8.yaml")
        ```

    === "CLI"

        ```bash
        yolo obb val model=yolo26n-obb.pt data=dota8.yaml
        ```

See full validation details in the [Val](../modes/val.md) section.
