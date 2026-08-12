---
comments: true
description: Master instance segmentation using YOLO26. Learn how to detect, segment and outline objects in images with detailed guides and examples.
keywords: instance segmentation, YOLO26, object detection, image segmentation, machine learning, deep learning, computer vision, COCO dataset, Ultralytics
model_name: yolo26n-seg
---

# Instance Segmentation with Ultralytics YOLO {#instance-segmentation}

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/instance-segmentation-examples.avif" alt="Ultralytics YOLO instance segmentation examples">

[Instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) goes a step further than object detection and involves identifying individual objects in an image and segmenting them from the rest of the image.

The output of an instance segmentation model is a set of masks or contours that outline each object in the image, along with class labels and confidence scores for each object. Instance segmentation is useful when you need to know not only where objects are in an image, but also what their exact shape is.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Run Segmentation with Pretrained Ultralytics YOLO Model in Python.
</p>

!!! tip

    YOLO26 Segment models use the `-seg` suffix, i.e., `yolo26n-seg.pt`, and are pretrained on [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 pretrained Segment models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, [Semantic](semantic.md) models are pretrained on [Cityscapes](../datasets/semantic/cityscapes.md), and Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                          | size<br><sup>(pixels)</sup> | mAP<sup>box<br>50-95(e2e)</sup> | mAP<sup>mask<br>50-95(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------------------------------------------------------------------------ | --------------------------- | ------------------------------- | -------------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLO26n-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-seg) | 640                         | 39.6                            | 33.9                             | 53.3 ± 0.5                           | 2.1 ± 0.0                                 | 2.7                      | 9.1                     |
| [YOLO26s-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-seg) | 640                         | 47.3                            | 40.0                             | 118.4 ± 0.9                          | 3.3 ± 0.0                                 | 10.4                     | 34.2                    |
| [YOLO26m-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-seg) | 640                         | 52.5                            | 44.1                             | 328.2 ± 2.4                          | 6.7 ± 0.1                                 | 23.6                     | 121.5                   |
| [YOLO26l-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-seg) | 640                         | 54.4                            | 45.5                             | 387.0 ± 3.7                          | 8.0 ± 0.1                                 | 28.0                     | 139.8                   |
| [YOLO26x-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-seg) | 640                         | 56.5                            | 47.0                             | 787.0 ± 6.8                          | 16.4 ± 0.1                                | 62.8                     | 313.5                   |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val segment data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val segment data=coco.yaml batch=1 device=0|cpu`
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers and, for end2end models, removes the auxiliary one-to-many detection head. Pretrained checkpoints retain the full training architecture and may show higher counts.

## Train

Train YOLO26n-seg on the COCO8-seg dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-seg.yaml").load("yolo26n-seg.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.yaml pretrained=yolo26n-seg.pt epochs=100 imgsz=640
        ```

See full `train` mode details in the [Train](../modes/train.md) page. Segmentation models can also be trained with [Ultralytics Platform cloud training](../platform/train/cloud-training.md).

### Dataset format

YOLO segmentation dataset format can be found in detail in the [Dataset Guide](../datasets/segment/index.md). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics. You can also create segmentation masks with [Ultralytics Platform annotation](../platform/data/annotation.md) using polygon tools and SAM-powered smart annotation.

## Val

Validate trained YOLO26n-seg model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8-seg dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # a list containing mAP50-95(B) for each category
        metrics.box.image_metrics  # per-image metrics dictionary for det with precision, recall, F1, TP, FP, and FN
        metrics.seg.map  # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps  # a list containing mAP50-95(M) for each category
        metrics.seg.image_metrics  # per-image metrics dictionary for seg with precision, recall, F1, TP, FP, and FN
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolo26n-seg.pt  # val official model
        yolo segment val model=path/to/best.pt # val custom model
        ```

## Predict

Use a trained YOLO26n-seg model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            xy = result.masks.xy  # mask polygons in pixel coordinates
            xyn = result.masks.xyn  # normalized mask polygons
            masks = result.masks.data  # binary masks, shape (N,H,W), dtype torch.uint8
        ```

    === "CLI"

        ```bash
        yolo segment predict model=yolo26n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Results Output

YOLO instance segmentation returns one `Results` object per image. Each result stores object-level predictions, where
each detected instance has its own binary mask, class, confidence, and box.

| Attribute           | Type            | Shape         | Description                         |
| ------------------- | --------------- | ------------- | ----------------------------------- |
| `result.masks`      | `Masks`         | `(N)`         | Instance masks.                     |
| `result.masks.data` | `torch.uint8`   | `(N,H,W)`     | Binary masks, values `0` or `1`.    |
| `result.masks.xy`   | `np.float32`    | `list[(P,2)]` | Pixel polygons.                     |
| `result.masks.xyn`  | `np.float32`    | `list[(P,2)]` | Normalized polygons.                |
| `result.boxes`      | `Boxes`         | `(N)`         | Instance boxes/classes/confidences. |
| `result.boxes.cls`  | `torch.float32` | `(N,)`        | Class IDs; cast to `int` for names. |

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

### How This Differs from Semantic Segmentation

Instance segmentation is object-level segmentation: two cars produce two masks, two boxes, and two confidence scores.
[Semantic segmentation](semantic.md) is pixel-level classification: those same cars become pixels with the same class ID
in one image-sized class map, with no per-object boxes, confidences, or default polygon list.

## Export

Export a YOLO26n-seg model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-seg.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

Available YOLO26-seg export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-seg.onnx`. Usage examples are shown for your model after export completes.

!!! note

    CoreML embedded NMS pipelines (`nms=True`) only support object detection models. Segmentation exports to CoreML warn and force `nms=False`, producing a raw model without NMS.

| Format | `format` Argument | Model | Metadata | Arguments |
| ---------------------------------------------------------- | ----------------- | ------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| [PyTorch](https://pytorch.org/) | - | `yolo26n-seg.pt` | ✅ | - |
| [TorchScript](../integrations/torchscript.md) | `torchscript` | `yolo26n-seg.torchscript` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [ONNX](../integrations/onnx.md) | `onnx` | `yolo26n-seg.onnx` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [OpenVINO](../integrations/openvino.md) | `openvino` | `yolo26n-seg_openvino_model/` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TensorRT](../integrations/tensorrt.md) | `engine` | `yolo26n-seg.engine` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `workspace`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [CoreML](../integrations/coreml.md) | `coreml` | `yolo26n-seg.mlpackage` | ✅ | `imgsz`, `dynamic`, `quantize`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model` | `yolo26n-seg_saved_model/` | ✅ | `imgsz`, `keras`, `quantize`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TF GraphDef](../integrations/tf-graphdef.md) | `pb` | `yolo26n-seg.pb` | ❌ | `imgsz`, `opset`, `batch`, `device` |
| [TF Edge TPU](../integrations/edge-tpu.md) | `edgetpu` | `yolo26n-seg_edgetpu.tflite` | ✅ | `imgsz`, `quantize`, `opset`, `data`, `fraction`, `device` |
| [PaddlePaddle](../integrations/paddlepaddle.md) | `paddle` | `yolo26n-seg_paddle_model/` | ✅ | `imgsz`, `batch`, `device` |
| [MNN](../integrations/mnn.md) | `mnn` | `yolo26n-seg.mnn` | ✅ | `imgsz`, `batch`, `dynamic`, `quantize`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [NCNN](../integrations/ncnn.md) | `ncnn` | `yolo26n-seg_ncnn_model/` | ✅ | `imgsz`, `quantize`, `batch`, `device` |
| [IMX500](../integrations/sony-imx500.md):material-information-outline:{ title="IMX format is currently only supported for YOLOv8n, YOLO11n models" } | `imx` | `yolo26n-seg_imx_model/` | ✅ | `imgsz`, `quantize`, `data`, `fraction`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [RKNN](../integrations/rockchip-rknn.md) | `rknn` | `yolo26n-seg_rknn_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [ExecuTorch](../integrations/executorch.md) | `executorch` | `yolo26n-seg_executorch_model/` | ✅ | `imgsz`, `batch`, `device` |
| [Axelera](../integrations/axelera.md) | `axelera` | `yolo26n-seg_axelera_model/` | ✅ | `imgsz`, `batch`, `quantize`, `data`, `fraction`, `device` |
| [DEEPX](../integrations/deepx.md) | `deepx` | `yolo26n-seg_deepx_model/` | ✅ | `imgsz`, `quantize`, `simplify`, `opset`, `data`, `optimize`, `device` |
| [Qualcomm QNN](../integrations/qnn.md) | `qnn` | `yolo26n-seg_qnn.onnx` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [LiteRT](../integrations/litert.md) | `litert` | `yolo26n-seg.tflite` | ✅ | `imgsz`, `quantize`, `batch`, `data`, `fraction`, `device` |
| [Hailo](../integrations/hailo.md) | `hailo` | `yolo26n-seg_hailo_model/` | ✅ | `imgsz`, `name`, `quantize`, `data`, `fraction`, `simplify`, `conf`, `iou` |
| [Huawei Ascend](../integrations/ascend.md) | `ascend` | `yolo26n-seg_ascend_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `opset`, `simplify`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" } |

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I train a YOLO26 segmentation model on a custom dataset?

To train a YOLO26 segmentation model on a custom dataset, you first need to prepare your dataset in the YOLO segmentation format. You can use tools like [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) to convert datasets from other formats. Once your dataset is ready, you can train the model using Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26 segment model
        model = YOLO("yolo26n-seg.pt")

        # Train the model
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo segment train data=path/to/your_dataset.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

Check the [Configuration](../usage/cfg.md) page for more available arguments.

### What is the difference between [object detection](https://www.ultralytics.com/glossary/object-detection) and instance segmentation in YOLO26?

Object detection identifies and localizes objects within an image by drawing bounding boxes around them, whereas instance segmentation not only identifies the bounding boxes but also delineates the exact shape of each object. YOLO26 instance segmentation models provide masks or contours that outline each detected object, which is particularly useful for tasks where knowing the precise shape of objects is important, such as medical imaging or autonomous driving.

### Why use YOLO26 for instance segmentation?

Ultralytics YOLO26 is a state-of-the-art model recognized for its high accuracy and real-time performance, making it ideal for instance segmentation tasks. YOLO26 Segment models come pretrained on the [COCO dataset](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), ensuring robust performance across a variety of objects. Additionally, YOLO supports training, validation, prediction, and export functionalities with seamless integration, making it highly versatile for both research and industry applications.

### How do I load and validate a pretrained YOLO segmentation model?

Loading and validating a pretrained YOLO segmentation model is straightforward. Here's how you can do it using both Python and CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-seg.pt")

        # Validate the model
        metrics = model.val()
        print("Mean Average Precision for boxes:", metrics.box.map)
        print("Mean Average Precision for masks:", metrics.seg.map)
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolo26n-seg.pt
        ```

These steps will provide you with validation metrics like [Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP), crucial for assessing model performance.

### How can I export a YOLO segmentation model to ONNX format?

Exporting a YOLO segmentation model to ONNX format is simple and can be done using Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-seg.pt")

        # Export the model to ONNX format
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-seg.pt format=onnx
        ```

For more details on exporting to various formats, refer to the [Export](../modes/export.md) page.
