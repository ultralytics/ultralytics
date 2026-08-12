---
comments: true
description: Master image classification using YOLO26. Learn to train, validate, predict, and export models efficiently.
keywords: YOLO26, image classification, AI, machine learning, pretrained models, ImageNet, model export, predict, train, validate
model_name: yolo26n-cls
---

# Image Classification with Ultralytics YOLO {#image-classification}

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-classification-examples.avif" alt="Ultralytics YOLO image classification of objects and scenes">

[Image classification](https://www.ultralytics.com/glossary/image-classification) is the simplest of the supported tasks and involves classifying an entire image into one of a set of predefined classes.

The output of an image classifier is a single class label and a confidence score. Image classification is useful when you need to know only what class an image belongs to and don't need to know where objects of that class are located or what their exact shape is.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5BO0Il_YYAg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Explore Ultralytics YOLO Tasks: Image Classification using Ultralytics Platform
</p>

!!! tip

    YOLO26 Classify models use the `-cls` suffix, i.e., `yolo26n-cls.pt`, and are pretrained on [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 pretrained Classify models are shown here. Detect, Segment, and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, [Semantic](semantic.md) models are pretrained on [Cityscapes](../datasets/semantic/cityscapes.md), and Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) are downloaded automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                          | size<br><sup>(pixels)</sup> | acc<br><sup>top1</sup> | acc<br><sup>top5</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B) at 224</sup> |
| ------------------------------------------------------------------------------ | --------------------------- | ---------------------- | ---------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ------------------------------ |
| [YOLO26n-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-cls) | 224                         | 71.4                   | 90.1                   | 5.0 ± 0.3                            | 1.1 ± 0.0                                 | 2.8                      | 0.5                            |
| [YOLO26s-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-cls) | 224                         | 76.0                   | 92.9                   | 7.9 ± 0.2                            | 1.3 ± 0.0                                 | 6.7                      | 1.6                            |
| [YOLO26m-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-cls) | 224                         | 78.1                   | 94.2                   | 17.2 ± 0.4                           | 2.0 ± 0.0                                 | 11.6                     | 4.9                            |
| [YOLO26l-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-cls) | 224                         | 79.0                   | 94.6                   | 23.2 ± 0.3                           | 2.8 ± 0.0                                 | 14.1                     | 6.2                            |
| [YOLO26x-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-cls) | 224                         | 79.9                   | 95.0                   | 41.4 ± 0.9                           | 3.8 ± 0.0                                 | 29.6                     | 13.6                           |

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set. <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers. Pretrained checkpoints retain the full training architecture and may show higher counts.

## Train

Train YOLO26n-cls on the MNIST160 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 64. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-cls.yaml").load("yolo26n-cls.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo classify train data=mnist160 model=yolo26n-cls.yaml epochs=100 imgsz=64

        # Start training from a pretrained *.pt model
        yolo classify train data=mnist160 model=yolo26n-cls.pt epochs=100 imgsz=64

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo classify train data=mnist160 model=yolo26n-cls.yaml pretrained=yolo26n-cls.pt epochs=100 imgsz=64
        ```

!!! tip

    Ultralytics YOLO classification uses [`torchvision.transforms.RandomResizedCrop`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html) for training and [`torchvision.transforms.CenterCrop`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.CenterCrop.html) for validation and inference.
    These cropping-based transforms assume square inputs and may inadvertently crop out important regions from images with extreme aspect ratios, potentially causing loss of critical visual information during training.
    To preserve the full image while maintaining its proportions, consider using [`torchvision.transforms.Resize`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html) instead of cropping transforms.

    You can implement this by customizing your augmentation pipeline through a custom `ClassificationDataset` and `ClassificationTrainer`.


    ```python
    import torch
    import torchvision.transforms as T

    from ultralytics import YOLO
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


    class CustomizedDataset(ClassificationDataset):
        """A customized dataset class for image classification with enhanced data augmentation transforms."""

        def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
            """Initialize a customized classification dataset with enhanced data augmentation transforms."""
            super().__init__(root, args, augment, prefix)

            # Add your custom training transforms here
            train_transforms = T.Compose(
                [
                    T.Resize((args.imgsz, args.imgsz)),
                    T.RandomHorizontalFlip(p=args.fliplr),
                    T.RandomVerticalFlip(p=args.flipud),
                    T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                    T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                    T.ToTensor(),
                    T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                    T.RandomErasing(p=args.erasing, inplace=True),
                ]
            )

            # Add your custom validation transforms here
            val_transforms = T.Compose(
                [
                    T.Resize((args.imgsz, args.imgsz)),
                    T.ToTensor(),
                    T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                ]
            )
            self.torch_transforms = train_transforms if augment else val_transforms


    class CustomizedTrainer(ClassificationTrainer):
        """A customized trainer class for YOLO classification models with enhanced dataset handling."""

        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            """Build a customized dataset for classification training and the validation during training."""
            return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


    class CustomizedValidator(ClassificationValidator):
        """A customized validator class for YOLO classification models with enhanced dataset handling."""

        def build_dataset(self, img_path: str):
            """Build a customized dataset for classification standalone validation (no augmentation)."""
            return CustomizedDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)


    model = YOLO("yolo26n-cls.pt")
    model.train(data="imagenet1000", trainer=CustomizedTrainer, epochs=10, imgsz=224, batch=64)
    model.val(data="imagenet1000", validator=CustomizedValidator, imgsz=224, batch=64)
    ```

### Dataset format

YOLO classification dataset format can be found in detail in the [Dataset Guide](../datasets/classify/index.md). Classification datasets can also be managed and labeled with [Ultralytics Platform annotation tools](../platform/data/annotation.md).

## Val

Validate trained YOLO26n-cls model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the MNIST160 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.top1  # top1 accuracy
        metrics.top5  # top5 accuracy
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolo26n-cls.pt  # val official model
        yolo classify val model=path/to/best.pt # val custom model
        ```

!!! tip

    As mentioned in the [training section](#train), you can handle extreme aspect ratios during training by using a custom `ClassificationTrainer`. You need to apply the same approach for consistent validation results by implementing a custom `ClassificationValidator` when calling the `val()` method. Refer to the complete code example in the [training section](#train) for implementation details.

## Predict

Use a trained YOLO26n-cls model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            top1 = result.probs.top1  # top predicted class ID
            top1_conf = result.probs.top1conf  # top prediction confidence
            top1_name = result.names[top1]  # top predicted class name
        ```

    === "CLI"

        ```bash
        yolo classify predict model=yolo26n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Results Output

Image classification returns one `Results` object per image. The primary prediction field is `result.probs`, which
contains the class probability vector and helpers for top predictions.

| Attribute               | Type            | Shape   | Description            |
| ----------------------- | --------------- | ------- | ---------------------- |
| `result.probs`          | `Probs`         | `(C,)`  | Class probabilities.   |
| `result.probs.data`     | `torch.float32` | `(C,)`  | Probability per class. |
| `result.probs.top1`     | `int`           | `()`    | Top class ID.          |
| `result.probs.top1conf` | `torch.float32` | `()`    | Top confidence.        |
| `result.probs.top5`     | `list[int]`     | `(<=5)` | Top-5 class IDs.       |

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

## Export

Export a YOLO26n-cls model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-cls.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

Available YOLO26-cls export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-cls.onnx`. Usage examples are shown for your model after export completes.

| Format | `format` Argument | Model | Metadata | Arguments |
| ---------------------------------------------------------- | ----------------- | ------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| [PyTorch](https://pytorch.org/) | - | `yolo26n-cls.pt` | ✅ | - |
| [TorchScript](../integrations/torchscript.md) | `torchscript` | `yolo26n-cls.torchscript` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [ONNX](../integrations/onnx.md) | `onnx` | `yolo26n-cls.onnx` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [OpenVINO](../integrations/openvino.md) | `openvino` | `yolo26n-cls_openvino_model/` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TensorRT](../integrations/tensorrt.md) | `engine` | `yolo26n-cls.engine` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `workspace`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [CoreML](../integrations/coreml.md) | `coreml` | `yolo26n-cls.mlpackage` | ✅ | `imgsz`, `dynamic`, `quantize`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model` | `yolo26n-cls_saved_model/` | ✅ | `imgsz`, `keras`, `quantize`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TF GraphDef](../integrations/tf-graphdef.md) | `pb` | `yolo26n-cls.pb` | ❌ | `imgsz`, `opset`, `batch`, `device` |
| [TF Edge TPU](../integrations/edge-tpu.md) | `edgetpu` | `yolo26n-cls_edgetpu.tflite` | ✅ | `imgsz`, `quantize`, `opset`, `data`, `fraction`, `device` |
| [PaddlePaddle](../integrations/paddlepaddle.md) | `paddle` | `yolo26n-cls_paddle_model/` | ✅ | `imgsz`, `batch`, `device` |
| [MNN](../integrations/mnn.md) | `mnn` | `yolo26n-cls.mnn` | ✅ | `imgsz`, `batch`, `dynamic`, `quantize`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [NCNN](../integrations/ncnn.md) | `ncnn` | `yolo26n-cls_ncnn_model/` | ✅ | `imgsz`, `quantize`, `batch`, `device` |
| [IMX500](../integrations/sony-imx500.md):material-information-outline:{ title="IMX format is currently only supported for YOLOv8n, YOLO11n models" } | `imx` | `yolo26n-cls_imx_model/` | ✅ | `imgsz`, `quantize`, `data`, `fraction`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [RKNN](../integrations/rockchip-rknn.md) | `rknn` | `yolo26n-cls_rknn_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [ExecuTorch](../integrations/executorch.md) | `executorch` | `yolo26n-cls_executorch_model/` | ✅ | `imgsz`, `batch`, `device` |
| [Axelera](../integrations/axelera.md) | `axelera` | `yolo26n-cls_axelera_model/` | ✅ | `imgsz`, `batch`, `quantize`, `data`, `fraction`, `device` |
| [DEEPX](../integrations/deepx.md) | `deepx` | `yolo26n-cls_deepx_model/` | ✅ | `imgsz`, `quantize`, `simplify`, `opset`, `data`, `optimize`, `device` |
| [Qualcomm QNN](../integrations/qnn.md) | `qnn` | `yolo26n-cls_qnn.onnx` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [LiteRT](../integrations/litert.md) | `litert` | `yolo26n-cls.tflite` | ✅ | `imgsz`, `quantize`, `batch`, `data`, `fraction`, `device` |
| [Hailo](../integrations/hailo.md) | `hailo` | `yolo26n-cls_hailo_model/` | ✅ | `imgsz`, `name`, `quantize`, `data`, `fraction`, `simplify`, `conf`, `iou` |
| [Huawei Ascend](../integrations/ascend.md) | `ascend` | `yolo26n-cls_ascend_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `opset`, `simplify`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" } |

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is the purpose of YOLO26 in image classification?

YOLO26 models, such as `yolo26n-cls.pt`, are designed for efficient image classification. They assign a single class label to an entire image along with a confidence score. This is particularly useful for applications where knowing the specific class of an image is sufficient, rather than identifying the location or shape of objects within the image.

### How do I train a YOLO26 model for image classification?

To train a YOLO26 model, you can use either Python or CLI commands. For example, to train a `yolo26n-cls` model on the MNIST160 dataset for 100 epochs at an image size of 64:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        yolo classify train data=mnist160 model=yolo26n-cls.pt epochs=100 imgsz=64
        ```

For more configuration options, visit the [Configuration](../usage/cfg.md) page.

### Where can I find pretrained YOLO26 classification models?

Pretrained YOLO26 classification models can be found in the [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26) section. Models like `yolo26n-cls.pt`, `yolo26s-cls.pt`, `yolo26m-cls.pt`, etc., are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset and can be easily downloaded and used for various image classification tasks.

### How can I export a trained YOLO26 model to different formats?

You can export a trained YOLO26 model to various formats using Python or CLI commands. For instance, to export a model to ONNX format:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load the trained model

        # Export the model to ONNX
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-cls.pt format=onnx # export the trained model to ONNX format
        ```

For detailed export options, refer to the [Export](../modes/export.md) page.

### How do I validate a trained YOLO26 classification model?

To validate a trained model's accuracy on a dataset like MNIST160, you can use the following Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load the trained model

        # Validate the model
        metrics = model.val()  # no arguments needed, uses the dataset and settings from training
        metrics.top1  # top1 accuracy
        metrics.top5  # top5 accuracy
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolo26n-cls.pt # validate the trained model
        ```

For more information, visit the [Validate](#val) section.
