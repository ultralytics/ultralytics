# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
from typing import Generator, Union

import numpy as np
import torch
from pandas import DataFrame
from PIL import Image

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils.metrics import ClassifyMetrics, DetMetrics, OBBMetrics, PoseMetrics, SegmentMetrics

DROP = {"self", "__class__"}


class ModelMeta(Model):
    """
    A class representing the meta information and functionality of a model.

    This class inherits from the base `Model` class and provides additional methods for exporting the model to different formats and performing predictions on input data.

    Args:
        model (str): The model file for training. Accepts a path to either a `.pt` pretrained model or a `.yaml` configuration file. Essential for defining the model structure or initializing weights.
        task (str): The task associated with the model, such as object detection, image classification, or semantic segmentation.
        verbose (bool): Whether to print detailed information during model operations.

    Methods:
        export: Exports the model to a target format.
        predict: Performs predictions on the given image source using the YOLO model.
        train: Trains the model using the specified dataset and training configuration.
    """

    def __init__(self, model: str = None, task: str = None, verbose: bool = False) -> None:
        super().__init__(model=model, task=task, verbose=verbose)

    def benchmark(
        self,
        *,
        model: str = None,
        data: str = None,
        imgsz: Union[int, tuple[int]] = 640,
        half: bool = False,
        int8: bool = False,
        device: str = None,
        verbose: bool = False,
        mode: str = None,
    ) -> DataFrame:
        """
        Benchmarks the model across various export formats to evaluate performance.

        This method assesses the model's performance in different export formats, such as ONNX, TorchScript, etc.
        It uses the 'benchmark' function from the ultralytics.utils.benchmarks module. The benchmarking is
        configured using a combination of default configuration values, model-specific arguments, method-specific
        defaults, and any additional user-provided keyword arguments.

        Args:
            model (str): Specifies the path to the model file. Accepts both `.pt` and `.yaml` formats, e.g., `"yolov8n.pt"` for pre-trained models or configuration files.
            data (str): Path to a YAML file defining the dataset for benchmarking, typically including paths and settings for validation data. Example: `"coco8.yaml"`.
            imgsz (int or tuple): The input image size for the model. Can be a single integer for square images or a tuple `(width, height)` for non-square, e.g., `(640, 480)`.
            half (bool): Enables FP16 (half-precision) inference, reducing memory usage and possibly increasing speed on compatible hardware. Use `half=True` to enable.
            int8 (bool): Activates INT8 quantization for further optimized performance on supported devices, especially useful for edge devices. Set `int8=True` to use.
            device (str or list): Defines the computation device(s) for benchmarking, such as `"cpu"`, `"cuda:0"`, or a list of devices like `"cuda:0,1"` for multi-GPU setups.
            verbose (bool or float): Controls the level of detail in logging output. A boolean value; set `verbose=True` for detailed logs or a float for thresholding errors.
            mode (str): YOLO mode, i.e. train, val, predict, export, track, benchmark

        Returns:
            (DataFrame): A pandas DataFrame containing the results of the benchmarking process, including metrics for
                different export formats.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolov8n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        return super().benchmark(**{k: v for k, v in locals().items() if k not in DROP})

    def export(
        self,
        *,
        format: str = "torchscript",
        imgsz: Union[int, tuple] = 640,
        keras: bool = False,
        optimize: bool = False,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        simplify: bool = False,
        opset: int = None,
        workspace: float = 4.0,
        nms: bool = False,
        batch: int = 1,
    ) -> str:
        """
        Exports the model to a target format.

        Args:
            format (str): Target format for the exported model, such as 'onnx', 'torchscript', 'tensorflow', or others, defining compatibility with various deployment environments.
            imgsz (int | tuple): Desired image size for the model input. Can be an integer for square images or a tuple (height, width) for specific dimensions.
            keras (bool): Enables export to Keras format for TensorFlow SavedModel, providing compatibility with TensorFlow serving and APIs.
            optimize (bool): Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving performance.
            half (bool): Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.
            int8 (bool): Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices.
            dynamic (bool): Allows dynamic input sizes for ONNX and TensorRT exports, enhancing flexibility in handling varying image dimensions.
            simplify (bool): Simplifies the model graph for ONNX exports with 'onnxslim', potentially improving performance and compatibility.
            opset (int): Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version.
            workspace (float): Sets the maximum workspace size in GiB for TensorRT optimizations, balancing memory usage and performance.
            nms (bool): Adds Non-Maximum Suppression (NMS) to the CoreML export, essential for accurate and efficient detection post-processing.
            batch (int): Specifies export model batch inference size or the max number of images the exported model will process concurrently in 'predict' mode.

        Returns:
            (str): The path to the exported model file.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            ValueError: If an unsupported export format is specified.
            RuntimeError: If the export process fails due to errors.

        Examples:
            >>> model = YOLO("yolov8n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        return super().export(**{k: v for k, v in locals().items() if k not in DROP})

    def predict(
        self,
        source: Union[
            str,
            Path,
            np.ndarray,
            Image.Image,
            torch.Tensor,
            list[Union[str, Path, np.ndarray, Image.Image, torch.Tensor]],
        ] = None,
        *,
        predictor: object = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Union[int, tuple[int, int]] = 640,
        half: bool = False,
        device: str = None,
        max_det: int = 100,
        vid_stride: int = 1,
        stream_buffer: bool = False,
        visualize: bool = False,
        augment: bool = False,
        agnostic_nms: bool = False,
        classes: list[int] = None,
        retina_masks: bool = False,
        embed: list[int] = None,
        stream: bool = False,
        show: bool = False,
        save: bool = False,
        save_frames: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        save_crop: bool = False,
        show_labels: bool = False,
        show_conf: bool = False,
        show_boxes: bool = False,
        line_width: int = None,
        verbose: bool = True,
    ) -> Union[list[Results], Generator[Results, None, None]]:
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode.

        Args:
            predictor (BasePredictor | None): An instance of a custom predictor class for making predictions. If None, the method uses a default predictor.
            source (str | Path | np.ndarray | Image.Image | torch.Tensor | list[str | Path | np.ndarray | Image.Image | torch.Tensor]): Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across different types of input.
            conf (float): Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.
            iou (float): Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
            imgsz (int | tuple[int, int]): Defines the image size for inference. Can be a single integer `640` for square resizing or a (height, width) tuple. Proper sizing can improve detection accuracy and processing speed.
            half (bool): Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.
            device (str): Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.
            max_det (int): Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.
            vid_stride (int): Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.
            stream_buffer (bool): Determines if all frames should be buffered when processing video streams (`True`), or if the model should return the most recent frame (`False`). Useful for real-time applications.
            visualize (bool): Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.
            augment (bool): Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.
            agnostic_nms (bool): Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.
            classes (list[int]): Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.
            retina_masks (bool): Uses high-resolution segmentation masks if available in the model. This can enhance mask quality for segmentation tasks, providing finer detail.
            embed (list[int]): Specifies the layers from which to extract feature vectors or embeddings. Useful for downstream tasks like clustering or similarity search.
            stream (bool): Returns a generator for memory efficient processing.
            show (bool): If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing.
            save (bool): Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results.
            save_frames (bool): When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis.
            save_txt (bool): Saves detection results in a text file, following the format `[class] [x_center] [y_center] [width] [height] [confidence]`. Useful for integration with other analysis tools.
            save_conf (bool): Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis.
            save_crop (bool): Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects.
            show_labels (bool): Displays labels for each detection in the visual output. Provides immediate understanding of detected objects.
            show_conf (bool): Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection.
            show_boxes (bool): Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames.
            line_width (None or int): Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity.
            verbose (bool): Enables verbose output during inference, providing detailed logs and progress updates. Useful for closely monitoring the inference process.

        Returns:
            (list[Results] | Generator[Results]): Inference results as list or generator of Results objects if `stream==True`.

        Examples:
        >>> model = YOLO("yolov8n.pt")
        >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
        >>> for r in results:
        ...     print(r.boxes.data)  # print detection bounding boxes

        Notes:
            - If 'source' is not provided, it defaults to the ASSETS constant with a warning.
            - The method sets up a new predictor if not already present and updates its arguments with each call.
            - For SAM-type models, 'prompts' can be passed as a keyword argument.
        """
        return super().predict(**{k: v for k, v in locals().items() if k not in DROP})

    def train(
        self,
        *,
        trainer: object = None,
        model: str = None,
        data: str = None,
        epochs: int = 100,
        time: int = None,
        patience: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        save: bool = True,
        save_period: int = -1,
        cache: Union[str, bool] = False,
        device: str = None,
        workers: int = 8,
        project: str = None,
        name: str = None,
        exist_ok: bool = False,
        pretrained: bool = True,
        optimizer: str = "auto",
        verbose: bool = False,
        seed: int = 0,
        deterministic: bool = True,
        single_cls: bool = False,
        rect: bool = False,
        cos_lr: bool = False,
        close_mosaic: int = 10,
        resume: bool = False,
        amp: bool = True,
        fraction: float = 1.0,
        profile: bool = False,
        freeze: Union[int, list[int]] = None,
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: Union[int, float] = 3.0,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        pose: float = 12.0,
        kobj: float = 2.0,
        label_smoothing: float = 0.0,
        nbs: int = 64,
        overlap_mask: bool = True,
        mask_ratio: int = 4,
        dropout: float = 0.0,
        val: bool = True,
        plots: bool = True,
    ) -> Union[DetMetrics, SegmentMetrics, PoseMetrics, OBBMetrics, ClassifyMetrics]:
        """
        Trains the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings. It supports training with a
        custom trainer or the default training approach. The method handles scenarios such as resuming training
        from a checkpoint, integrating with Ultralytics HUB, and updating model and configuration after training.

        When using Ultralytics HUB, if the session has a loaded model, the method prioritizes HUB training
        arguments and warns if local arguments are provided. It checks for pip updates and combines default
        configurations, method-specific defaults, and user-provided arguments to configure the training process.

        Args:
            trainer (BaseTrainer | None): Custom trainer instance for model training. If None, uses default.
            model (str): Specifies the model file for training. Accepts a path to either a `.pt` pretrained model or a `.yaml` configuration file. Essential for defining the model structure or initializing weights.
            data (str): Path to the dataset configuration file (e.g., `coco8.yaml`). This file contains dataset-specific parameters, including paths to training and validation data, class names, and number of classes.
            epochs (int): Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.
            time (int): Maximum training time in hours. If set, this overrides the epochs argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.
            patience (int): Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus.
            batch (int): Batch size, with three modes: set as an integer (e.g., `batch=16`), auto mode for 60% GPU memory utilization (`batch=-1`), or auto mode with specified utilization fraction (`batch=0.70`).
            imgsz (int): Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.
            save (bool): Enables saving of training checkpoints and final model weights. Useful for resuming training or model deployment.
            save_period (int): Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.
            cache (Union[str, bool]): Enables caching of dataset images in memory (`True`/`ram`), on disk (`disk`), or disables it (`False`). Improves training speed by reducing disk I/O at the cost of increased memory usage.
            device (str): Specifies the computational device(s) for training: a single GPU (`device=0`), multiple GPUs (`device=0,1`), CPU (`device=cpu`), or MPS for Apple silicon (`device=mps`).
            workers (int): Number of worker threads for data loading (per `RANK` if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.
            project (str): Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.
            name (str): Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.
            exist_ok (bool): If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.
            pretrained (bool): Determines whether to start training from a pretrained model. Can be a boolean value or a string path to a specific model from which to load weights. Enhances training efficiency and model performance.
            optimizer (str): Choice of optimizer for training. Options include `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp` etc., or `auto` for automatic selection based on model configuration. Affects convergence speed and stability.
            verbose (bool): Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.
            seed (int): Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.
            deterministic (bool): Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.
            single_cls (bool): Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.
            rect (bool): Enables rectangular training, optimizing batch composition for minimal padding. Can improve efficiency and speed but may affect model accuracy.
            cos_lr (bool): Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.
            close_mosaic (int): Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.
            resume (bool): Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.
            amp (bool): Enables Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.
            fraction (float): Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.
            profile (bool): Enables profiling of ONNX and TensorRT speeds during training, useful for optimizing model deployment.
            freeze (int | list[int]): Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.
            lr0 (float): Initial learning rate (i.e. `SGD=1E-2`, `Adam=1E-3`) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
            lrf (float): Final learning rate as a fraction of the initial rate = (`lr0 * lrf`), used in conjunction with schedulers to adjust the learning rate over time.
            momentum (float): Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.
            weight_decay (float): L2 regularization term, penalizing large weights to prevent overfitting.
            warmup_epochs (int | float): Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.
            warmup_momentum (float): Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.
            warmup_bias_lr (float): Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.
            box (float): Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.
            cls (float): Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.
            dfl (float): Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.
            pose (float): Weight of the pose loss in models trained for pose estimation, influencing the emphasis on accurately predicting pose keypoints.
            kobj (float): Weight of the keypoint objectness loss in pose estimation models, balancing detection confidence with pose accuracy.
            label_smoothing (float): Applies label smoothing, softening hard labels to a mix of the target label and a uniform distribution over labels, can improve generalization.
            nbs (int): Nominal batch size for normalization of loss.
            overlap_mask (bool): Determines whether segmentation masks should overlap during training, applicable in instance segmentation tasks.
            mask_ratio (int): Downsample ratio for segmentation masks, affecting the resolution of masks used during training.
            dropout (float): Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.
            val (bool): Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.
            plots (bool): Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.

        Returns:
            (DetMetrics | SegmentMetrics | PoseMetrics | OBBMetrics | ClassifyMetrics): The training metrics obtained from the final validation.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            PermissionError: If there is a permission issue with the HUB session.
            ModuleNotFoundError: If the HUB SDK is not installed.

        Examples:
            >>> model = YOLO("yolov8n.pt")
            >>> results = model.train(data="coco128.yaml", epochs=3)
        """
        return super().train(**{k: v for k, v in locals().items() if k not in DROP})

    def track(
        self,
        source: Union[
            str,
            Path,
            np.ndarray,
            Image.Image,
            torch.Tensor,
            list[Union[str, Path, np.ndarray, Image.Image, torch.Tensor]],
        ] = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        tracker: str = "botsort",
        stream: bool = False,
        persist: bool = False,
        imgsz: Union[int, tuple[int, int]] = 640,
        half: bool = False,
        device: str = None,
        max_det: int = 100,
        vid_stride: int = 1,
        stream_buffer: bool = False,
        visualize: bool = False,
        augment: bool = False,
        agnostic_nms: bool = False,
        classes: list[int] = None,
        retina_masks: bool = False,
        show: bool = False,
        save: bool = False,
        save_frames: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        save_crop: bool = False,
        show_labels: bool = False,
        show_conf: bool = False,
        show_boxes: bool = False,
        line_width: int = None,
    ) -> Union[list[Results], Generator[Results, None, None]]:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It handles
        various input sources such as file paths or video streams, and supports customization through keyword arguments.
        The method registers trackers if not already present and can persist them between calls.

        Args:
            source (str | Path | np.ndarray | Image.Image | torch.Tensor | list[str | Path | np.ndarray | Image.Image | torch.Tensor]): Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across different types of input.
            conf (float): Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.
            iou (float): Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
            tracker (str): Tracker type to use for object tracking, such as 'bytetrack' or 'botsort'.
            stream (bool): If True, treats the input source as a continuous video stream. Defaults to False.
            persist (bool): If True, persists trackers between different calls to this method. Defaults to False.
            imgsz (int | tuple[int, int]): Defines the image size for inference. Can be a single integer `640` for square resizing or a (height, width) tuple. Proper sizing can improve detection accuracy and processing speed.
            half (bool): Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.
            device (str): Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.
            max_det (int): Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.
            vid_stride (int): Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.
            stream_buffer (bool): Determines if all frames should be buffered when processing video streams (`True`), or if the model should return the most recent frame (`False`). Useful for real-time applications.
            visualize (bool): Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.
            augment (bool): Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.
            agnostic_nms (bool): Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.
            classes (list[int]): Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.
            retina_masks (bool): Uses high-resolution segmentation masks if available in the model. This can enhance mask quality for segmentation tasks, providing finer detail.
            stream (bool): Returns a generator for memory efficient processing.
            show (bool): If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing.
            save (bool): Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results.
            save_frames (bool): When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis.
            save_txt (bool): Saves detection results in a text file, following the format `[class] [x_center] [y_center] [width] [height] [confidence]`. Useful for integration with other analysis tools.
            save_conf (bool): Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis.
            save_crop (bool): Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects.
            show_labels (bool): Displays labels for each detection in the visual output. Provides immediate understanding of detected objects.
            show_conf (bool): Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection.
            show_boxes (bool): Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames.
            line_width (None or int): Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity.

        Returns:
            (list[Results] | Generator[Results]): A list, or generator when `stream==True`, of tracking results, each a Results object.

        Raises:
            AttributeError: If the predictor does not have registered trackers.

        Examples:
            >>> model = YOLO("yolov8n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # print tracking IDs

        Notes:
            - This method sets a default confidence threshold of 0.1 for ByteTrack-based tracking.
            - The tracking mode is explicitly set in the keyword arguments.
            - Batch size is set to 1 for tracking in videos.
        """
        return super().track(**{k: v for k, v in locals().items() if k not in DROP})

    def val(
        self,
        *,
        data: str = None,
        imgsz: int = 640,
        batch: int = 16,
        save_json: bool = False,
        save_hybrid: bool = False,
        conf: float = 0.001,
        iou: float = 0.6,
        max_det: int = 300,
        half: bool = True,
        device: str = None,
        dnn: bool = False,
        plots: bool = False,
        rect: bool = False,
        split: str = "val",
    ) -> Union[DetMetrics, SegmentMetrics, PoseMetrics, OBBMetrics, ClassifyMetrics]:
        """
        Validates the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for customization through various settings. It
        supports validation with a custom validator or the default validation approach. The method combines default
        configurations, method-specific defaults, and user-provided arguments to configure the validation process.

        Args:
            data (str): Specifies the path to the dataset configuration file (e.g., `coco8.yaml`). This file includes paths to validation data, class names, and number of classes.
            imgsz (int): Defines the size of input images. All images are resized to this dimension before processing.
            batch (int): Sets the number of images per batch. Use `-1` for AutoBatch, which automatically adjusts based on GPU memory availability.
            save_json (bool): If `True`, saves the results to a JSON file for further analysis or integration with other tools.
            save_hybrid (bool): If `True`, saves a hybrid version of labels that combines original annotations with additional model predictions.
            conf (float): Sets the minimum confidence threshold for detections. Detections with confidence below this threshold are discarded.
            iou (float): Sets the Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Helps in reducing duplicate detections.
            max_det (int): Limits the maximum number of detections per image. Useful in dense scenes to prevent excessive detections.
            half (bool): Enables half-precision (FP16) computation, reducing memory usage and potentially increasing speed with minimal impact on accuracy.
            device (str): Specifies the device for validation (`cpu`, `cuda:0`, etc.). Allows flexibility in utilizing CPU or GPU resources.
            dnn (bool): If `True`, uses the OpenCV DNN module for ONNX model inference, offering an alternative to PyTorch inference methods.
            plots (bool): When set to `True`, generates and saves plots of predictions versus ground truth for visual evaluation of the model's performance.
            rect (bool): If `True`, uses rectangular inference for batching, reducing padding and potentially increasing speed and efficiency.
            split (str): Determines the dataset split to use for validation (`val`, `test`, or `train`). Allows flexibility in choosing the data segment for performance evaluation.

        Returns:
            (DetMetrics | SegmentMetrics | PoseMetrics | OBBMetrics | ClassifyMetrics): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolov8n.pt")
            >>> results = model.val(data="coco128.yaml", imgsz=640)
            >>> print(results.box.map)  # Print mAP50-95
        """
        return super().val(**{k: v for k, v in locals().items() if k not in DROP})
