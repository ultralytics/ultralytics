# Ultralytics YOLOv8 Modes

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png">

Ultralytics YOLOv8 supports several **modes** that can be used to perform different tasks. These modes are:

**Train**: For training a YOLOv8 model on a custom dataset.  
**Val**: For validating a YOLOv8 model after it has been trained.  
**Predict**: For making predictions using a trained YOLOv8 model on new images or videos.  
**Export**: For exporting a YOLOv8 model to a format that can be used for deployment.  
**Track**: For tracking objects in real-time using a YOLOv8 model.  
**Benchmark**: For benchmarking YOLOv8 exports (ONNX, TensorRT, etc.) speed and accuracy.

## [Train](train.md)

Train mode is used for training a YOLOv8 model on a custom dataset. In this mode, the model is trained using the
specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can
accurately predict the classes and locations of objects in an image.

[Train Examples](train.md){ .md-button .md-button--primary}

## [Val](val.md)

Val mode is used for validating a YOLOv8 model after it has been trained. In this mode, the model is evaluated on a
validation set to measure its accuracy and generalization performance. This mode can be used to tune the hyperparameters
of the model to improve its performance.

[Val Examples](val.md){ .md-button .md-button--primary}

## [Predict](predict.md)

Predict mode is used for making predictions using a trained YOLOv8 model on new images or videos. In this mode, the
model is loaded from a checkpoint file, and the user can provide images or videos to perform inference. The model
predicts the classes and locations of objects in the input images or videos.

[Predict Examples](predict.md){ .md-button .md-button--primary}

## [Export](export.md)

Export mode is used for exporting a YOLOv8 model to a format that can be used for deployment. In this mode, the model is
converted to a format that can be used by other software applications or hardware devices. This mode is useful when
deploying the model to production environments.

[Export Examples](export.md){ .md-button .md-button--primary}

## [Track](track.md)

Track mode is used for tracking objects in real-time using a YOLOv8 model. In this mode, the model is loaded from a
checkpoint file, and the user can provide a live video stream to perform real-time object tracking. This mode is useful
for applications such as surveillance systems or self-driving cars.

[Track Examples](track.md){ .md-button .md-button--primary}

## [Benchmark](benchmark.md)

Benchmark mode is used to profile the speed and accuracy of various export formats for YOLOv8. The benchmarks provide
information on the size of the exported format, its `mAP50-95` metrics (for object detection and segmentation)
or `accuracy_top5` metrics (for classification), and the inference time in milliseconds per image across various export
formats like ONNX, OpenVINO, TensorRT and others. This information can help users choose the optimal export format for
their specific use case based on their requirements for speed and accuracy.

[Benchmark Examples](benchmark.md){ .md-button .md-button--primary}
