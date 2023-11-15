---
comments: true
description: Find solutions to your common Ultralytics YOLO related queries. Learn about hardware requirements, fine-tuning YOLO models, conversion to ONNX/TensorFlow, and more.
keywords: Ultralytics, YOLO, FAQ, hardware requirements, ONNX, TensorFlow, real-time detection, YOLO accuracy
---

# Ultralytics YOLO Frequently Asked Questions (FAQ)

This FAQ section addresses some common questions and issues users might encounter while working with Ultralytics YOLO repositories.

## 1. What are the hardware requirements for running Ultralytics YOLO?

Ultralytics YOLO can be run on a variety of hardware configurations, including CPUs, GPUs, and even some edge devices. However, for optimal performance and faster training and inference, we recommend using a GPU with a minimum of 8GB of memory. NVIDIA GPUs with CUDA support are ideal for this purpose.

## 2. How do I fine-tune a pre-trained YOLO model on my custom dataset?

To fine-tune a pre-trained YOLO model on your custom dataset, you'll need to create a dataset configuration file (YAML) that defines the dataset's properties, such as the path to the images, the number of classes, and class names. Next, you'll need to modify the model configuration file to match the number of classes in your dataset. Finally, use the `train.py` script to start the training process with your custom dataset and the pre-trained model. You can find a detailed guide on fine-tuning YOLO in the Ultralytics documentation.

## 3. How do I convert a YOLO model to ONNX or TensorFlow format?

Ultralytics provides built-in support for converting YOLO models to ONNX format. You can use the `export.py` script to convert a saved model to ONNX format. If you need to convert the model to TensorFlow format, you can use the ONNX model as an intermediary and then use the ONNX-TensorFlow converter to convert the ONNX model to TensorFlow format.

## 4. Can I use Ultralytics YOLO for real-time object detection?

Yes, Ultralytics YOLO is designed to be efficient and fast, making it suitable for real-time object detection tasks. The actual performance will depend on your hardware configuration and the complexity of the model. Using a GPU and optimizing the model for your specific use case can help achieve real-time performance.

## 5. How can I improve the accuracy of my YOLO model?

Improving the accuracy of a YOLO model may involve several strategies, such as:

- Fine-tuning the model on more annotated data
- Data augmentation to increase the variety of training samples
- Using a larger or more complex model architecture
- Adjusting the learning rate, batch size, and other hyperparameters
- Using techniques like transfer learning or knowledge distillation

Remember that there's often a trade-off between accuracy and inference speed, so finding the right balance is crucial for your specific application.

If you have any more questions or need assistance, don't hesitate to consult the Ultralytics documentation or reach out to the community through GitHub Issues or the official discussion forum.
