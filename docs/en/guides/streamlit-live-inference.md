---
title: YOLO26 Live Inference with Streamlit
comments: true
description: Build a live inference web app with Streamlit and Ultralytics YOLO26 to run real-time object detection on webcam, video, or image input in your browser.
keywords: Streamlit, YOLO26, Real-time Object Detection, Streamlit Application, YOLO26 Streamlit Tutorial, Webcam Object Detection
---

# Live Inference with Streamlit Application using Ultralytics YOLO26

## Introduction

[Streamlit](https://streamlit.io/) makes it simple to build and deploy interactive web applications. Combining it with Ultralytics YOLO26 allows for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and analysis directly in your browser. YOLO26's speed and accuracy make it suitable for live video streams in applications such as security and retail. For a walkthrough of the app, see [running an interactive AI app with Streamlit and Ultralytics YOLO11](https://www.ultralytics.com/blog/run-an-interactive-ai-app-with-streamlit-and-ultralytics-yolo11).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Fm72tfuQG70"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Build a Live Inference App with Ultralytics YOLO26 & Streamlit | Detection & Segmentation 🚀
</p>

|                                              Aquaculture                                               |                                             Animal Husbandry                                              |
| :----------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: |
| ![Fish Detection using Ultralytics YOLO26](https://cdn.ul.run/i/bc7f3d6a3540c0b6e4b7fbd559d4cd54.avif) | ![Animals Detection using Ultralytics YOLO26](https://cdn.ul.run/i/88f6470efef82fd4195c3a0507e2d739.avif) |
|                                Fish Detection using Ultralytics YOLO26                                 |                                Animals Detection using Ultralytics YOLO26                                 |

## Advantages of Live Inference

- **Real-Time Object Detection**: Streamlit combined with YOLO26 enables real-time object detection directly from your webcam feed. This allows for immediate analysis and insights, making it ideal for [applications requiring instant feedback](../modes/predict.md).
- **User-Friendly Deployment**: Streamlit's interactive interface makes it easy to deploy and use the application without extensive technical knowledge. Users can start live inference with a simple click, enhancing accessibility and usability.
- **Efficient Resource Utilization**: YOLO26's optimized algorithms ensure high-speed processing with minimal computational resources. This efficiency allows for smooth and reliable webcam inference even on standard hardware, making advanced [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) accessible to a wider audience.

## Streamlit Application Code

!!! tip "Ultralytics Installation"

    Before you start building the application, ensure you have the Ultralytics Python package installed.

    ```bash
    pip install ultralytics
    ```

!!! example "Inference using Streamlit with Ultralytics YOLO"

    === "CLI"

        ```bash
        yolo solutions inference

        yolo solutions inference model="path/to/model.pt"

        yolo solutions help # list the available solution commands and arguments
        ```

        These commands launch the default Streamlit interface that ships with Ultralytics.

    === "Python"

        ```python
        from ultralytics import solutions

        inf = solutions.Inference(
            model="yolo26n.pt",  # you can use any model that Ultralytics supports, e.g., YOLO26, or a custom-trained model
        )

        inf.inference()

        # Make sure to run the file using command `streamlit run path/to/file.py`
        ```

This will launch the Streamlit application in your default web browser. You will see the main title, subtitle, and the sidebar with configuration options. Select your desired YOLO26 model, set the confidence and [NMS thresholds](https://www.ultralytics.com/glossary/non-maximum-suppression-nms), and click the "Start" button to begin the real-time object detection.

## How It Works

Under the hood, the Streamlit application uses the [Ultralytics solutions module](../reference/solutions/streamlit_inference.md) to create an interactive interface. When you start the inference, the application:

1. Captures frames from your webcam, an uploaded video file, or uploaded images
2. Processes each frame through the YOLO26 model
3. Applies object detection with your specified confidence and IoU thresholds
4. Displays both the original and annotated frames in real time
5. Optionally enables object tracking if selected

The application provides a clean, user-friendly interface with controls to adjust model parameters and start/stop inference at any time.

## Conclusion

By following this guide, you have created a real-time object detection application using Streamlit and Ultralytics YOLO26. The application runs YOLO26 on your webcam, video, or image input through a simple interface, and you can stop the stream at any time.

For further enhancements, you can explore adding more features such as recording the video stream, saving the annotated frames, or integrating with other [computer vision libraries](https://www.ultralytics.com/blog/exploring-vision-ai-frameworks-tensorflow-pytorch-and-opencv).

## FAQ

### How can I set up a real-time object detection application using Streamlit and Ultralytics YOLO26?

Setting up a real-time object detection application with Streamlit and Ultralytics YOLO26 is straightforward. First, ensure you have the Ultralytics Python package installed using:

```bash
pip install ultralytics
```

Then, you can create a basic Streamlit application to run live inference:

!!! example "Streamlit Application"

    === "Python"

        ```python
        from ultralytics import solutions

        inf = solutions.Inference(
            model="yolo26n.pt",  # you can use any model that Ultralytics supports, e.g., YOLO26, or a custom-trained model
        )

        inf.inference()

        # Make sure to run the file using command `streamlit run path/to/file.py`
        ```

    === "CLI"

        ```bash
        yolo solutions inference
        ```

For more details on the practical setup, refer to the [Streamlit Application Code section](#streamlit-application-code) of the documentation.

### What are the main advantages of using Ultralytics YOLO26 with Streamlit for real-time object detection?

Using Ultralytics YOLO26 with Streamlit for real-time object detection offers several advantages:

- **Real-Time Detection**: Achieve high-[accuracy](https://www.ultralytics.com/glossary/accuracy), real-time object detection directly from webcam feeds.
- **User-Friendly Interface**: Streamlit's intuitive interface allows easy use and deployment without extensive technical knowledge.
- **Resource Efficiency**: YOLO26's optimized algorithms ensure high-speed processing with minimal computational resources.

Learn more about these benefits in the [Advantages of Live Inference section](#advantages-of-live-inference).

### How do I deploy a Streamlit object detection application in my web browser?

After coding your Streamlit application integrating Ultralytics YOLO26, you can deploy it by running:

```bash
streamlit run path/to/file.py
```

This command will launch the application in your default web browser, enabling you to select YOLO26 models, set confidence and IoU thresholds, and start real-time object detection with a simple click. For a detailed guide, refer to the [Streamlit Application Code](#streamlit-application-code) section.

### What are some use cases for real-time object detection using Streamlit and Ultralytics YOLO26?

Real-time object detection using Streamlit and Ultralytics YOLO26 can be applied in various sectors:

- **Security**: Real-time monitoring for unauthorized access and [security alarm systems](security-alarm-system.md).
- **Retail**: Customer counting, shelf management, and [inventory tracking](https://www.ultralytics.com/blog/from-shelves-to-sales-exploring-yolov8s-impact-on-inventory-management).
- **Wildlife and Agriculture**: Monitoring animals and crop conditions for [conservation efforts](https://www.ultralytics.com/blog/ai-in-wildlife-conservation).

For more in-depth use cases and examples, explore [Ultralytics Solutions](../solutions/index.md).

### How does Ultralytics YOLO26 compare to other object detection models like YOLOv5 and RCNNs?

Ultralytics YOLO26 provides several enhancements over prior models like YOLOv5 and R-CNN-family detectors:

- **Higher Speed and Accuracy**: Improved performance for real-time applications.
- **Ease of Use**: Simplified interfaces and deployment.
- **Resource Efficiency**: Optimized for better speed with minimal computational requirements.

For a detailed comparison, check the [Ultralytics YOLO26 documentation](../models/yolo26.md) and our [comparison of the best object detection models](https://www.ultralytics.com/blog/best-object-detection-models).
