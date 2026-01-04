---
comments: true
description: Learn how to set up a real-time object detection application using Streamlit and Ultralytics YOLO11. Follow this step-by-step guide to implement webcam-based object detection.
keywords: Streamlit, YOLO11, Real-time Object Detection, Streamlit Application, YOLO11 Streamlit Tutorial, Webcam Object Detection
---

# Live Inference with Streamlit Application using Ultralytics YOLO11

## Introduction

Streamlit makes it simple to build and deploy interactive web applications. Combining this with Ultralytics YOLO11 allows for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and analysis directly in your browser. YOLO11's high accuracy and speed ensure seamless performance for live video streams, making it ideal for applications in security, retail, and beyond.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/N8TxB43y-xM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Use Streamlit with Ultralytics for Real-Time <a href="https://www.ultralytics.com/glossary/computer-vision-cv">Computer Vision</a> in Your Browser
</p>

|                                                                Aquaculture                                                                 |                                                           Animal Husbandry                                                           |
| :----------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| ![Fish Detection using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/fish-detection-ultralytics-yolov8.avif) | ![Animals Detection using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/animals-detection-yolov8.avif) |
|                                                  Fish Detection using Ultralytics YOLO11                                                   |                                              Animals Detection using Ultralytics YOLO11                                              |

## Advantages of Live Inference

- **Seamless Real-Time Object Detection**: Streamlit combined with YOLO11 enables real-time object detection directly from your webcam feed. This allows for immediate analysis and insights, making it ideal for [applications requiring instant feedback](https://docs.ultralytics.com/modes/predict/).
- **User-Friendly Deployment**: Streamlit's interactive interface makes it easy to deploy and use the application without extensive technical knowledge. Users can start live inference with a simple click, enhancing accessibility and usability.
- **Efficient Resource Utilization**: YOLO11's optimized algorithms ensure high-speed processing with minimal computational resources. This efficiency allows for smooth and reliable webcam inference even on standard hardware, making advanced [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) accessible to a wider audience.

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
        ```

        These commands launch the default Streamlit interface that ships with Ultralytics. Use `yolo solutions inference --help` to view additional flags such as `source`, `conf`, or `persist` if you want to customize the experience without editing Python code.

    === "Python"

        ```python
        from ultralytics import solutions

        inf = solutions.Inference(
            model="yolo11n.pt",  # you can use any model that Ultralytics support, i.e. YOLO11, or custom trained model
        )

        inf.inference()

        # Make sure to run the file using command `streamlit run path/to/file.py`
        ```

This will launch the Streamlit application in your default web browser. You will see the main title, subtitle, and the sidebar with configuration options. Select your desired YOLO11 model, set the confidence and [NMS thresholds](https://www.ultralytics.com/glossary/non-maximum-suppression-nms), and click the "Start" button to begin the real-time object detection.

## How It Works

Under the hood, the Streamlit application uses the [Ultralytics solutions module](https://docs.ultralytics.com/reference/solutions/streamlit_inference/) to create an interactive interface. When you start the inference, the application:

1. Captures video from your webcam or uploaded video file
2. Processes each frame through the YOLO11 model
3. Applies object detection with your specified confidence and IoU thresholds
4. Displays both the original and annotated frames in real-time
5. Optionally enables object tracking if selected

The application provides a clean, user-friendly interface with controls to adjust model parameters and start/stop inference at any time.

## Conclusion

By following this guide, you have successfully created a real-time object detection application using Streamlit and Ultralytics YOLO11. This application allows you to experience the power of YOLO11 in detecting objects through your webcam, with a user-friendly interface and the ability to stop the video stream at any time.

For further enhancements, you can explore adding more features such as recording the video stream, saving the annotated frames, or integrating with other [computer vision libraries](https://www.ultralytics.com/blog/exploring-vision-ai-frameworks-tensorflow-pytorch-and-opencv).

## Share Your Thoughts with the Community

Engage with the community to learn more, troubleshoot issues, and share your projects:

### Where to Find Help and Support

- **GitHub Issues:** Visit the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/issues) to raise questions, report bugs, and suggest features.
- **Ultralytics Discord Server:** Join the [Ultralytics Discord server](https://discord.com/invite/ultralytics) to connect with other users and developers, get support, share knowledge, and brainstorm ideas.

### Official Documentation

- **Ultralytics YOLO11 Documentation:** Refer to the [official YOLO11 documentation](https://docs.ultralytics.com/) for comprehensive guides and insights on various computer vision tasks and projects.

## FAQ

### How can I set up a real-time object detection application using Streamlit and Ultralytics YOLO11?

Setting up a real-time object detection application with Streamlit and Ultralytics YOLO11 is straightforward. First, ensure you have the Ultralytics Python package installed using:

```bash
pip install ultralytics
```

Then, you can create a basic Streamlit application to run live inference:

!!! example "Streamlit Application"

    === "Python"

        ```python
        from ultralytics import solutions

        inf = solutions.Inference(
            model="yolo11n.pt",  # you can use any model that Ultralytics support, i.e. YOLO11, YOLOv10
        )

        inf.inference()

        # Make sure to run the file using command `streamlit run path/to/file.py`
        ```

    === "CLI"

        ```bash
        yolo solutions inference
        ```

For more details on the practical setup, refer to the [Streamlit Application Code section](#streamlit-application-code) of the documentation.

### What are the main advantages of using Ultralytics YOLO11 with Streamlit for real-time object detection?

Using Ultralytics YOLO11 with Streamlit for real-time object detection offers several advantages:

- **Seamless Real-Time Detection**: Achieve high-[accuracy](https://www.ultralytics.com/glossary/accuracy), real-time object detection directly from webcam feeds.
- **User-Friendly Interface**: Streamlit's intuitive interface allows easy use and deployment without extensive technical knowledge.
- **Resource Efficiency**: YOLO11's optimized algorithms ensure high-speed processing with minimal computational resources.

Learn more about these benefits in the [Advantages of Live Inference section](#advantages-of-live-inference).

### How do I deploy a Streamlit object detection application in my web browser?

After coding your Streamlit application integrating Ultralytics YOLO11, you can deploy it by running:

```bash
streamlit run path/to/file.py
```

This command will launch the application in your default web browser, enabling you to select YOLO11 models, set confidence and NMS thresholds, and start real-time object detection with a simple click. For a detailed guide, refer to the [Streamlit Application Code](#streamlit-application-code) section.

### What are some use cases for real-time object detection using Streamlit and Ultralytics YOLO11?

Real-time object detection using Streamlit and Ultralytics YOLO11 can be applied in various sectors:

- **Security**: Real-time monitoring for unauthorized access and [security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system/).
- **Retail**: Customer counting, shelf management, and [inventory tracking](https://www.ultralytics.com/blog/from-shelves-to-sales-exploring-yolov8s-impact-on-inventory-management).
- **Wildlife and Agriculture**: Monitoring animals and crop conditions for [conservation efforts](https://www.ultralytics.com/blog/ai-in-wildlife-conservation).

For more in-depth use cases and examples, explore [Ultralytics Solutions](https://docs.ultralytics.com/solutions/).

### How does Ultralytics YOLO11 compare to other object detection models like YOLOv5 and RCNNs?

Ultralytics YOLO11 provides several enhancements over prior models like YOLOv5 and RCNNs:

- **Higher Speed and Accuracy**: Improved performance for real-time applications.
- **Ease of Use**: Simplified interfaces and deployment.
- **Resource Efficiency**: Optimized for better speed with minimal computational requirements.

For a comprehensive comparison, check [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/) and related blog posts discussing model performance.
