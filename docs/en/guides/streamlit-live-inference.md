---
comments: true
description: Learn how to set up a real-time object detection application using Streamlit and Ultralytics YOLO11. Follow this step-by-step guide to implement webcam-based object detection.
keywords: Streamlit, YOLO11, Real-time Object Detection, Streamlit Application, YOLO11 Streamlit Tutorial, Webcam Object Detection
---

# Live Inference with Streamlit Application using Ultralytics YOLO11

## Introduction

Streamlit makes it simple to build and deploy interactive web applications. Combining this with Ultralytics YOLO11 allows for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and analysis directly in your browser. YOLO11 high accuracy and speed ensure seamless performance for live video streams, making it ideal for applications in security, retail, and beyond.

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

|                                                                Aquaculture                                                                 |                                                          Animals husbandry                                                           |
| :----------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| ![Fish Detection using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/fish-detection-ultralytics-yolov8.avif) | ![Animals Detection using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/animals-detection-yolov8.avif) |
|                                                  Fish Detection using Ultralytics YOLO11                                                   |                                              Animals Detection using Ultralytics YOLO11                                              |

## Advantages of Live Inference

- **Seamless Real-Time Object Detection**: Streamlit combined with YOLO11 enables real-time object detection directly from your webcam feed. This allows for immediate analysis and insights, making it ideal for applications requiring instant feedback.
- **User-Friendly Deployment**: Streamlit's interactive interface makes it easy to deploy and use the application without extensive technical knowledge. Users can start live inference with a simple click, enhancing accessibility and usability.
- **Efficient Resource Utilization**: YOLO11 optimized algorithm ensure high-speed processing with minimal computational resources. This efficiency allows for smooth and reliable webcam inference even on standard hardware, making advanced computer vision accessible to a wider audience.

## Streamlit Application Code

!!! tip "Ultralytics Installation"

    Before you start building the application, ensure you have the Ultralytics Python Package installed. You can install it using the command **pip install ultralytics**

!!! example "Streamlit Application"

    === "CLI"

        ```bash
        yolo streamlit-predict
        ```

    === "Python"

        ```python
        from ultralytics import solutions

        solutions.inference()

        ### Make sure to run the file using command `streamlit run <file-name.py>`
        ```

This will launch the Streamlit application in your default web browser. You will see the main title, subtitle, and the sidebar with configuration options. Select your desired YOLO11 model, set the confidence and NMS thresholds, and click the "Start" button to begin the real-time object detection.

You can optionally supply a specific model in Python:

!!! example "Streamlit Application with a custom model"

    === "Python"

        ```python
        from ultralytics import solutions

        # Pass a model as an argument
        solutions.inference(model="path/to/model.pt")

        ### Make sure to run the file using command `streamlit run <file-name.py>`
        ```

## Conclusion

By following this guide, you have successfully created a real-time object detection application using Streamlit and Ultralytics YOLO11. This application allows you to experience the power of YOLO11 in detecting objects through your webcam, with a user-friendly interface and the ability to stop the video stream at any time.

For further enhancements, you can explore adding more features such as recording the video stream, saving the annotated frames, or integrating with other computer vision libraries.

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

        solutions.inference()

        ### Make sure to run the file using command `streamlit run <file-name.py>`
        ```

    === "CLI"

        ```bash
        yolo streamlit-predict
        ```

For more details on the practical setup, refer to the [Streamlit Application Code section](#streamlit-application-code) of the documentation.

### What are the main advantages of using Ultralytics YOLO11 with Streamlit for real-time object detection?

Using Ultralytics YOLO11 with Streamlit for real-time object detection offers several advantages:

- **Seamless Real-Time Detection**: Achieve high-[accuracy](https://www.ultralytics.com/glossary/accuracy), real-time object detection directly from webcam feeds.
- **User-Friendly Interface**: Streamlit's intuitive interface allows easy use and deployment without extensive technical knowledge.
- **Resource Efficiency**: YOLO11's optimized algorithms ensure high-speed processing with minimal computational resources.

Discover more about these advantages [here](#advantages-of-live-inference).

### How do I deploy a Streamlit object detection application in my web browser?

After coding your Streamlit application integrating Ultralytics YOLO11, you can deploy it by running:

```bash
streamlit run <file-name.py>
```

This command will launch the application in your default web browser, enabling you to select YOLO11 models, set confidence, and NMS thresholds, and start real-time object detection with a simple click. For a detailed guide, refer to the [Streamlit Application Code](#streamlit-application-code) section.

### What are some use cases for real-time object detection using Streamlit and Ultralytics YOLO11?

Real-time object detection using Streamlit and Ultralytics YOLO11 can be applied in various sectors:

- **Security**: Real-time monitoring for unauthorized access.
- **Retail**: Customer counting, shelf management, and more.
- **Wildlife and Agriculture**: Monitoring animals and crop conditions.

For more in-depth use cases and examples, explore [Ultralytics Solutions](https://docs.ultralytics.com/solutions/).

### How does Ultralytics YOLO11 compare to other object detection models like YOLOv5 and RCNNs?

Ultralytics YOLO11 provides several enhancements over prior models like YOLOv5 and RCNNs:

- **Higher Speed and Accuracy**: Improved performance for real-time applications.
- **Ease of Use**: Simplified interfaces and deployment.
- **Resource Efficiency**: Optimized for better speed with minimal computational requirements.

For a comprehensive comparison, check [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolov8/) and related blog posts discussing model performance.
