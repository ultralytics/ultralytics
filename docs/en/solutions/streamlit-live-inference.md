---
comments: true
description: Learn how to set up a real-time object detection application using Streamlit and Ultralytics YOLOv8. Follow this step-by-step guide to implement webcam-based object detection.
keywords: Streamlit, YOLOv8, Real-time Object Detection, Streamlit Application, YOLOv8 Streamlit Tutorial, Webcam Object Detection
---

# Live Inference with Streamlit Application using Ultralytics YOLOv8

## Introduction

Streamlit makes it simple to build and deploy interactive web applications. Combining this with Ultralytics YOLOv8 allows for real-time object detection and analysis directly in your browser. YOLOv8 high accuracy and speed ensure seamless performance for live video streams, making it ideal for applications in security, retail, and beyond.

|                                                                   Aquaculture                                                                   |                                                                 Animals husbandry                                                                  |
|:-----------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Fish Detection using Ultralytics YOLOv8](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/ea6d7ece-cded-4db7-b810-1f8433df2c96) | ![Animals Detection using Ultralytics YOLOv8](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/2e1f4781-60ab-4e72-b3e4-726c10cd223c) |
|                                                     Fish Detection using Ultralytics YOLOv8                                                     |                                                     Animals Detection using Ultralytics YOLOv8                                                     |


## Advantages of Live Inference

- **Seamless Real-Time Object Detection**: Streamlit combined with YOLOv8 enables real-time object detection directly from your webcam feed. This allows for immediate analysis and insights, making it ideal for applications requiring instant feedback.
- **User-Friendly Deployment**: Streamlit’s interactive interface makes it easy to deploy and use the application without extensive technical knowledge. Users can start live inference with a simple click, enhancing accessibility and usability.
- **Efficient Resource Utilization**: YOLOv8 optimized algorithm ensure high-speed processing with minimal computational resources. This efficiency allows for smooth and reliable webcam inference even on standard hardware, making advanced computer vision accessible to a wider audience.

## Streamlit Application Code

!!! tip "Ultralytics Installation"

    Before you start building the application, ensure you have the Ultralytics Python Package installed. You can install it using the command **pip install ultralytics**

!!! Example "Streamlit Application"

    === "Python"
        ```Python
        from ultralytics import solutions

        solutions.inference()

        ### Make sure to run the file using command `streamlit run <file-name.py>`
        ```
    
    === "CLI"
        ```bash
        yolo live-inference
        ```
    
This will launch the Streamlit application in your default web browser. You will see the main title, subtitle, and the sidebar with configuration options. Select your desired YOLOv8 model, set the confidence and NMS thresholds, and click the "Start" button to begin the real-time object detection.

## Conclusion

By following this guide, you have successfully created a real-time object detection application using Streamlit and Ultralytics YOLOv8. This application allows you to experience the power of YOLOv8 in detecting objects through your webcam, with a user-friendly interface and the ability to stop the video stream at any time.

For further enhancements, you can explore adding more features such as recording the video stream, saving the annotated frames, or integrating with other computer vision libraries.

## Share Your Thoughts with the Community

Engage with the community to learn more, troubleshoot issues, and share your projects:

### Where to Find Help and Support

- **GitHub Issues:** Visit the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/issues) to raise questions, report bugs, and suggest features.
- **Ultralytics Discord Server:** Join the [Ultralytics Discord server](https://ultralytics.com/discord/) to connect with other users and developers, get support, share knowledge, and brainstorm ideas.

### Official Documentation

- **Ultralytics YOLOv8 Documentation:** Refer to the [official YOLOv8 documentation](https://docs.ultralytics.com/) for comprehensive guides and insights on various computer vision tasks and projects.
