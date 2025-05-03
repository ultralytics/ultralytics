---
comments: true
description: Enhance your security with real-time object detection using Ultralytics YOLO11. Reduce false positives and integrate seamlessly with existing systems.
keywords: YOLO11, Security Alarm System, real-time object detection, Ultralytics, computer vision, integration, false positives
---

# Security Alarm System Project Using Ultralytics YOLO11

<img src="https://github.com/ultralytics/docs/releases/download/0/security-alarm-system-ultralytics-yolov8.avif" alt="Security Alarm System">

The Security Alarm System Project utilizing Ultralytics YOLO11 integrates advanced [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) capabilities to enhance security measures. YOLO11, developed by Ultralytics, provides real-time [object detection](https://www.ultralytics.com/glossary/object-detection), allowing the system to identify and respond to potential security threats promptly. This project offers several advantages:

- **Real-time Detection:** YOLO11's efficiency enables the Security Alarm System to detect and respond to security incidents in real-time, minimizing response time.
- **[Accuracy](https://www.ultralytics.com/glossary/accuracy):** YOLO11 is known for its accuracy in object detection, reducing false positives and enhancing the reliability of the security alarm system.
- **Integration Capabilities:** The project can be seamlessly integrated with existing security infrastructure, providing an upgraded layer of intelligent surveillance.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/DTjtBnSK2fY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Security Alarm System with Ultralytics YOLO11 + Solutions <a href="https://www.ultralytics.com/glossary/object-detection">Object Detection</a>
</p>
 
???+ note

    App Password Generation is necessary

- Navigate to [App Password Generator](https://myaccount.google.com/apppasswords), designate an app name such as "security project," and obtain a 16-digit password. Copy this password and paste it into the designated `password` field in the code below.

!!! example "Security Alarm System using Ultralytics YOLO"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        from_email = "abc@gmail.com"  # the sender email address
        password = "---- ---- ---- ----"  # 16-digits password generated via: https://myaccount.google.com/apppasswords
        to_email = "xyz@gmail.com"  # the receiver email address

        # Initialize security alarm object
        securityalarm = solutions.SecurityAlarm(
            show=True,  # display the output
            model="yolo11n.pt",  # i.e. yolo11s.pt, yolo11m.pt
            records=1,  # total detections count to send an email
        )

        securityalarm.authenticate(from_email, password, to_email)  # authenticate the email server

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = securityalarm(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

That's it! When you execute the code, you'll receive a single notification on your email if any object is detected. The notification is sent immediately, not repeatedly. However, feel free to customize the code to suit your project requirements.

#### Email Received Sample

<img width="256" src="https://github.com/ultralytics/docs/releases/download/0/email-received-sample.avif" alt="Email Received Sample">

### `SecurityAlarm` Arguments

Here's a table with the `SecurityAlarm` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "records"]) }}

The `SecurityAlarm` solution supports a variety of `track` parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization settings are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## How It Works

The Security Alarm System uses [object tracking](https://docs.ultralytics.com/modes/track/) to monitor video feeds and detect potential security threats. When the system detects objects that exceed the specified threshold (set by the `records` parameter), it automatically sends an email notification with an image attachment showing the detected objects.

The system leverages the [SecurityAlarm class](https://docs.ultralytics.com/reference/solutions/security_alarm/) which provides methods to:

1. Process frames and extract object detections
2. Annotate frames with bounding boxes around detected objects
3. Send email notifications when detection thresholds are exceeded

This implementation is ideal for home security, retail surveillance, and other monitoring applications where immediate notification of detected objects is critical.

## FAQ

### How does Ultralytics YOLO11 improve the accuracy of a security alarm system?

Ultralytics YOLO11 enhances security alarm systems by delivering high-accuracy, real-time object detection. Its advanced algorithms significantly reduce false positives, ensuring that the system only responds to genuine threats. This increased reliability can be seamlessly integrated with existing security infrastructure, upgrading the overall surveillance quality.

### Can I integrate Ultralytics YOLO11 with my existing security infrastructure?

Yes, Ultralytics YOLO11 can be seamlessly integrated with your existing security infrastructure. The system supports various modes and provides flexibility for customization, allowing you to enhance your existing setup with advanced object detection capabilities. For detailed instructions on integrating YOLO11 in your projects, visit the [integration section](https://docs.ultralytics.com/integrations/).

### What are the storage requirements for running Ultralytics YOLO11?

Running Ultralytics YOLO11 on a standard setup typically requires around 5GB of free disk space. This includes space for storing the YOLO11 model and any additional dependencies. For cloud-based solutions, [Ultralytics HUB](https://docs.ultralytics.com/hub/) offers efficient project management and dataset handling, which can optimize storage needs. Learn more about the [Pro Plan](../hub/pro.md) for enhanced features including extended storage.

### What makes Ultralytics YOLO11 different from other object detection models like Faster R-CNN or SSD?

Ultralytics YOLO11 provides an edge over models like Faster R-CNN or SSD with its real-time detection capabilities and higher accuracy. Its unique architecture allows it to process images much faster without compromising on [precision](https://www.ultralytics.com/glossary/precision), making it ideal for time-sensitive applications like security alarm systems. For a comprehensive comparison of object detection models, you can explore our [guide](https://docs.ultralytics.com/models/).

### How can I reduce the frequency of false positives in my security system using Ultralytics YOLO11?

To reduce false positives, ensure your Ultralytics YOLO11 model is adequately trained with a diverse and well-annotated dataset. Fine-tuning hyperparameters and regularly updating the model with new data can significantly improve detection accuracy. Detailed [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning) techniques can be found in our [hyperparameter tuning guide](../guides/hyperparameter-tuning.md).
