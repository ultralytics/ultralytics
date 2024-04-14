---
comments: true
description: Security Alarm System Project Using Ultralytics YOLOv8. Learn How to implement a Security Alarm System Using ultralytics YOLOv8
keywords: Object Detection, Security Alarm, Object Tracking, YOLOv8, Computer Vision Projects
---

# Security Alarm System Project Using Ultralytics YOLOv8

<img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/f4e4a613-fb25-4bd0-9ec5-78352ddb62bd" alt="Security Alarm System">

The Security Alarm System Project utilizing Ultralytics YOLOv8 integrates advanced computer vision capabilities to enhance security measures. YOLOv8, developed by Ultralytics, provides real-time object detection, allowing the system to identify and respond to potential security threats promptly. This project offers several advantages:

- **Real-time Detection:** YOLOv8's efficiency enables the Security Alarm System to detect and respond to security incidents in real-time, minimizing response time.
- **Accuracy:** YOLOv8 is known for its accuracy in object detection, reducing false positives and enhancing the reliability of the security alarm system.
- **Integration Capabilities:** The project can be seamlessly integrated with existing security infrastructure, providing an upgraded layer of intelligent surveillance.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_1CmwUzoxY4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Security Alarm System Project with Ultralytics YOLOv8 Object Detection
</p>

### Code

#### Import Libraries

```python
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
```

#### Set up the parameters of the message

???+ tip "Note"

    App Password Generation is necessary

- Navigate to [App Password Generator](https://myaccount.google.com/apppasswords), designate an app name such as "security project," and obtain a 16-digit password. Copy this password and paste it into the designated password field as instructed.

```python
password = ""
from_email = ""  # must match the email used to generate the password
to_email = ""  # receiver email
```

#### Server creation and authentication

```python
server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)
```

#### Email Send Function

```python
def send_email(to_email, from_email, object_detected=1):
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "Security Alert"
    # Add in the message body
    message_body = f'ALERT - {object_detected} objects has been detected!!'

    message.attach(MIMEText(message_body, 'plain'))
    server.sendmail(from_email, to_email, message.as_string())
```

#### Object Detection and Alert Sender

```python
class ObjectDetection:
    def __init__(self, capture_index):
        # default parameters
        self.capture_index = capture_index
        self.email_sent = False

        # model information
        self.model = YOLO("yolov8n.pt")

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device information
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if len(class_ids) > 0:  # Only send email If not sent before
                if not self.email_sent:
                    send_email(to_email, from_email, len(class_ids))
                    self.email_sent = True
            else:
                self.email_sent = False

            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        server.quit()
```

#### Call the Object Detection class and Run the Inference

```python
detector = ObjectDetection(capture_index=0)
detector()
```

That's it! When you execute the code, you'll receive a single notification on your email if any object is detected. The notification is sent immediately, not repeatedly. However, feel free to customize the code to suit your project requirements.

#### Email Received Sample

<img width="256" src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/db79ccc6-aabd-4566-a825-b34e679c90f9" alt="Email Received Sample">
