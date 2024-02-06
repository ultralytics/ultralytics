---
comments: true
description: Quick start guide to setting up YOLO on a Raspberry Pi with a Pi Camera using the libcamera stack. Detailed comparison between Raspberry Pi 3, 4 and 5 models.
keywords: Ultralytics, YOLO, Raspberry Pi, Pi Camera, libcamera, quick start guide, Raspberry Pi 4 vs Raspberry Pi 5, YOLO on Raspberry Pi, hardware setup, machine learning, AI
---

# Quick Start Guide: Raspberry Pi and Pi Camera with YOLOv5 and YOLOv8

This comprehensive guide aims to expedite your journey with YOLO object detection models on a [Raspberry Pi](https://www.raspberrypi.com/) using a [Pi Camera](https://www.raspberrypi.com/products/camera-module-v2/). Whether you're a student, hobbyist, or a professional, this guide is designed to get you up and running in less than 30 minutes. The instructions here are rigorously tested to minimize setup issues, allowing you to focus on utilizing YOLO for your specific projects.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yul4gq_LrOI"
    title="Introducing Raspberry Pi 5" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Raspberry Pi 5 updates and improvements.
</p>

## Prerequisites

- Raspberry Pi 3, 4 or 5
- Pi Camera
- 64-bit Raspberry Pi Operating System

Connect the Pi Camera to your Raspberry Pi via a CSI cable and install the 64-bit Raspberry Pi Operating System. Verify your camera with the following command:

```bash
libcamera-hello
```

You should see a video feed from your camera.

## Choose Your YOLO Version: YOLOv5 or YOLOv8

This guide offers you the flexibility to start with either [YOLOv5](https://github.com/ultralytics/yolov5) or [YOLOv8](https://github.com/ultralytics/ultralytics). Both versions have their unique advantages and use-cases. The choice is yours, but remember, the guide's aim is not just quick setup but also a robust foundation for your future work in object detection.

## Hardware Specifics: At a Glance

To assist you in making an informed hardware decision, we've summarized the key hardware specifics of Raspberry Pi 3, 4, and 5 in the table below:

| Feature                    | Raspberry Pi 3                                                                           | Raspberry Pi 4                                                                           | Raspberry Pi 5                                                       |
|----------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| **CPU**                    | 1.2GHz Quad-Core ARM Cortex-A53                                                          | 1.5GHz Quad-core 64-bit ARM Cortex-A72                                                   | 2.4GHz Quad-core 64-bit Arm Cortex-A76                               |
| **RAM**                    | 1GB LPDDR2                                                                               | 2GB, 4GB or 8GB LPDDR4                                                                   | *Details not yet available*                                          |
| **USB Ports**              | 4 x USB 2.0                                                                              | 2 x USB 2.0, 2 x USB 3.0                                                                 | 2 x USB 3.0, 2 x USB 2.0                                             |
| **Network**                | Ethernet & Wi-Fi 802.11n                                                                 | Gigabit Ethernet & Wi-Fi 802.11ac                                                        | Gigabit Ethernet with PoE+ support, Dual-band 802.11ac Wi-Fi®        |
| **Performance**            | Slower, may require lighter YOLO models                                                  | Faster, can run complex YOLO models                                                      | *Details not yet available*                                          |
| **Power Requirement**      | 2.5A power supply                                                                        | 3.0A USB-C power supply                                                                  | *Details not yet available*                                          |
| **Official Documentation** | [Link](https://www.raspberrypi.org/documentation/hardware/raspberrypi/bcm2837/README.md) | [Link](https://www.raspberrypi.org/documentation/hardware/raspberrypi/bcm2711/README.md) | [Link](https://www.raspberrypi.com/news/introducing-raspberry-pi-5/) |

Please make sure to follow the instructions specific to your Raspberry Pi model to ensure a smooth setup process.

## Quick Start with YOLOv5

This section outlines how to set up YOLOv5 on a Raspberry Pi with a Pi Camera. These steps are designed to be compatible with the libcamera camera stack introduced in Raspberry Pi OS Bullseye.

### Install Necessary Packages

1. Update the Raspberry Pi:

    ```bash
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get autoremove -y
    ```

2. Clone the YOLOv5 repository:

    ```bash
    cd ~
    git clone https://github.com/Ultralytics/yolov5.git
    ```

3. Install the required dependencies:

    ```bash
    cd ~/yolov5
    pip3 install -r requirements.txt
    ```

4. For Raspberry Pi 3, install compatible versions of PyTorch and Torchvision (skip for Raspberry Pi 4):

    ```bash
    pip3 uninstall torch torchvision
    pip3 install torch==1.11.0 torchvision==0.12.0
    ```

### Modify `detect.py`

To enable TCP streams via SSH or the CLI, minor modifications are needed in `detect.py`.

1. Open `detect.py`:

    ```bash
    sudo nano ~/yolov5/detect.py
    ```

2. Find and modify the `is_url` line to accept TCP streams:

    ```python
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://', 'tcp://'))
    ```

3. Comment out the `view_img` line:

    ```python
    # view_img = check_imshow(warn=True)
    ```

4. Save and exit:

    ```bash
    CTRL + O -> ENTER -> CTRL + X
    ```

### Initiate TCP Stream with Libcamera

1. Start the TCP stream:

    ```bash
    libcamera-vid -n -t 0 --width 1280 --height 960 --framerate 1 --inline --listen -o tcp://127.0.0.1:8888
    ```

Keep this terminal session running for the next steps.

### Perform YOLOv5 Inference

1. Run the YOLOv5 detection:

    ```bash
    cd ~/yolov5
    python3 detect.py --source=tcp://127.0.0.1:8888
    ```

## Quick Start with YOLOv8

Follow this section if you are interested in setting up YOLOv8 instead. The steps are quite similar but are tailored for YOLOv8's specific needs.

### Install Necessary Packages

1. Update the Raspberry Pi:

    ```bash
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get autoremove -y
    ```

2. Install the `ultralytics` Python package:

    ```bash
    pip3 install ultralytics
    ```

3. Reboot:

    ```bash
    sudo reboot
    ```

### Initiate TCP Stream with Libcamera

1. Start the TCP stream:

    ```bash
    libcamera-vid -n -t 0 --width 1280 --height 960 --framerate 1 --inline --listen -o tcp://127.0.0.1:8888
    ```

### Perform YOLOv8 Inference

To perform inference with YOLOv8, you can use the following Python code snippet:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('tcp://127.0.0.1:8888', stream=True)

while True:
    for result in results:
        boxes = result.boxes
        probs = result.probs
```

## Next Steps

Congratulations on successfully setting up YOLO on your Raspberry Pi! For further learning and support, visit [Ultralytics](https://ultralytics.com/) and [Kashmir World Foundation](https://www.kashmirworldfoundation.org/).

## Acknowledgements and Citations

This guide was initially created by Daan Eeltink for Kashmir World Foundation, an organization dedicated to the use of YOLO for the conservation of endangered species. We acknowledge their pioneering work and educational focus in the realm of object detection technologies.

For more information about Kashmir World Foundation's activities, you can visit their [website](https://www.kashmirworldfoundation.org/).
