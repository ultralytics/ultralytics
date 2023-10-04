
# Quick Start with Raspberry Pi, Pi Camera, and libcamera

Created by Daan Eeltink for Kashmir World Foundation

At Kashmir World Foundation we teach students, hobbyists, and professionals from around the world how to use YOLO to protect endangered species. Most have little or no background in computer science, so we need to move quickly to capture their interests and get them on the path to success. This guide is intended to do just that: get you up and running YOLO on a Raspberry Pi with Pi Camera in less than 30 minutes!

On your own, chances are you would run into a few issues along the way delaying the excitement of seeing YOLO pop up on your screen. This guide avoids them so you will be confident that your hardware is working and capable of running YOLO. Then as you move forward with your own projects and run into an issue, you can always return to a comforting working state in 15 minutes or less.


## Prerequisites

This Quick Start works with a Raspberry Pi 3 or Raspberry Pi 4 computer. You also will also need a Pi Camera. Connect the Pi Camera to the Raspberry Pi using a CSI cable, then install the 64-bit Raspberry Pi Operating System.

Verify that the camera is working:

    libcamera-hello

A screen should pop up displaying video from your camera!


## YOLOv5 or YOLOv8

This guide gives you some options depending on your setup. The first option is whether you want to start with YOLOv5 or YOLOv8. At Kashmir World Foundation we usually begin with YOLOv5 and later proceed with YOLOv8. You can do the same with this guide, or you can jump ahead, but beware of shortcuts. The purpose of the guide is not just to get you up and running in less than 30 minutes, it is to give you a strong enough foundation from which to embark on your own YOLO adventures.


## Raspberry Pi 3 or Raspberry Pi 4

There also are some differences between proceeding with a Raspberry Pi 3 or Raspberry Pi 4. Be careful to follow the instructions that are specific to your version of Raspberry Pi.


## Quick Start with YOLOv5

Several tutorials are available for using YOLOv5 on Raspberry Pi single board computers with an official Raspberry Pi camera and what are now referred to as the “legacy camera stack”. With the release of Raspberry Pi OS Bullseye, the default camera stack is now libcamera. This guide explains how to deploy a trained YOLOv5 model on a Raspberry Pi 3 or Pi 4 running the 64 bit version of the operating system with the libcamera camera stack. You already verified that your camera is working, so let's get started!

### Install Necessary Packages
    1. Make sure the Raspberry Pi is up-to-date. It is good practice to run the following commands before installing new software:
        sudo apt-get update 
        sudo apt-get upgrade -y
        sudo apt-get autoremove -y
    2. Now clone the YOLOv5 repository:
        cd ~
        git clone https://github.com/Ultralytics/yolov5.git
    3. Now install the required dependencies
        cd ~/yolov5
        pip3 install -r requirements.txt
    4. If you have a Raspberry Pi 4, you are good to SKIP this step, but if you have a Raspberry Pi 3, you need to install the correct version of PyTorch and Torchvision. (At the time of writing, PyTorch 1.13.0 and Torchvision 0.14.0 or beyond were not supported on the Raspberry Pi 3. First install the versions specified below. You can try more recent versions after you have verified YOLO to be running properly.)
        cd ~/yolov5
        pip3 uninstall torch
        pip3 uninstall torchvision
        pip3 install torch==1.11.0
        pip3 install torchvision==0.12.0

### Modify detect.py
By default, detect.py doesn’t allow TCP streams to be used as a source and fails when used via SSH or the Raspberry Pi Command Line Interface. To fix this, make two minor modifications to detect.py

    1. Open detect.py and find the ‘is_url’ line
        cd ~/yolov5
        sudo nano detect.py
        CTRL + W --> is_url --> ENTER 
    2. Add TCP streams as an accepted url-format
        is_url = source.lower().startswith((‘rtsp://’, ‘rtmp://’, ‘http://’, ‘https://’, ‘tcp://’))
    3. Find the line that says ‘view_img = check_imshow(warn=True)’
        CTRL + W --> view_img = check_imshow(warn=True) --> ENTER
    4. Comment out this line
        #view_img = check_imshow(warn=True)
    5. Save the modifications and close detect.py
        CTRL + O --> ENTER --> CTRL + X

### Initiate a TCP stream with Libcamera
There are many parameters you could use with libcamera-vid. Start with the parameters below, then experiment with your own parameters.

    1.  libcamera-vid -n -t 0 --width 1280 --height 960 --framerate 1 --inline --listen -o tcp://127.0.0.1:8888
        Note: Keep this terminal or SSH session running during the next section!

### Perform YOLOv5 inference on the TCP stream (as of 2023.09.19, see docs.ultralytics.com for more current instructions)

    1.  Run detect.py with the TCP stream as its source
	 cd ~/yolov5
	 python3 detect.py --source=tcp://127.0.0.1:8888
        Note: by commenting out ‘view_img = check_imshow(warn=True)’ we made sure detect.py no longer displays its source + predicted bounding boxes to enable inference via SSH or the Command Line Interface. To re-enable this feature, we added the --view-img argument.

### Now sit back and wait while your Raspberry Pi begins running YOLOv5 on your video stream. Give it some fun things to process. Walk by the camera. Do it again carrying something of interest, and feel free to adjust parameters in your video stream and in detect.py


## Quick Start with YOLOv8

YOLOv8 has a substantially different method of installation. The following guide will get you up and running in less than 30 minutes. It will also provide a firm basis for further exploration of YOLOv8. Start with a fresh 64-bit Raspberry Pi operating system. You already verified that your camera is working, so let's get started!

### Install Necessary Packages
    1. Make sure the Raspberry Pi is up-to-date. It is good practice to run the following commands before installing new software:
        sudo apt-get update 
        sudo apt-get upgrade -y
        sudo apt-get autoremove -y
    2. Install YOLOv8
        pip3 install ultralytics
    3. Reboot the Raspberry Pi
        sudo reboot
    4. Locate the Ultralytics package folder
       Note: On the Raspberry Pi, this folder can be found at /home/pi/.local/lib/pythonX.X/site-packages
       Note: To display hidden files in your Raspberry Pi system through the terminal, use ls -a
       Note: The output will display the name of files and directories; the ones with a “.” at the start of their names are all the hidden content. Blue colored are the directories and white ones are the files
    5. If you have a Raspberry Pi 4, you are good to SKIP this step, but if you have a Raspberry Pi 3, you need to install the correct version of PyTorch and Torchvision. (At the time of writing, PyTorch 1.13.0 and Torchvision 0.14.0 or beyond were not supported on the Raspberry Pi 3. First install the versions specified below. You can try more recent versions after you have verified YOLO to be running properly.)
        pip3 uninstall torch
        pip3 uninstall torchvision
        pip3 install torch==1.11.0
        pip3 install torchvision==0.12.0

### Modify build.py
By default, YOLOv8 doesn’t allow TCP streams to be used as a source. To fix this, make a minor modifications to the Ultralytics package.

    1. Navigate to the Ultralytics package folder, which you have already located
       	cd /home/pi/.local/lib/pythonX.X/site-packages/ultralytics
    2. Open build.py and locate the ‘is_url’ line
        sudo nano data/build.py
        CTRL + W --> is_url --> ENTER 
    3. Add TCP streams as an accepted url-format
        is_url = source.lower().startswith((‘rtsp://’, ‘rtmp://’, ‘http://’, ‘https://’, ‘tcp://’))
    4. Save the modifications and close build.py
        CTRL + O --> ENTER --> CTRL + X


### Initiate a TCP stream with Libcamera
There are many parameters you could use with libcamera-vid. Start with the parameters below, then experiment with your own parameters.

    1.  libcamera-vid -n -t 0 --width 1280 --height 960 --framerate 1 --inline --listen -o tcp://127.0.0.1:8888
        Note: Keep this terminal or SSH session running during the next step
        
### Perform YOLOv8 inference on the TCP stream (as of 2023.09.19, see docs.ultralytics.com for more current instructions)
    1. To use YOLOv8 from the command line:
        yolo predict model=yolov8n.pt source=tcp://127.0.0.1:8888

    2. To display its source + predicted bounding boxes, run yolo with the --show=true argument:
        yolo predict model=yolov8n.pt source=tcp://127.0.0.1:8888 show=true

    3. To use YOLOv8 from within a Python environment:
        from ultralytics import YOLO
        model = YOLO(‘yolov8n.pt’)
        results = model(‘tcp://127.0.0.1:8888’, stream=True)
        while True:
          for result in results:
            boxes = result.boxes
            probs = result.probs

## Next Steps

Congratulations on having completed this Quick Start. There is much more to learn here on Ultralytics.com or at KashmirWorldFoundation.org. Find your passion, and then stay on the correct path to achieving your goals. There are many places along the way where you can get help, but beware of shortcuts!
