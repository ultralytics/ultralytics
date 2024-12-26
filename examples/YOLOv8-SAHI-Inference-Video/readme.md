# YOLO11 with SAHI (Inference on Video)

[SAHI](https://docs.ultralytics.com/guides/sahi-tiled-inference/) is designed to optimize object detection algorithms for large-scale and high-resolution imagery. It partitions images into manageable slices, performs object detection on each slice, and then stitches the results back together. This tutorial will guide you through the process of running YOLO11 inference on video files with the aid of SAHI.

## Table of Contents

- [Step 1: Install the Required Libraries](#step-1-install-the-required-libraries)
- [Step 2: Run the Inference with SAHI using Ultralytics YOLO11](#step-2-run-the-inference-with-sahi-using-ultralytics-yolo11)
- [Usage Options](#usage-options)
- [FAQ](#faq)

## Step 1: Install the Required Libraries

Clone the repository, install dependencies and `cd` to this local directory for commands in Step 2.

```bash
# Clone ultralytics repo
git clone https://github.com/ultralytics/ultralytics

# Install dependencies
pip install -U sahi ultralytics

# cd to local directory
cd ultralytics/examples/YOLOv8-SAHI-Inference-Video
```

## Step 2: Run the Inference with SAHI using Ultralytics YOLO11

Here are the basic commands for running the inference:

```bash
#if you want to save results
python yolov8_sahi.py --source "path/to/video.mp4" --save-img

#if you want to change model file
python yolov8_sahi.py --source "path/to/video.mp4" --save-img --weights "yolo11n.pt"
```

## Usage Options

- `--source`: Specifies the path to the video file you want to run inference on.
- `--save-img`: Flag to save the detection results as images.
- `--weights`: Specifies a different YOLO11 model file (e.g., `yolo11n.pt`, `yolov8s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`).

## FAQ

**1. What is SAHI?**

SAHI stands for Slicing Aided Hyper Inference. It is a library designed to optimize object detection algorithms for large-scale and high-resolution images. The library source code is available on [GitHub](https://github.com/obss/sahi).

**2. Why use SAHI with YOLO11?**

SAHI can handle large-scale images by slicing them into smaller, more manageable sizes without compromising the detection quality. This makes it a great companion to YOLO11, especially when working with high-resolution videos.

**3. How do I debug issues?**

You can add the `--debug` flag to your command to print out more information during inference:

```bash
python yolov8_sahi.py --source "path to video file" --debug
```

**4. Can I use other YOLO versions?**

Yes, you can specify different YOLO model weights using the `--weights` option.

**5. Where can I find more information?**

For a full guide to YOLO11 with SAHI see [https://docs.ultralytics.com/guides/sahi-tiled-inference](https://docs.ultralytics.com/guides/sahi-tiled-inference/).
