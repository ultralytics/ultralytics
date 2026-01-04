# YOLO11 with SAHI for Video Inference

[Slicing Aided Hyper Inference (SAHI)](https://github.com/obss/sahi) is a powerful technique designed to optimize [object detection](https://en.wikipedia.org/wiki/Object_detection) algorithms, particularly for large-scale and [high-resolution imagery](https://en.wikipedia.org/wiki/Image_resolution). It works by partitioning images or video frames into manageable slices, performing detection on each slice using models like [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), and then intelligently merging the results. This approach significantly improves detection accuracy for small objects and maintains performance on high-resolution inputs.

This tutorial guides you through running Ultralytics YOLO11 inference on video files using the SAHI library for enhanced detection capabilities. For a detailed guide on using SAHI with Ultralytics models, see the [SAHI Tiled Inference guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/).

## 📋 Table of Contents

- [Step 1: Install Required Libraries](#-step-1-install-required-libraries)
- [Step 2: Run Inference with SAHI using Ultralytics YOLO11](#-step-2-run-inference-with-sahi-using-ultralytics-yolo11)
- [Usage Options](#-usage-options)
- [Contribute](#-contribute)

## ⚙️ Step 1: Install Required Libraries

First, clone the [Ultralytics repository](https://github.com/ultralytics/ultralytics) to access the example script. Then, install the necessary [Python](https://www.python.org/) packages, including `sahi` and `ultralytics`, using [pip](https://pip.pypa.io/en/stable/). Finally, navigate into the example directory.

```bash
# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Install dependencies
# Ensure you have Python 3.8 or later installed
pip install -U sahi ultralytics opencv-python

# Change directory to the example folder
cd ultralytics/examples/YOLOv8-SAHI-Inference-Video
```

## 🚀 Step 2: Run Inference with SAHI using Ultralytics YOLO11

Once the setup is complete, you can run inference on your video file. The provided script `yolov8_sahi.py` leverages SAHI for tiled inference with a specified YOLO11 model.

Execute the script using the command line, specifying the path to your video file. You can also choose different YOLO11 model weights.

```bash
# Run inference and save the output video with bounding boxes
python yolov8_sahi.py --source "path/to/your/video.mp4" --save-img

# Run inference using a specific YOLO11 model (e.g., yolo11n.pt) and save results
python yolov8_sahi.py --source "path/to/your/video.mp4" --save-img --weights "yolo11n.pt"

# Run inference with smaller slices for potentially better small object detection
python yolov8_sahi.py --source "path/to/your/video.mp4" --save-img --slice-height 512 --slice-width 512
```

This script processes the video frame by frame, applying SAHI's slicing and inference logic before stitching the detections back onto the original frame dimensions. The output, annotated images with detections, will be saved in the `runs/detect/predict` directory. Learn more about prediction with Ultralytics models in the [Predict mode documentation](https://docs.ultralytics.com/modes/predict/).

## 🛠️ Usage Options

The script `yolov8_sahi.py` accepts several command-line arguments to customize the inference process:

- `--source`: **Required**. Path to the input video file (e.g., `"../path/to/video.mp4"`).
- `--weights`: Optional. Path to the YOLO11 model weights file (e.g., `"yolo11n.pt"`, `"yolo11s.pt"`). Defaults to `"yolo11n.pt"`. You can download various models or use your custom-trained ones. See [Ultralytics YOLO models](https://docs.ultralytics.com/models/) for more options.
- `--save-vid`: Optional. Flag to save the output video with detection results. Saved to `runs/detect/predict`.
- `--slice-height`: Optional. Height of each image slice for SAHI. Defaults to `1024`.
- `--slice-width`: Optional. Width of each image slice for SAHI. Defaults to `1024`.

Experiment with these options, especially slice dimensions, to optimize detection performance for your specific [video processing](https://en.wikipedia.org/wiki/Video_processing) task and target object sizes. Using appropriate [datasets](https://docs.ultralytics.com/datasets/) for training can also significantly impact performance.

## ✨ Contribute

Contributions to enhance this example or add new features are welcome! If you encounter issues or have suggestions, please open an issue or submit a pull request in the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics). Check out our [contribution guide](https://docs.ultralytics.com/help/contributing/) for more details.
