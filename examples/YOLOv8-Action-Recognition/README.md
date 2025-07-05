# Zero-Shot Action Recognition with Ultralytics YOLOv8 (Inference on Video)

Action recognition is a [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) technique used to identify and classify actions performed by individuals in a video. This process enables more advanced analyses when multiple actions are considered. Using models like [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), actions can be detected and classified in real time. This system leverages **zero-shot learning**, allowing it to recognize actions it wasn't explicitly trained on by using descriptive labels. Learn more about zero-shot concepts on [Wikipedia](https://en.wikipedia.org/wiki/Zero-shot_learning).

The system can be customized to recognize specific actions based on the user's preferences and requirements by providing different text labels.

## 🎬 Table of Contents

- [Step 1: Install the Required Libraries](#step-1-install-the-required-libraries)
- [Step 2: Run Action Recognition Using Ultralytics YOLOv8](#step-2-run-action-recognition-using-ultralytics-yolov8)
- [Usage Options](#usage-options)
- [FAQ](#faq)

## ⚙️ Step 1: Install the Required Libraries

Clone the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) using [Git](https://git-scm.com/), install dependencies with [pip](https://pip.pypa.io/en/stable/), and navigate (`cd`) to this local directory for the commands in Step 2.

```bash
# Clone ultralytics repo
git clone https://github.com/ultralytics/ultralytics

# cd to local directory
cd ultralytics/examples/YOLOv8-Action-Recognition

# Install dependencies using Python's package manager
pip install -U -r requirements.txt
```

## 🚀 Step 2: Run Action Recognition Using Ultralytics YOLOv8

Here are the basic commands for running inference:

### Note

The action recognition model will automatically perform [object detection](https://www.ultralytics.com/glossary/object-detection) and [tracking](https://docs.ultralytics.com/modes/track/) for people in the video, and classify their actions based on the specified labels. The results will be displayed in real-time on the video output. You can customize the action labels by modifying the `--labels` argument when running the [Python](https://www.python.org/) script. This utilizes a video classifier model, often sourced from platforms like [Hugging Face Models](https://huggingface.co/models).

```bash
# Quick start with default video and labels
python action_recognition.py

# Basic usage with a YouTube video and custom labels
python action_recognition.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --labels "dancing" "singing a song"

# Use a local video file
python action_recognition.py --source path/to/video.mp4

# Use a medium-sized YOLOv8 model for potentially better detector performance
python action_recognition.py --weights yolov8m.pt

# Run inference on the CPU instead of GPU
python action_recognition.py --device cpu

# Use a different video classifier model from TorchVision
python action_recognition.py --video-classifier-model "s3d"

# Use FP16 (half-precision) for faster inference (only for HuggingFace models)
python action_recognition.py --fp16

# Export the output video with recognized actions to an mp4 file
python action_recognition.py --output-path output.mp4

# Combine multiple options: specific YouTube source, GPU device 0, specific HuggingFace model, custom labels, and FP16
python action_recognition.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device 0 --video-classifier-model "microsoft/xclip-base-patch32" --labels "dancing" "singing a song" --fp16
```

## 🛠️ Usage Options

- `--weights`: Path to the YOLO [model weights](https://www.ultralytics.com/glossary/model-weights) file (default: `"yolov8n.pt"`). You can choose other models like `yolov8s.pt`, `yolov8m.pt`, etc.
- `--device`: Cuda device identifier (e.g., `0` or `0,1,2,3`) or `cpu` to run on the [CPU](https://www.ultralytics.com/glossary/cpu) (default: auto-detects available [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit)).
- `--source`: Path to a local video file or a YouTube URL (default: "[rickroll](https://www.youtube.com/watch?v=dQw4w9WgXcQ)").
- `--output-path`: Path to save the output video file (e.g., `output.mp4`). If not specified, the video is displayed in a window.
- `--crop-margin-percentage`: Percentage of margin to add around detected objects before cropping for classification (default: `10`).
- `--num-video-sequence-samples`: Number of video frames sampled from a sequence to feed into the classifier (default: `8`).
- `--skip-frame`: Number of frames to skip between detections to speed up processing (default: `1`).
- `--video-cls-overlap-ratio`: Overlap ratio between consecutive video sequences sent for classification (default: `0.25`).
- `--fp16`: Use [FP16 (half-precision)](https://www.ultralytics.com/glossary/half-precision) for inference, potentially speeding it up on compatible hardware (only applicable to Hugging Face models).
- `--video-classifier-model`: Name or path of the video classifier model (default: `"microsoft/xclip-base-patch32"`). Can be a Hugging Face model name or a [TorchVision model](https://docs.pytorch.org/vision/stable/models.html) name.
- `--labels`: A list of text labels for zero-shot video classification (default: `["dancing", "singing a song"]`).

## 🤔 FAQ

### 1. What Does Action Recognition Involve?

Action recognition is a computational method used to identify and classify actions or activities performed by individuals in recorded video or real-time streams. This technique is widely used in video analysis, surveillance, and human-computer interaction, enabling the detection and understanding of human behaviors based on their motion patterns and context. It often combines [object tracking](https://www.ultralytics.com/glossary/object-tracking) with classification. Explore more on [video classification research](https://arxiv.org/).

### 2. Are Custom Action Labels Supported by Action Recognition?

Yes, custom action labels are supported. The `action_recognition.py` script allows users to specify their own custom labels for **zero-shot video classification**. This is done using the `--labels` argument. For example:

```bash
python action_recognition.py --source https://www.youtube.com/watch?v=dQw4w9WgXcQ --labels "walking" "running" "jumping"
```

You can adjust these labels to match the specific actions you want the system to recognize in your video. The system will then attempt to classify detected actions based on these custom labels using its understanding derived from large datasets.

Additionally, you can choose between different video classification models:

1.  **Hugging Face Models**: You can use any compatible video classification model available on Hugging Face Hub. The default is:
    - `"microsoft/xclip-base-patch32"`
2.  **TorchVision Models**: These models do not support zero-shot classification with custom text labels but offer pre-trained classification capabilities. Options include:
    - `"s3d"`
    - `"r3d_18"`
    - `"swin3d_t"`
    - `"swin3d_b"`
    - `"mvit_v1_b"`
    - `"mvit_v2_s"`

### 3. Why Combine Action Recognition with YOLOv8?

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) excels at fast and accurate [object detection](https://docs.ultralytics.com/tasks/detect/) and tracking in video streams. Combining it with action recognition allows the system not only to locate individuals (using YOLOv8's detection capabilities) but also to understand _what_ they are doing. This synergy provides a richer analysis of video content, crucial for applications like automated surveillance, sports analytics, or human-robot interaction. See our blog post on [object detection and tracking](https://www.ultralytics.com/blog/object-detection-and-tracking-with-ultralytics-yolov8).

### 4. Can I Employ Other YOLO Versions?

Certainly! While this example defaults to `yolov8n.pt`, you have the flexibility to specify different Ultralytics YOLO model weights using the `--weights` option. For instance, you could use `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, or `yolov8x.pt` for potentially higher detection accuracy at the cost of inference speed. You can even use models trained for other tasks if applicable, though detection models are standard here. Check the [Ultralytics documentation](https://docs.ultralytics.com/) for available models and their performance metrics.

---

We hope this guide helps you implement zero-shot action recognition using Ultralytics YOLOv8! Feel free to explore the code and experiment with different options. If you encounter issues or have suggestions, please consider contributing by opening an issue or pull request on the [GitHub repository](https://github.com/ultralytics/ultralytics). See our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details.
