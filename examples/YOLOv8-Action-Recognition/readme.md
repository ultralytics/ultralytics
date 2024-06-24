# Zero-shot Action Recognition with YOLOv8 (Inference on Video)

- Action recognition is a technique used to identify and classify actions performed by individuals in a video. This process enables more advanced analyses when multiple actions are considered. The actions can be detected and classified in real time.
- The system can be customized to recognize specific actions based on the user's preferences and requirements.

## Table of Contents

- [Step 1: Install the Required Libraries](#step-1-install-the-required-libraries)
- [Step 2: Run the Action Recognition Using Ultralytics YOLOv8](#step-2-run-the-action-recognition-using-ultralytics-yolov8)
- [Usage Options](#usage-options)
- [FAQ](#faq)

## Step 1: Install the Required Libraries

Clone the repository, install dependencies and `cd` to this local directory for commands in Step 2.

```bash
# Clone ultralytics repo
git clone https://github.com/ultralytics/ultralytics

# cd to local directory
cd examples/YOLOv8-Action-Recognition

# Install dependencies
pip install -U -r requirements.txt
```

## Step 2: Run the Action Recognition Using Ultralytics YOLOv8

Here are the basic commands for running the inference:

### Note

The action recognition model will automatically detect and track people in the video, and classify their actions based on the specified labels. The results will be displayed in real-time on the video output. You can customize the action labels by modifying the `--zero-shot-labels` argument when running the script.

```bash
# Basic usage
python action_recognition.py --source path/to/video.mp4

# YouTube video inference
python action_recognition.py --source https://www.youtube.com/watch?v=dQw4w9WgXcQ

# Run on CPU
python action_recognition.py --source path/to/video.mp4 --device cpu

# Use a different video classifier model
python action_recognition.py --source path/to/video.mp4 --video-classifier-model "s3d"

# Custom zero-shot labels
python action_recognition.py --source path/to/video.mp4 --zero-shot-labels "dancing" "singing" "jumping"

# Use FP16 for inference (only for HuggingFace models)
python action_recognition.py --source path/to/video.mp4 --fp16

# Export output as mp4
python action_recognition.py --source path/to/video.mp4 --output-path output.mp4

# Combine multiple options
python action_recognition.py --source https://www.youtube.com/watch?v=dQw4w9WgXcQ --device 0 --video-classifier-model "microsoft/xclip-base-patch32" --zero-shot-labels "dancing" "singing" "playing guitar" --fp16
```

## Usage Options

- `--weights`: Path to the YOLO model weights (default: "yolov8n.pt")
- `--device`: Cuda device, i.e. 0 or 0,1,2,3 or cpu (default: auto-detect)
- `--source`: Video file path or YouTube URL (required)
- `--output-path`: Output video file path
- `--crop-margin-percentage`: Percentage of margin to add around detected objects (default: 10)
- `--num-video-sequence-samples`: Number of video frames to use for classification (default: 8)
- `--skip-frame`: Number of frames to skip between detections (default: 1)
- `--video-cls-overlap-ratio`: Overlap ratio between video sequences (default: 0.25)
- `--fp16`: Use FP16 for inference (only for HuggingFace models)
- `--video-classifier-model`: Video classifier model name or path (default: "microsoft/xclip-base-patch32")
- `--zero-shot-labels`: Labels for zero-shot video classification (default: \["walking", "running", "brushing teeth", "looking into phone", "weight lifting", "cooking", "sitting"\])

## FAQ

**1. What Does Action Recognition Involve?**

Action recognition is a computational method used to identify and classify actions or activities performed by individuals in recorded video or real-time streams. This technique is widely used in video analysis, surveillance, and human-computer interaction, enabling the detection and understanding of human behaviors based on their motion patterns and context.

**2. Is Custom Action Labels Supported by the Action Recognition?**

Yes, the Action Recognition module supports custom action labels for zero-shot video classification. You can specify your own set of labels by modifying the `zero_shot_labels` variable in the `action_recognition.py` file. Here's an example of how to set custom labels:

```python
zero_shot_labels = [
    "walking",
    "running",
    "brushing teeth",
    "looking into phone",
    "weight lifting",
    "cooking",
    "sitting",
]
```

You can adjust these labels to match the specific actions you want to recognize in your video. The system will then attempt to classify the detected actions based on these custom labels.

Additionally, you can choose between different video classification models:

1. For Hugging Face models, you can use any compatible video classification model. The default is set to:

   - "microsoft/xclip-base-patch32"

2. For TorchVision models (no support for zero-shot labels), you can select from the following options:

   - "s3d"
   - "r3d_18"
   - "swin3d_t"
   - "swin3d_b"
   - "mvit_v1_b"
   - "mvit_v2_s"

**3. Why Combine Action Recognition with YOLOv8?**

YOLOv8 specializes in the detection and tracking of objects in video streams. Action recognition complements this by enabling the identification and classification of actions performed by individuals, making it a valuable application of YOLOv8.

**4. Can I Employ Other YOLO Versions?**

Certainly, you have the flexibility to specify different YOLO model weights using the `--weights` option.
