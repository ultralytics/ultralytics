# Ultralytics YOLO Interactive [Object Tracking](https://docs.ultralytics.com/modes/track/) UI 🚀

A real-time [object detection](https://docs.ultralytics.com/tasks/detect/) and tracking UI built with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) and OpenCV, designed for interactive demos and seamless integration of tracking overlays. Whether you're just getting started with object tracking or looking to enhance it with additional features, this project is for you.

![Ultralytics YOLO Interactive UI Demo](https://github.com/ultralytics/assets/releases/download/v0.0.0/Ultralytics-YOLO-Interactive-UI-Demo.mp4)

## Features

- Real-time object detection and visual tracking
- Click-to-track any detected object
- Scope lines and bold [bounding boxes](https://docs.ultralytics.com/usage/simple-utilities/#bounding-boxes) for active tracking
- Dashed boxes for passive (non-tracked) objects
- [Live terminal output](https://docs.ultralytics.com/guides/view-results-in-terminal/): object ID, label, confidence, and center coordinates
- Adjustable object tracking algorithms (`bytetrack`, `botsort`)
- Supports:
  - PyTorch `.pt` models (for GPU devices like [Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) or CUDA-enabled desktops)
  - [NCNN](https://docs.ultralytics.com/integrations/ncnn/) `.param + .bin` models (for CPU-only devices like Raspberry Pi or ARM boards)

## Project structure

```
YOLO-Interactive-Tracking-UI/
├── interactive_tracker.py   # Main Python tracking UI script
└── README.md                # You're here!
```

## Hardware & [Model](https://docs.ultralytics.com/models/) compatibility

| Platform         | Model Format       | Example Model        | GPU Acceleration | Notes                           |
| ---------------- | ------------------ | -------------------- | ---------------- | ------------------------------- |
| Raspberry Pi 4/5 | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌ CPU only      | Recommended format for Pi/ARM   |
| Jetson Nano      | PyTorch (.pt)      | `yolov8n.pt`         | ✅ CUDA          | Real-time performance possible  |
| Desktop w/ GPU   | PyTorch (.pt)      | `yolov8s.pt`         | ✅ CUDA          | Best performance                |
| CPU-only laptops | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌               | Decent performance (~10–15 FPS) |

## Installation

### Basic dependencies

```bash
pip install ultralytics
```

> Use a virtual environment like `venv` or `conda` (recommended).

> Install PyTorch based on your system and CUDA version: https://pytorch.org/get-started/locally/

## Quickstart

### Step 1: Download, convert, or specify model

- For pre-trained YOLO models (e.g., `yolo11s.pt` or `yolov8s.pt`), simply specify the model name as `MODEL_PATH_CPU` or `MODEL_PATH_GPU`. These models will be auto-downloaded and cached, or you can manually download them from [Ultralytics releases](https://github.com/ultralytics/assets/releases) and place them in the folder.
- If you're using a custom YOLO model, ensure the model file is in the project folder or provide its relative path in the parameters.

- **Supported Formats:**
  - `yolov8n.pt` (for GPU with PyTorch)
  - `yolov8n_ncnn_model` (for CPU with NCNN)

### Step 2: Configure the script

Edit the top global parameters of `interactive_tracker.py`:

```python
enable_gpu = False  # Set True if running with CUDA
model_file = "yolo11s.pt"  # Path to model file
show_fps = True  # If True, shows current FPS in top-left corner
show_conf = False  # Display or hide the confidence score
save_video = False  # Set True to save output video
video_output_path = "interactive_tracker_output.avi"  # Output video file name


conf = 0.3  # Min confidence for object detection (lower = more detections, possibly more false positives)
iou = 0.3  # IoU threshold for NMS (higher = less overlap allowed)
max_det = 20  # Maximum objects per im (increase for crowded scenes)

tracker = "bytetrack.yaml"  # Tracker config: 'bytetrack.yaml', 'botsort.yaml', etc.
track_args = {
    "persist": True,  # Keep frames history as a stream for continuous tracking
    "verbose": False,  # Print debug info from tracker
}

window_name = "Ultralytics YOLO Interactive Tracking"  # Output window name
```

### Step 3: Run the object tracking

```bash
python interactive_tracker.py
```

### Controls

- 🖱️ Left-click → Select an object to track
- 🔄 Press `c` → Cancel/reset tracking
- ❌ Press `q` → Quit the app

### Saving Output Video (Optional)

You can choose to save the visualized output as a video file. To enable this, set the following in `interactive_tracker.py`:

```python
save_video = True  # Enables video recording
video_output_path = "output.avi"  # Customize your output file name
```

The video will be saved in the working directory upon exit.

## Author

- [Connect with author on LinkedIn](https://www.linkedin.com/in/alireza787b)
- Published Date ![Published date](https://img.shields.io/badge/published_Date-2025--04--01-purple)

## License & Disclaimer

This project is released under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). For full licensing terms, please visit the [Ultralytics YOLO licensing page](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

This project is intended solely for educational and demonstration purposes. Please use it responsibly and at your own discretion. The author assumes no liability for any misuse or unintended consequences. Feedback, forks, and contributions are highly encouraged and always appreciated!
