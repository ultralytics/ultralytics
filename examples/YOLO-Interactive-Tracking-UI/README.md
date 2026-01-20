# Ultralytics YOLO Interactive Object Tracking UI 🚀

A real-time [object detection](https://docs.ultralytics.com/tasks/detect/) and [tracking](https://docs.ultralytics.com/modes/track/) UI built with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) and [OpenCV](https://opencv.org/), designed for interactive demos and seamless integration of tracking overlays. Whether you're just getting started with object tracking or looking to enhance it with additional features, this project provides a solid foundation.

https://github.com/user-attachments/assets/723e919e-555b-4cca-8e60-18e711d4f3b2

## ✨ Features

- Real-time object detection and visual tracking
- Click-to-track any detected object
- Scope lines and bold [bounding boxes](https://docs.ultralytics.com/usage/simple-utilities/#bounding-boxes) for active tracking
- Dashed boxes for passive (non-tracked) objects
- [Live terminal output](https://docs.ultralytics.com/guides/view-results-in-terminal/): object ID, label, [confidence](https://www.ultralytics.com/glossary/confidence), and center coordinates
- Adjustable object tracking algorithms ([ByteTrack](https://docs.ultralytics.com/reference/trackers/byte_tracker/), [BoT-SORT](https://docs.ultralytics.com/reference/trackers/bot_sort/))
- Supports:
  - [PyTorch](https://pytorch.org/) `.pt` models (for GPU devices like [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) or [CUDA](https://developer.nvidia.com/cuda)-enabled desktops)
  - [NCNN](https://docs.ultralytics.com/integrations/ncnn/) `.param + .bin` models (for CPU-only devices like [Raspberry Pi](https://www.raspberrypi.org/) or ARM boards)

## 🏗️ Project Structure

```
YOLO-Interactive-Tracking-UI/
├── interactive_tracker.py   # Main Python tracking UI script
└── README.md                # You're here!
```

## 💻 Hardware & Model Compatibility

| Platform         | Model Format       | Example Model        | GPU Acceleration | Notes                           |
| ---------------- | ------------------ | -------------------- | ---------------- | ------------------------------- |
| Raspberry Pi 4/5 | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌ CPU only      | Recommended format for Pi/ARM   |
| Jetson Nano      | PyTorch (.pt)      | `yolov8n.pt`         | ✅ CUDA          | Real-time performance possible  |
| Desktop w/ GPU   | PyTorch (.pt)      | `yolov8s.pt`         | ✅ CUDA          | Best performance                |
| CPU-only laptops | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌               | Decent performance (~10–15 FPS) |

_Note: Performance may vary based on the specific hardware, model complexity, and input resolution._

## 🛠️ Installation

### Basic Dependencies

Install the core `ultralytics` package:

```bash
pip install ultralytics
```

> **Tip:** Use a virtual environment like `venv` or [`conda`](https://docs.ultralytics.com/guides/conda-quickstart/) (recommended) to manage dependencies.

> **GPU Support:** Install PyTorch based on your system and CUDA version by following the official guide: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## 🚀 Quickstart

### Step 1: Download, Convert, or Specify Model

- For pre-trained Ultralytics YOLO [models](https://docs.ultralytics.com/models/) (e.g., `yolo11s.pt` or `yolov8s.pt`), simply specify the model name in the script parameters (`model_file`). These models will be automatically downloaded and cached. You can also manually download them from [Ultralytics Assets Releases](https://github.com/ultralytics/assets/releases) and place them in the project folder.
- If you're using a custom-trained YOLO model, ensure the model file is in the project folder or provide its relative path.
- For CPU-only devices, export your chosen model (e.g., `yolov8n.pt`) to the [NCNN format](https://docs.ultralytics.com/integrations/ncnn/) using the Ultralytics `export` mode.

- **Supported Formats:**
  - `yolo11s.pt` (for GPU with PyTorch)
  - `yolov8n_ncnn_model` (directory containing `.param` and `.bin` files for CPU with NCNN)

### Step 2: Configure the Script

Edit the global parameters at the top of `interactive_tracker.py`:

```python
# --- Configuration ---
enable_gpu = False  # Set True if running with CUDA and PyTorch model
model_file = "yolo11s.pt"  # Path to model file (.pt for GPU, _ncnn_model dir for CPU)
show_fps = True  # Display current FPS in the top-left corner
show_conf = False  # Display confidence score for each detection
save_video = False  # Set True to save the output video stream
video_output_path = "interactive_tracker_output.avi"  # Output video file name

# --- Detection & Tracking Parameters ---
conf = 0.3  # Minimum confidence threshold for object detection
iou = 0.3  # IoU threshold for Non-Maximum Suppression (NMS)
max_det = 20  # Maximum number of objects to detect per frame

tracker = "bytetrack.yaml"  # Tracker configuration: 'bytetrack.yaml' or 'botsort.yaml'
track_args = {
    "persist": True,  # Keep track history across frames
    "verbose": False,  # Suppress detailed tracker debug output
}

window_name = "Ultralytics YOLO Interactive Tracking"  # Name for the OpenCV display window
# --- End Configuration ---
```

- **`enable_gpu`**: Set to `True` if you have a CUDA-compatible GPU and are using a `.pt` model. Keep `False` for NCNN models or CPU-only execution.
- **`model_file`**: Ensure this points to the correct model file or directory based on `enable_gpu`.
- **`conf`**: Adjust the [confidence](https://www.ultralytics.com/glossary/confidence) threshold. Lower values detect more objects but may increase false positives.
- **`iou`**: Set the [Intersection over Union (IoU)](https://www.ultralytics.com/glossary/intersection-over-union-iou) threshold for [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms). Higher values allow more overlapping boxes.
- **`tracker`**: Choose between available tracker configuration files ([ByteTrack](https://docs.ultralytics.com/reference/trackers/byte_tracker/), [BoT-SORT](https://docs.ultralytics.com/reference/trackers/bot_sort/)).

### Step 3: Run the Object Tracking

Execute the script from your terminal:

```bash
python interactive_tracker.py
```

### Controls

- 🖱️ **Left-click** on a detected object's bounding box to start tracking it.
- 🔄 Press the **`c`** key to cancel the current tracking and select a new object.
- ❌ Press the **`q`** key to quit the application.

### Saving Output Video (Optional)

If you want to record the tracking session, enable the `save_video` option in the configuration:

```python
save_video = True  # Enables video recording
video_output_path = "output.avi"  # Customize your output file name (e.g., .mp4, .avi)
```

The video file will be saved in the project's working directory when you quit the application by pressing `q`.

## 👤 Author

- **Alireza**
- [Connect on LinkedIn](https://www.linkedin.com/in/alireza787b)
- Published: 2025-04-01

## 📜 License & Disclaimer

This project is released under the [AGPL-3.0 license](https://www.ultralytics.com/legal/agpl-3-0-software-license). For full licensing details, please refer to the [Ultralytics Licensing page](https://www.ultralytics.com/license).

This software is provided "as is" for educational and demonstration purposes. Use it responsibly and at your own risk. The author assumes no liability for misuse or unintended consequences.

## 🤝 Contributing

Contributions, feedback, and bug reports are welcome! Feel free to open an issue or submit a pull request on the original repository if you have improvements or suggestions.
