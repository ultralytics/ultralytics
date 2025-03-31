# Ultralytics YOLO Interactive Tracking UI 🎯

An educational and modular object tracking interface built using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and OpenCV. This project is ideal for:

- Learning [Ultralytics YOLO](https://docs.ultralytics.com/) + [object tracking](https://docs.ultralytics.com/modes/track/) integration
- Testing on edge devices ([Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/), [NVIDIA Jetson Nano](https://docs.ultralytics.com/guides/nvidia-jetson/), etc.)
- Building real-time, interactive tracking demos
- Enhancing vision pipelines with UX-friendly overlays

## Project Demo

![yolo-ezgif com-optimize](https://github.com/user-attachments/assets/179f62e1-97ba-4345-b7cd-a6aa80681996)

## Features

- ✅ Real-time object detection using Ultralytics YOLOv8
- 🖱️ Click on any object to begin tracking it
- 🟩 YOLO-style bounding boxes with class names and confidence
- 🔭 Scope lines + center marker for selected tracked object
- 🔁 Dashed box style for inactive objects
- 📟 Terminal prints: object class, ID, confidence, center
- 🎛 Customizable settings:

  - FPS display toggle
  - Max detections
  - Tracker backend (`bytetrack`, `botsort`, etc.)
  - Confidence / IoU thresholds

- Supports both:
  - ✅ PyTorch models (`.pt`) for GPU (Jetson/desktop)
  - ✅ NCNN models (`.param`/`.bin`) for CPU-only (Pi, ARM boards)

## Hardware & Model Support

| Platform          | Format Used        | Model Example        | GPU Acceleration | Notes                      |
| ----------------- | ------------------ | -------------------- | ---------------- | -------------------------- |
| Raspberry Pi 4/5  | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌ CPU only      | Lightweight, fast enough   |
| Jetson Nano       | PyTorch (.pt)      | `yolov8n.pt`         | ✅ CUDA          | Great for GPU acceleration |
| Desktop PC w/ GPU | PyTorch (.pt)      | `yolov8s.pt`         | ✅ CUDA          | Fastest option             |
| CPU-only laptops  | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌               | Still usable at ~10–15 FPS |

## Project Structure

```
YOLO-Interactive-Tracking-UI/
├── yolo/                    # Store your models here (.pt or .ncnn)
├── add_yolo_model.py        # Helper: downloads + exports YOLO to NCNN
├── interactive_tracker.py   # Main OpenCV tracking + UI demo
└── README.md                # You're reading it
```

## Installation

### Python (3.8+ required)

```bash
pip install ultralytics opencv-python
```

> You can use `venv` or `conda` if you prefer virtual environments.

### Optional: GPU Support (PyTorch + CUDA)

If you're using a CUDA GPU (desktop or Jetson Nano):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> Replace `cu118` with your CUDA version if needed.

## Quick Start

### 1. Prepare a YOLO Model

- For CPU/NCNN:

  ```bash
  python add_yolo_model.py
  ```

  This will download + convert `yolov8n.pt` to NCNN automatically.

- For GPU:
  Download `.pt` models like `yolov8n.pt` or `yolov8s.pt` from [Ultralytics assets](https://github.com/ultralytics/assets/releases).

Place them in the `yolo/` folder.

### 2. Run the Object Tracking

```bash
python interactive_tracker.py
```

Inside the script, configure this block to toggle CPU vs GPU:

```python
USE_GPU = True  # Set to False for Raspberry Pi or CPU-only
```

### 3. Controls

- 🖱️ Left click: Select an object to track
- 🔄 Press `c`: Cancel/reset tracking
- ❌ Press `q`: Quit the app

## Customization

In `interactive_tracker.py`, modify the following:

```python
USE_GPU = True  # GPU (CUDA) or False for CPU
MODEL_PATH_GPU = "yolo/yolov8n.pt"
MODEL_PATH_CPU = "yolo/yolov8n_ncnn_model"

CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
MAX_DETECTION = 20
SHOW_FPS = True
TRACKER_TYPE = "bytetrack.yaml"
```

You can also change bounding box styles, text font, colors, etc. The code is well-commented for educational clarity.

## 👤 Author

- Connect with author: [here](https://www.linkedin.com/in/alireza787b)
- Published Date ![Published Date](https://img.shields.io/badge/published_Date-2025--04--01-purple)

## License & Disclaimer

This project is released under the **AGPL-3.0 license**. For full licensing terms, please visit the [Ultralytics YOLO License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

It is intended solely for educational and demonstration purposes. Please use it responsibly and at your own discretion. The author assumes no liability for any misuse or unintended consequences. Feedback, forks, and contributions are highly encouraged and always appreciated!
