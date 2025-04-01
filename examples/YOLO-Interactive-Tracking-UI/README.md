# Ultralytics YOLO Interactive Tracking UI 🎯

A modular, educational real-time object detection and tracking UI built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and OpenCV.

This project is ideal for:

- Learning how to integrate [YOLO](https://docs.ultralytics.com) with [object tracking](https://docs.ultralytics.com/modes/track/)
- Testing on edge devices (e.g., Raspberry Pi, Jetson Nano)
- Real-time demos with interactive user input
- Enhancing CV pipelines with tracking UI/UX overlays

---

## 📽️ Demo

![yolo-ezgif com-optimize](https://github.com/user-attachments/assets/179f62e1-97ba-4345-b7cd-a6aa80681996)

---

## ✨ Features

- ✅ Real-time object detection and visual tracking
- 🖱️ Click on any object to initiate tracking
- 🟢 Scope lines and bold bounding box for active tracking
- 🟡 Dashed boxes for passive (non-tracked) objects
- 📟 Live terminal updates with object ID, label, confidence, center
- ⚙️ Configurable thresholds and tracker engine (e.g. `bytetrack`, `botsort`)
- 💡 Supports:
  - ✅ PyTorch `.pt` models for GPU (Jetson, desktop with CUDA)
  - ✅ NCNN `.param + .bin` models for CPU-only (Raspberry Pi, ARM)

---

## 💻 Hardware & Model Compatibility

| Platform         | Model Format       | Example Model        | GPU Acceleration | Notes                           |
| ---------------- | ------------------ | -------------------- | ---------------- | ------------------------------- |
| Raspberry Pi 4/5 | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌ CPU only      | Recommended format for Pi/ARM   |
| Jetson Nano      | PyTorch (.pt)      | `yolov8n.pt`         | ✅ CUDA          | Real-time performance possible  |
| Desktop w/ GPU   | PyTorch (.pt)      | `yolov8s.pt`         | ✅ CUDA          | Best performance                |
| CPU-only laptops | NCNN (.param/.bin) | `yolov8n_ncnn_model` | ❌               | Decent performance (~10–15 FPS) |

---

## 📁 Project Structure

```
YOLO-Interactive-Tracking-UI/
├── interactive_tracker.py   # Main Python tracking UI script
├── yolo11s.pt               # (Optional) Place PyTorch model here
├── yolov8n_ncnn_model      # (Optional) Place NCNN model here
└── README.md                # You're here!
```

> ✅ You are now free to organize model files as you wish. Set the full or relative path directly in `interactive_tracker.py`.

---

## ⚙️ Installation

### Basic Dependencies

```bash
pip install ultralytics opencv-python
```

> Use a virtual environment like `venv` or `conda` (recommended).

---

### Optional: Enable GPU (for PyTorch models)

Install PyTorch based on your system and CUDA version:  
👉 https://pytorch.org/get-started/locally/

---

## 🚀 Usage

### Step 1: Download or convert your model manually

- From Ultralytics: https://github.com/ultralytics/assets/releases
- Supported formats:
  - `yolov8n.pt` (for GPU with PyTorch)
  - `yolov8n_ncnn_model` (for CPU with NCNN)

---

### Step 2: Configure the script

Edit the top of `interactive_tracker.py`:

```python
USE_GPU = True  # Set to False for Raspberry Pi or CPU-only systems

MODEL_PATH_GPU = "yolov8n.pt"
MODEL_PATH_CPU = "yolov8n_ncnn_model"  # Path without file extension
```

---

### Step 3: Run the tracker

```bash
python interactive_tracker.py
```

---

### Controls

- 🖱️ Left-click → Select an object to track
- 🔄 Press `c` → Cancel/reset tracking
- ❌ Press `q` → Quit the app

---

## 🛠 Customization

All config options are at the top of `interactive_tracker.py`:

```python
USE_GPU = True or False
MODEL_PATH_GPU = "yolov8n.pt"
MODEL_PATH_CPU = "yolov8n_ncnn_model"

CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
MAX_DETECTION = 20
SHOW_FPS = True
TRACKER_TYPE = "bytetrack.yaml"
```

Other things you can tweak:

- Colors, line styles, fonts
- Tracker type (e.g., try `botsort.yaml`)
- Object filters, click behavior, frame source (webcam/video)

---

## 👤 Author

**Alireza Ghaderi**  
📅 March 2025  
🔗 [LinkedIn – alireza787b](https://www.linkedin.com/in/alireza787b)

---

## 📜 License & Disclaimer

Released under the **AGPL-3.0 License**.  
Refer to [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for details.

> This project is for **educational and demo purposes** only.  
> Use at your own discretion. Contributions and feedback are welcome!
