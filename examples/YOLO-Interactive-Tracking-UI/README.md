# YOLO Interactive Tracking UI 🎯

An educational and modular object tracking interface built using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and OpenCV — featuring live object detection, manual target selection via mouse click, styled visual overlays (bounding boxes, scope lines), and terminal updates.

This project is ideal for:

- Learning YOLO + tracking integration
- Demonstrating real-time object tracking
- Research prototypes with manual interaction
- UI/UX-enhanced vision tools

---

## 📸 Demo

![yolo-ezgif com-optimize](https://github.com/user-attachments/assets/179f62e1-97ba-4345-b7cd-a6aa80681996)

---

## ✨ Features

- Click on any object to begin tracking it
- Automatically detects and draws YOLO-style bounding boxes
- Visual enhancements:
  - Dashed boxes for inactive objects
  - Bold box with scope lines and center point for tracked object
- Real-time terminal updates (ID, label, center, confidence)
- Configurable:
  - Confidence threshold, IoU threshold, max detections
  - Toggle FPS display
  - Tracker backend (`bytetrack`, `botsort`, etc.)
- Supports custom YOLO model selection and NCNN export

---

## 📁 Folder Structure

```
YOLO-Interactive-Tracking-UI/
├── yolo/                    # Stores downloaded and exported YOLO models (in NCNN format)
├── add_yolo_model.py        # Script to download + export model to NCNN format
├── interactive_tracker.py   # Main tracking interface (with OpenCV display)
└── README.md                # You are here!
```

---

## ⚙️ Prerequisites

Make sure you have:

```bash
pip install ultralytics opencv-python torch requests
```

> Using a virtual Python environment (eg. venv) is highly recommended.

> ⚠️ Requires **Python 3.8+**

---

## 🚀 Quick Start

### 1. Download and Export a YOLO Model (to NCNN)

```bash
python add_yolo_model.py
```

> You can interactively enter any versions, or custom models, or download from a URL.
> Follow the instructions to load a pre-trained model.

---

### 2. Run the Interactive Tracker

```bash
python interactive_tracker.py
```

Then:

- 🖱 Click on an object to start tracking
- 🔄 Press `c` to cancel tracking
- ❌ Press `q` to exit

---

## 🛠 Customization

You can tune behavior in `interactive_tracker.py`:

```python
SHOW_FPS = True
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
MAX_DETECTION = 20
TRACKER_TYPE = "bytetrack.yaml"
```

You may also modify the appearance of bounding boxes, overlay colors, scope styles, or tracking behavior.

---

## 👤 Author

**Alireza Ghaderi**  
📅 March 2025  
🔗 [LinkedIn](https://www.linkedin.com/in/alireza787b)

---

## 📜 License & Disclaimer

This project is provided **for educational and research purposes only**.  
The author makes **no guarantees** and assumes **no responsibility** for any use of this code in production or commercial environments.  
Feel free to modify, extend, and contribute via pull requests. Feedback is always welcome!
