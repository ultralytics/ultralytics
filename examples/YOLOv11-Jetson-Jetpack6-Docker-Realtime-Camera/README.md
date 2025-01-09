# YOLOv11 Real-time Object Detection for Jetson Jetpack 6.1 using Docker

Real-time object detection on NVIDIA Jetson devices using YOLOv11 in a Docker container. This repository provides a streamlined setup for:

- Running YOLOv11 with hardware acceleration.
- Processing camera feeds in real-time
- Containerized deployment using Docker that have OpenCV with Gstreamer enabled.

## 📋 Prerequisites

- NVIDIA Jetson device with JetPack 6.1
- Docker Engine
- NVIDIA Container Runtime
- USB or CSI camera connected to your Jetson

## 🛠️ Installation

### Method 1: Pre-built Docker Image (Recommended)

```bash
t=ultralytics/ultralytics:latest-jetson-jetpack6
sudo docker pull $t
```

### Method 2: Build from Source

1. Clone the repository:

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
```

2. Prepare Dockerfile:

   - Copy `docker/Dockerfile-jetson-jetpack6` to root directory

3. Build Docker image:

```bash
sudo docker build -f Dockerfile-jetson-jetpack6 -t latest-jetson-jetpack6:6.1 .
```

## 📷 Create the Docker Image and Running Realtime Camera Object Detection

1. Copy example folder:

   ```bash
   cp -r examples/YOLOv11-Jetson-Jetpack6-Docker-Realtime-Camera /your/preferred/path/
   ```

2. Configure settings:

   - Update folder paths in `run.sh`
   - Modify camera ID in `main.py` if needed

3. Launch container:

   ```bash
   sh run.sh
   ```

4. Start detection insider the container:
   ```bash
   cd public
   python3 main.py
   ```

## 🔧 Configuration

### Camera Setup

- Default: USB camera (ID: 0)
- CSI camera: Modify `camera_id` in `main.py`
- Multiple cameras: Update camera index based on your setup

### Docker Container

- Modify mount points in `run.sh`
- Adjust resource allocation if needed
- Configure network settings as required

## 🚨 Troubleshooting

Common issues and solutions:

1. Camera Access

   - Verify camera permissions
   - Check camera connection
   - Confirm camera ID

2. Docker Issues
   - Ensure NVIDIA runtime is installed
   - Verify JetPack version
   - Check container privileges
