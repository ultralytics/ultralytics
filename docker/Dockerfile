# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Builds ultralytics/ultralytics:latest image on DockerHub https://hub.docker.com/r/ultralytics/ultralytics
# Image is CUDA-optimized for YOLO11 single/multi-GPU training and inference

# Start FROM PyTorch image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=3

# Download fonts to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6 && \
    apt upgrade --no-install-recommends -y openssl tar && \
    rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /ultralytics

# Copy source code and configure git
COPY . .
RUN sed -i '/^\[http "https:\/\/github\.com\/"]/,\+1d' .git/config && \
    rm -rf tmp /root/.config/Ultralytics/persistent_cache.json

# Download a sample YOLO model
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt tmp/yolo11n.pt

# Install Python dependencies and run exports
RUN pip install uv && \
    # Install Python packages with UV
    uv pip install --system -e ".[export]" \
        tensorrt-cu12==9.2.0 \
        albumentations==1.4.6 \
        comet-ml==3.35.5 \
        pycocotools==2.0.7 \
        paddlepaddle==2.6.0 \
        x2paddle==0.9.9 \
        numpy==1.23.5 && \
    # Edge TPU export fails the first time, so run it twice
    ( yolo export model=tmp/yolo11n.pt format=edgetpu imgsz=32 || \
      yolo export model=tmp/yolo11n.pt format=edgetpu imgsz=32 ) && \
    yolo export model=tmp/yolo11n.pt format=ncnn imgsz=32 && \
    rm -rf tmp /root/.config/Ultralytics/persistent_cache.json

# Optional: Add entrypoint or CMD if needed
# CMD ["python3"]