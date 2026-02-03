# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Builds ultralytics/ultralytics:latest image on DockerHub https://hub.docker.com/r/ultralytics/ultralytics
# Image is CUDA-optimized for YOLO single/multi-GPU training and inference

# Start FROM PyTorch image https://hub.docker.com/r/pytorch/pytorch or nvcr.io/nvidia/pytorch:25.02-py3
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Set environment variables
# Avoid DDP error "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library"
# Suppress TensorFlow cuDNN, cuBLAS, and cuFFT Registration Warnings and PyTorch NNPACK warnings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TORCH_CPP_LOG_LEVEL=ERROR

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
# gnupg required for Edge TPU install
# libsm6 required by libqxcb to create QT-based windows for visualization; set 'QT_DEBUG_PLUGINS=1' to test in docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 gnupg libsm6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /ultralytics

# Copy contents and configure git
COPY . .
RUN sed -i '/^\[http "https:\/\/github\.com\/"\]/,+1d' .git/config && \
    sed -i'' -e 's/"opencv-python/"opencv-python-headless/' pyproject.toml
ADD https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt .

# Install pip packages (uv already installed in base image)
RUN uv pip install --system -e "." albumentations faster-coco-eval nvidia-ml-py && \
    # Remove extra build files \
    rm -rf tmp /root/.config/Ultralytics/persistent_cache.json

# Usage --------------------------------------------------------------------------------------------------------------

# Production builds: https://github.com/ultralytics/ultralytics/blob/main/.github/workflows/docker.yml
# Example (build): t=ultralytics/ultralytics:latest && docker build -f docker/Dockerfile -t $t .
# Example (push): docker push $t
# Example (pull): t=ultralytics/ultralytics:latest && docker pull $t
# Example (run-gpu): docker run -it --ipc=host --runtime=nvidia --gpus all $t
# Example (run-gpu-subset): docker run -it --ipc=host --runtime=nvidia --gpus "device=2,3" $t
# Note: device=2,3 maps to CUDA 0,1 inside the container.
# Example (run-with-volume): docker run -it --ipc=host --runtime=nvidia --gpus all -v "$PWD/shared/datasets:/datasets" $t
# Example (tag-release): t=ultralytics/ultralytics:latest tnew=ultralytics/ultralytics:vX.Y && docker tag $t $tnew && docker push $tnew
