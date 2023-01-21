# Ultralytics YOLO 🚀, GPL-3.0 license
# Builds ultralytics/ultralytics:latest image on DockerHub https://hub.docker.com/r/ultralytics/ultralytics
# Image is CUDA-optimized for YOLOv8 single/multi-GPU training and inference

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Remove torch nightly and install torch stable
RUN rm -rf /opt/pytorch  # remove 1.2GB dir
RUN pip uninstall -y torchtext pillow torch torchvision
RUN pip install --no-cache torch torchvision

# Install linux packages
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# Create working directory
RUN mkdir -p /usr/src/ultralytics
WORKDIR /usr/src/ultralytics

# Copy contents
# COPY . /usr/src/app  (issues as not a .git directory)
RUN git clone https://github.com/ultralytics/ultralytics /usr/src/ultralytics

# Install pip packages
RUN python -m pip install --upgrade pip wheel
RUN pip install --no-cache ultralytics albumentations comet gsutil notebook 'opencv-python<4.6.0.66'

# Set environment variables
ENV OMP_NUM_THREADS=1


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/ultralytics:latest && sudo docker build -f docker/Dockerfile -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/ultralytics:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/ultralytics:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/ultralytics:latest)

# DockerHub tag update
# t=ultralytics/ultralytics:latest tnew=ultralytics/ultralytics:v6.2 && sudo docker pull $t && sudo docker tag $t $tnew && sudo docker push $tnew

# Clean up
# sudo docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/ultralytics/ultralytics:latest
