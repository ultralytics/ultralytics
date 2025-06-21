# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

from ultralytics import YOLO

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_NTHREADS"] = "2"
os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":
    model = YOLO("yolo11s.pt")  # Load a pretrained model
    results = model.train(
        data="skinning.yaml",
        epochs=100,
        imgsz=640,
        batch=24,
        mosaic=0,
        device=[0, 1],
        amp=False,
        translate=0.12431,
        scale=0.07643,
    )

# for node = 0
# torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=192.168.0.60 --master_port=23456 main.py

# for node = 1
# torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=192.168.0.60 --master_port=23456 main.py
