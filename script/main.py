# import sys
# import os
# from pathlib import Path

# # 1. Get the absolute path to the repository root
# # __file__ is script/main.py, so its parent's parent is the repo root
# current_dir = Path(__file__).resolve().parent
# repo_root = current_dir.parent
# # 2. Insert the repo root at the very beginning of sys.path
# # This ensures python finds your local `ultralytics` dir before the PIP version
# sys.path.insert(0, str(repo_root))
# # Check where ultralytics is being imported from to test it works
# import ultralytics

# print(f"Imported ultralytics from: {ultralytics.__file__}")

import torch
from ultralytics.models import RTDETR


def main():
    # Enable PyTorch anomaly detection
    # This will trace the forward pass and throw an exact stack trace when a NaN occurs
    torch.autograd.set_detect_anomaly(True)

    model = RTDETR("rtdetr-l.yaml")

    # Train the model to see the loss debugging points in action
    print("Starting training...")
    model.train(data="coco8.yaml", epochs=1, imgsz=640)


if __name__ == "__main__":
    main()
