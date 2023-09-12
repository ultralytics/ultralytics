### YOLOv8 with SAHI (Inference on Video)

SAHI designed to optimize object detection algorithms for large-scale and high-resolution imagery. Its core functionality lies in partitioning images into manageable slices, running object detection on each slice, and then stitching the results back together.

#### Step 1: Install the required libraries

Just clone the repository and run

```

pip install sahi ultralytics

```

#### Step 2: Run the inference with SAHI using Ultralytics YOLOv8

```

#if you want to save results

python yolov8_sahi.py --source "path to video file" --save-img


#if you want to change model file

python yolov8_sahi --source "path to video file" --save-img --weights "yolov8m.pt"

```
