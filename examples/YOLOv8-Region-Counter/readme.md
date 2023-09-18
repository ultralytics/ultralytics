# Regions Counting Using YOLOv8 (Inference on Video)

- Region counting is a method employed to tally the objects within a specified area, allowing for more sophisticated analyses when multiple regions are considered. These regions can be adjusted interactively using a Left Mouse Click, and the counting process occurs in real time.
- Regions can be adjusted to suit the user's preferences and requirements.

<div>
  <p align="center">
  <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/978c8dd4-936d-468e-b41e-1046741ec323" width="45%"/>
    <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/069fd81b-8451-40f3-9f14-709a7ac097ca" width="45%"/>
&nbsp; &nbsp; &nbsp; &nbsp;
</p>
</div>

## Table of Contents

- [Step 1: Install the Required Libraries](#step-1-install-the-required-libraries)
- [Step 2: Run the Region Counting Using Ultralytics YOLOv8](#step-2-run-the-region-counting-using-ultralytics-yolov8)
- [Usage Options](#usage-options)
- [FAQ](#faq)

## Step 1: Install the Required Libraries

Clone the repository, install dependencies and `cd` to this local directory for commands in Step 2.

```bash
# Clone ultralytics repo
git clone https://github.com/ultralytics/ultralytics

# cd to local directory
cd ultralytics/examples/YOLOv8-Region-Counter
```

## Step 2: Run the Region Counting Using Ultralytics YOLOv8

Here are the basic commands for running the inference:

### Note

After the video begins playing, you can freely move the region anywhere within the video by simply clicking and dragging using the left mouse button.

```bash
# If you want to save results
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --view-img

# If you want to change model file
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --weights "path/to/model.pt"

# If you dont want to save results
python yolov8_region_counter.py --source "path/to/video.mp4" --view-img
```

## Usage Options

- `--source`: Specifies the path to the video file you want to run inference on.
- `--save-img`: Flag to save the detection results as images.
- `--weights`: Specifies a different YOLOv8 model file (e.g., `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`).
- `--line-thickness`: Specifies the bounding box thickness
- `--region-thickness`: Specific the region boxes thickness

## FAQ

**1. What Does Region Counting Involve?**

Region counting is a computational method utilized to ascertain the quantity of objects within a specific area in recorded video or real-time streams. This technique finds frequent application in image processing, computer vision, and pattern recognition, facilitating the analysis and segmentation of objects or features based on their spatial relationships.

**2. Why Combine Region Counting with YOLOv8?**

YOLOv8 specializes in the detection and tracking of objects in video streams. Region counting complements this by enabling object counting within designated areas, making it a valuable application of YOLOv8.

**3. How Can I Troubleshoot Issues?**

To gain more insights during inference, you can include the `--debug` flag in your command:

```bash
python yolov8_region_counter.py --source "path to video file" --debug
```

**4. Can I Employ Other YOLO Versions?**

Certainly, you have the flexibility to specify different YOLO model weights using the `--weights` option.

**5. Where Can I Access Additional Information?**

For a comprehensive guide on using YOLOv8 with Object Tracking, please refer to [Multi-Object Tracking with Ultralytics YOLO](https://docs.ultralytics.com/modes/track/).
