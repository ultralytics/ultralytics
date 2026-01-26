---
comments: true
description: Learn to perform real-time action recognition using Ultralytics YOLO26 combined with TorchVision video classification models.
keywords: action recognition, Ultralytics YOLO26, video classification, TorchVision, Kinetics-400, real-time, computer vision
---

# Action Recognition using Ultralytics YOLO26

Action recognition combines [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) object detection with TorchVision video classification models to identify human actions in real-time video streams. The system tracks individuals and classifies their activities using models pretrained on Kinetics-400.

## Installation

```bash
pip install ultralytics torchvision
```

## Advantages of Action Recognition

- **Real-time Analysis:** Process video streams and identify actions as they happen.
- **Pretrained Models:** Uses TorchVision models trained on Kinetics-400 (400 action classes).
- **Multiple Architectures:** Support for S3D, R3D, Swin3D, and MViT models.
- **Integrated Tracking:** Combines YOLO detection with per-person action classification.

!!! example "Action Recognition Example"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        action = solutions.ActionRecognition(
            show=True,
            model="yolo26n.pt",
            video_classifier_model="s3d",  # TorchVision model
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            results = action(im0)
            video_writer.write(results.plot_im)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Webcam"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "Error accessing webcam"

        action = solutions.ActionRecognition(show=True, video_classifier_model="s3d")

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            action(im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        ```

!!! tip "What You'll See"

    - **"detecting..."** appears while collecting frames (first ~16 frames)
    - **Action labels** with confidence appear after enough frames (e.g., "walking (0.85)")
    - Each tracked person gets their own action prediction

### Video Classifier Models

All models are pretrained on [Kinetics-400](https://www.deepmind.com/open-source/kinetics) with 400 action classes.

| Model       | Description                    |
| ----------- | ------------------------------ |
| `s3d`       | S3D - fast and lightweight     |
| `r3d_18`    | R3D-18 - good balance          |
| `swin3d_t`  | Swin Transformer 3D (tiny)     |
| `swin3d_b`  | Swin Transformer 3D (base)     |
| `mvit_v1_b` | MViT v1 (base)                 |
| `mvit_v2_s` | MViT v2 (small)                |

### Arguments

| Argument                     | Type  | Default     | Description                              |
| ---------------------------- | ----- | ----------- | ---------------------------------------- |
| `model`                      | `str` | `yolo26n.pt`| YOLO model for person detection          |
| `video_classifier_model`     | `str` | `s3d`       | TorchVision video classifier model       |
| `crop_margin_percentage`     | `int` | `10`        | Margin percentage for person crop        |
| `num_video_sequence_samples` | `int` | `8`         | Number of frames per sequence            |
| `skip_frame`                 | `int` | `2`         | Frame skip interval                      |

## FAQ

### What action classes are supported?

The models are trained on Kinetics-400 which includes 400 action classes like: walking, running, jumping, swimming, playing guitar, cooking, etc. See the [full list](https://github.com/google-deepmind/kinetics-i3d/blob/master/data/label_map.txt).

### Why do I see "detecting..." instead of action labels?

The system collects 8 frames before making a prediction. With `skip_frame=2`, this takes ~16 video frames. Action labels appear after enough frames are collected.

### Which model should I use?

- **`s3d`** - Fastest, good for real-time on CPU
- **`r3d_18`** - Good balance of speed and accuracy
- **`swin3d_t`** / **`mvit_v2_s`** - More accurate but slower

### How do I get the output image?

```python
results = action(frame)
cv2.imwrite("output.jpg", results.plot_im)
```
