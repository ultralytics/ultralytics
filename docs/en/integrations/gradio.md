---
comments: true
description: Learn to use Gradio and Ultralytics YOLOv8 for interactive object detection. Upload images and adjust detection parameters in real-time.
keywords: Gradio, Ultralytics YOLOv8, object detection, interactive AI, Python
---

# Interactive Object Detection: Gradio & Ultralytics YOLOv8 🚀

## Introduction to Interactive Object Detection

This Gradio interface provides an easy and interactive way to perform object detection using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) model. Users can upload images and adjust parameters like confidence threshold and intersection-over-union (IoU) threshold to get real-time detection results.

## Why Use Gradio for Object Detection?

* **User-Friendly Interface:** Gradio offers a straightforward platform for users to upload images and visualize detection results without any coding requirement.
* **Real-Time Adjustments:** Parameters such as confidence and IoU thresholds can be adjusted on the fly, allowing for immediate feedback and optimization of detection results.
* **Broad Accessibility:** The Gradio web interface can be accessed by anyone, making it an excellent tool for demonstrations, educational purposes, and quick experiments.

<p align="center">
   <img width="800" alt="Gradio example screenshot" src="https://github.com/RizwanMunawar/ultralytics/assets/26833433/52ee3cd2-ac59-4c27-9084-0fd05c6c33be">
</p>

## How to Install the Gradio

```bash
pip install gradio
```

## How to Use the Interface

1. **Upload Image:** Click on 'Upload Image' to choose an image file for object detection.
2. **Adjust Parameters:**
    * **Confidence Threshold:** Slider to set the minimum confidence level for detecting objects.
    * **IoU Threshold:** Slider to set the IoU threshold for distinguishing different objects.
3. **View Results:** The processed image with detected objects and their labels will be displayed.

## Example Use Cases

* **Sample Image 1:** Bus detection with default thresholds.
* **Sample Image 2:** Detection on a sports image with default thresholds.

## Usage Example

This section provides the Python code used to create the Gradio interface with the Ultralytics YOLOv8 model. Supports classification tasks, detection tasks, segmentation tasks, and key point tasks.

```python
import PIL.Image as Image
import gradio as gr

from ultralytics import ASSETS, YOLO

model = YOLO("yolov8n.pt")


def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics Gradio",
    description="Upload images for inference. The Ultralytics YOLOv8n model is used by default.",
    examples=[
        [ASSETS / "bus.jpg", 0.25, 0.45],
        [ASSETS / "zidane.jpg", 0.25, 0.45],
    ]
)

if __name__ == '__main__':
    iface.launch()
```

## Parameters Explanation

| Parameter Name   | Type    | Description                                              |
|------------------|---------|----------------------------------------------------------|
| `img`            | `Image` | The image on which object detection will be performed.   |
| `conf_threshold` | `float` | Confidence threshold for detecting objects.              |
| `iou_threshold`  | `float` | Intersection-over-union threshold for object separation. |

### Gradio Interface Components

| Component    | Description                              |
|--------------|------------------------------------------|
| Image Input  | To upload the image for detection.       |
| Sliders      | To adjust confidence and IoU thresholds. |
| Image Output | To display the detection results.        |
