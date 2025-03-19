---
comments: true
description: YOLOE is a real-time open-vocabulary detection and segmentation model that extends YOLO with text, image, or internal vocabulary prompts, enabling detection of any object class with state-of-the-art zero-shot performance.
keywords: YOLOE, open-vocabulary detection, real-time object detection, instance segmentation, YOLO, text prompts, visual prompts, zero-shot detection
---

# YOLOE: Real-Time Seeing Anything

## Introduction

![YOLOE Prompting Options](https://github.com/THU-MIG/yoloe/raw/main/figures/visualization.svg)

[**YOLOE** (You Only Look Once **E**nhanced)](https://arxiv.org/html/2503.07465v1) is a new advancement in zero-shot, promptable YOLO models, designed for **open-vocabulary** detection and segmentation. Unlike previous YOLO models limited to fixed categories, YOLOE uses text, image, or internal vocabulary prompts, enabling real-time detection of any object class. Built upon YOLOv10 and inspired by [YOLO-World](https://docs.ultralytics.com/models/yolo-world/), YOLOE achieves **state-of-the-art zero-shot performance** with minimal impact on speed and accuracy.

!!! note "Ultralytics Integration Status 🚧"

    The Ultralytics integration for YOLOE is currently under construction 🔨. The usage examples shown in this documentation will work once the integration is complete ✅. Please check back for updates 🔄 or follow our [GitHub repository](https://github.com/ultralytics/ultralytics) 🚀 for the latest developments.
  
Compared to earlier YOLO models, YOLOE significantly boosts efficiency and accuracy. It improves by **+3.5 AP** over YOLO-Worldv2 on LVIS while using just a third of the training resources and achieving 1.4× faster inference speeds. Fine-tuned on COCO, YOLOE-large surpasses YOLOv8-L by **~0.6 mAP**, using nearly **4× less training time**. This demonstrates YOLOE's exceptional balance of accuracy, efficiency, and versatility. The sections below explore YOLOE's architecture, benchmark comparisons, and integration with the [Ultralytics](https://www.ultralytics.com/) framework.

## Architecture Overview

<p align="center">
  <img src="https://github.com/THU-MIG/yoloe/raw/main/figures/pipeline.svg" alt="YOLOE Architecture" width=90%>
</p>

YOLOE retains the standard YOLO structure—a convolutional **backbone** (e.g., CSP-Darknet) for feature extraction, a **neck** (e.g., PAN-FPN) for multi-scale fusion, and an **anchor-free, decoupled** detection **head** (as in YOLOv8/YOLO11) predicting objectness, classes, and boxes independently. YOLOE introduces three novel modules enabling open-vocabulary detection:

- **Re-parameterizable Region-Text Alignment (RepRTA)**: Supports **text-prompted detection** by refining text [embeddings](https://www.ultralytics.com/glossary/embeddings) (e.g., from CLIP) via a small auxiliary network. At inference, this network is folded into the main model, ensuring zero overhead. YOLOE thus detects arbitrary text-labeled objects (e.g., unseen "traffic light") without runtime penalties.

- **Semantic-Activated Visual Prompt Encoder (SAVPE)**: Enables **visual-prompted detection** via a lightweight embedding branch. Given a reference image, SAVPE encodes semantic and activation features, conditioning the model to detect visually similar objects—a one-shot detection capability useful for logos or specific parts.

- **Lazy Region-Prompt Contrast (LRPC)**: In **prompt-free mode**, YOLOE performs open-set recognition using internal embeddings trained on large vocabularies (1200+ categories from LVIS and Objects365). Without external prompts or encoders, YOLOE identifies objects via embedding similarity lookup, efficiently handling large label spaces at inference.

Additionally, YOLOE integrates real-time **instance segmentation** by extending the detection head with a mask prediction branch (similar to YOLACT or YOLOv8-Seg), adding minimal overhead.

Crucially, YOLOE's open-world modules introduce **no inference cost** when used as a regular closed-set YOLO. Post-training, YOLOE parameters can be re-parameterized into a standard YOLO head, preserving identical FLOPs and speed (e.g., matching [YOLO11](https://docs.ultralytics.com/models/yolo11/) exactly).

## YOLOE Performance Comparison

YOLOE matches or exceeds the accuracy of closed-set YOLO models on standard benchmarks like COCO, without compromising speed or model size. The table below compares YOLOE-L (built on YOLO11) against corresponding [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and YOLO11 models:

| Model                     | COCO mAP<sub>50-95</sub> | Inference Speed (T4)             | Parameters | GFLOPs (640px)     |
|---------------------------|--------------------------|----------------------------------|------------|--------------------|
| **YOLOv8-L** (closed-set) | 52.9%                    | **9.06 ms** (110 FPS)            | 43.7 M     | 165.2 B            |
| **YOLO11-L** (closed-set) | ~53%                     | **7.7 ms**<sup>†</sup> (130 FPS) | 26.2 M     | 232.0 B            |
| **YOLOE-L** (open-vocab)  | ~53.5%                   | **7.7 ms** (130 FPS)             | 26.2 M     | ~232 B<sup>†</sup> |

<sup>†</sup> *YOLO11-L and YOLOE-L have identical architectures (prompt modules disabled in YOLO11-L), resulting in identical inference speed and similar GFLOPs estimates.*

YOLOE-L achieves **~53.5% mAP**, surpassing YOLOv8-L (**52.9%**) with roughly **40% fewer parameters** (26M vs. 43.7M). It processes 640×640 images in **7.7 ms (130 FPS)** compared to YOLOv8-L's **9.06 ms (110 FPS)**, highlighting YOLO11's efficiency. Crucially, YOLOE's open-vocabulary modules incur **no inference cost**, demonstrating a **"no free lunch trade-off"** design.

For zero-shot and transfer tasks, YOLOE excels: on LVIS, YOLOE-small improves over YOLO-Worldv2 by **+3.5 AP** using **3× less training resources**. Fine-tuning YOLOE-L from LVIS to COCO also required **4× less training time** than YOLOv8-L, underscoring its efficiency and adaptability. YOLOE further maintains YOLO's hallmark speed, achieving **300+ FPS** on a T4 GPU and **~64 FPS** on iPhone 12 via CoreML, ideal for edge and mobile deployments.

!!! note

    **Benchmark conditions:** YOLOE results are from models pre-trained on Objects365, GoldG, and LVIS, then fine-tuned or evaluated on COCO. YOLOE's slight mAP advantage over YOLOv8 comes from extensive pre-training. Without this open-vocab training, YOLOE matches similar-sized YOLO models, affirming its SOTA accuracy and open-world flexibility without performance penalties.

## Comparison with Previous Models

YOLOE introduces notable advancements over prior YOLO models and open-vocabulary detectors:

- **YOLOE vs YOLOv5:**  
  [YOLOv5](yolov5.md) offered good speed-accuracy balance but required retraining for new classes and used anchor-based heads. In contrast, YOLOE is **anchor-free** and dynamically detects new classes. YOLOE, building on YOLOv8's improvements, achieves higher accuracy (~53% vs. YOLOv5's ~50% mAP on COCO) and integrates instance segmentation, unlike YOLOv5.

- **YOLOE vs YOLOv8:**  
  YOLOE extends [YOLOv8](yolov8.md)'s redesigned architecture, achieving similar or superior accuracy (**~53.5% mAP with ~26M parameters** vs. YOLOv8-L's **52.9% with ~44M parameters**). It significantly reduces training time due to stronger pre-training. The key advancement is YOLOE's **open-world capability**, detecting unseen objects (e.g., "**bird scooter**" or "**peace symbol**") via prompts, unlike YOLOv8's closed-set design.

- **YOLOE vs YOLO11:**  
  [YOLO11](yolo11.md) improves upon YOLOv8 with enhanced efficiency and fewer parameters (~22% reduction). YOLOE inherits these gains directly, matching YOLO11's inference speed and parameter count (~26M parameters), while adding **open-vocabulary detection and segmentation**. In closed-set scenarios, YOLOE is equivalent to YOLO11, but crucially adds adaptability to detect unseen classes, achieving **YOLO11 + open-world capability** without compromising speed.

- **YOLOE vs previous open-vocabulary detectors:**  
  Earlier open-vocab models (GLIP, OWL-ViT, [YOLO-World](yolo-world.md)) relied heavily on vision-language [transformers](https://www.ultralytics.com/glossary/transformer), leading to slow inference. YOLOE surpasses these in zero-shot accuracy (e.g., **+3.5 AP vs. YOLO-Worldv2**) while running **1.4× faster** with significantly lower training resources. Compared to transformer-based approaches (e.g., GLIP), YOLOE offers orders-of-magnitude faster inference, effectively bridging the accuracy-efficiency gap in open-set detection.

In summary, YOLOE maintains YOLO's renowned speed and efficiency, surpasses predecessors in accuracy, integrates segmentation, and introduces powerful open-world detection, making it uniquely versatile and practical.

## Use Cases and Applications

YOLOE's open-vocabulary detection and segmentation enable diverse applications beyond traditional fixed-class models:

- **Open-World Object Detection:**  
  Ideal for dynamic scenarios like **robotics**, where robots recognize previously unseen objects using prompts, or **security systems** quickly adapting to new threats (e.g., hazardous items) without retraining.

- **Few-Shot and One-Shot Detection:**  
  Using visual prompts (SAVPE), YOLOE rapidly learns new objects from single reference images—perfect for **industrial inspection** (identifying parts or defects instantly) or **custom surveillance**, enabling visual searches with minimal setup.

- **Large-Vocabulary & Long-Tail Recognition:**  
  Equipped with a vocabulary of 1000+ classes, YOLOE excels in tasks like **biodiversity monitoring** (detecting rare species), **museum collections**, **retail inventory**, or **e-commerce**, reliably identifying many classes without extensive per-class training.

- **Interactive Detection and Segmentation:**  
  YOLOE supports real-time interactive applications such as **searchable video/image retrieval**, **augmented reality (AR)**, and intuitive **image editing**, driven by natural inputs (text or visual prompts). Users can dynamically isolate, identify, or edit objects precisely using segmentation masks.

- **Automated Data Labeling and Bootstrapping:**  
  YOLOE facilitates rapid dataset creation by providing initial bounding box and segmentation annotations, significantly reducing human labeling efforts. Particularly valuable in **analytics of large media collections**, where it can auto-identify objects present, assisting in building specialized models faster.

- **Segmentation for Any Object:**  
  Extends segmentation capabilities to arbitrary objects through prompts—particularly beneficial for **medical imaging**, **microscopy**, or **satellite imagery analysis**, automatically identifying and precisely segmenting structures without specialized pre-trained models. Unlike models like [SAM](https://docs.ultralytics.com/models/sam/), YOLOE simultaneously recognizes and segments objects automatically, aiding in tasks like **content creation** or **scene understanding**.

Across all these use cases, YOLOE's core advantage is **versatility**, providing a unified model for detection, recognition, and segmentation across dynamic scenarios. Its efficiency ensures real-time performance on resource-constrained devices, ideal for robotics, autonomous driving, defense, and beyond.

!!! tip

    Choose YOLOE's mode based on your needs:
    - **Closed-set mode:** For fixed-class tasks (max speed and accuracy).
    - **Prompted mode:** Add new objects quickly via text or visual prompts.
    - **Prompt-free open-set mode:** General detection across many categories (ideal for cataloging and discovery).  

    Often, combining modes—such as prompt-free discovery followed by targeted prompts—leverages YOLOE's full potential.

## Training and Inference

YOLOE integrates seamlessly with the [Ultralytics Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/), similar to other YOLO models (YOLOv8, YOLO-World). Here's how to quickly get started:

!!! note "Ultralytics Integration Status 🚧"

    The Ultralytics integration for YOLOE is currently under development 🔨. The examples below demonstrate how the API will work once integration is complete ✅.
 
!!! Example "Training and inference with YOLOE"

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load pre-trained YOLOE model and train on custom data
        model = YOLO("yoloe-s.pt")
        model.train(data="path/to/data.yaml", epochs=50, imgsz=640)
        
        # Run inference using text prompts ("person", "bus")
        model.set_classes(["person", "bus"])
        results = model.predict(source="test_images/street.jpg")
        results[0].save()  # save annotated output
        ```
    
        Here, YOLOE behaves like a standard detector by default but easily switches to prompted detection by specifying classes (`set_classes`). Results contain bounding boxes, masks, and labels.
    
    === "CLI"
    
        ```bash
        # Training YOLOE on custom dataset
        yolo train model=yoloe-s.pt data=path/to/data.yaml epochs=50 imgsz=640
        
        # Inference with text prompts
        yolo predict model=yoloe-s.pt source="test_images/street.jpg" classes="person,bus"
        ```
    
        CLI prompts (`classes`) guide YOLOE similarly to Python's `set_classes`. Visual prompting (image-based queries) currently requires the Python API.

### Other Supported Tasks

- **Validation:** Evaluate accuracy easily with `model.val()` or `yolo val`.
- **Export:** Export YOLOE models (`model.export()`) to ONNX, TensorRT, etc., facilitating deployment.
- **Tracking:** YOLOE supports object tracking (`yolo track`) when integrated, useful for tracking prompted classes in videos.

!!! note

    YOLOE automatically includes **segmentation masks** in inference results (`results[0].masks`), simplifying pixel-precise tasks like object extraction or measurement without needing separate models.

## Getting Started

Quickly set up YOLOE with Ultralytics by following these steps:

1. **Installation**:
   Install or update the Ultralytics package:
   ```bash
   pip install -U ultralytics
   ```

2. **Download YOLOE Weights**:
   Pre-trained YOLOE models (e.g., YOLOE-v8-S/L, YOLOE-11 variants) are available from the YOLOE GitHub releases. Simply download your desired `.pt` file to load into the Ultralytics YOLO class.

3. **Hardware Requirements**:
    - **Inference**: Recommended GPU (NVIDIA with ≥4-8GB VRAM). Small models run efficiently on edge GPUs (e.g., Jetson) or CPUs at lower resolutions.
    - **Training**: Fine-tuning YOLOE on custom data typically requires just one GPU. Extensive open-vocabulary pre-training (LVIS/Objects365) used by authors required substantial compute (8× RTX 4090 GPUs).

4. **Configuration**:
   YOLOE configurations use standard Ultralytics YAML files. Default configs (e.g., `yoloe-s.yaml`) typically suffice, but you can modify backbone, classes, or image size as needed.

5. **Running YOLOE**:
    - **Quick inference** (prompt-free):
      ```bash
      yolo predict model=yoloe-s.pt source="image.jpg"
      ```
    - **Prompted detection** (text prompt example):
      ```bash
      yolo predict model=yoloe-s.pt source="kitchen.jpg" classes="bowl,apple"
      ```
      In Python:
      ```python
      from ultralytics import YOLO
 
      model = YOLO("yoloe-s.pt")
      model.set_classes(["bowl", "apple"])
      results = model.predict("kitchen.jpg")
      results[0].save()
      ```

6. **Integration Tips**:
    - **Class names**: Default YOLOE outputs use LVIS categories; use `set_classes()` to specify your own labels.
    - **Speed**: YOLOE has no overhead unless using prompts. Text prompts have minimal impact; visual prompts slightly more.
    - **Batch inference**: Supported directly (`model.predict([img1, img2])`). For image-specific prompts, run images individually.

The [Ultralytics documentation](https://docs.ultralytics.com/) provides further resources. YOLOE lets you easily explore powerful open-world capabilities within the familiar YOLO ecosystem.

!!! tip

    **Pro Tip:**  
    To maximize YOLOE's zero-shot accuracy, fine-tune from provided checkpoints rather than training from scratch. Use prompt words aligning with common training labels (see LVIS categories) to improve detection accuracy.

## Citations and Acknowledgements

If YOLOE has contributed to your research or project, please cite the original paper by **Ao Wang, Lihao Liu, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding** from **Tsinghua University**:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{wang2025yoloerealtimeseeing,
              title={YOLOE: Real-Time Seeing Anything}, 
              author={Ao Wang and Lihao Liu and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
              year={2025},
              eprint={2503.07465},
              archivePrefix={arXiv},
              primaryClass={cs.CV},
              url={https://arxiv.org/abs/2503.07465}, 
        }
        ```

For further reading, the original YOLOE paper is available on [arXiv](https://arxiv.org/html/2503.07465v1). The project's source code and additional resources can be accessed via their [GitHub repository](https://github.com/THU-MIG/yoloe).

## FAQ

### How does YOLOE differ from YOLO-World?

While both YOLOE and [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) enable open-vocabulary detection, YOLOE offers several advantages. YOLOE achieves +3.5 AP higher accuracy on LVIS while using 3× less training resources and running 1.4× faster than YOLO-Worldv2. YOLOE also supports three prompting modes (text, visual, and internal vocabulary), whereas YOLO-World primarily focuses on text prompts. Additionally, YOLOE includes built-in instance segmentation capabilities, providing pixel-precise masks for detected objects without additional overhead.

### Can I use YOLOE as a regular YOLO model?

Yes, YOLOE can function exactly like a standard YOLO model with no performance penalty. When used in closed-set mode (without prompts), YOLOE's open-vocabulary modules are re-parameterized into the standard detection head, resulting in identical speed and accuracy to equivalent YOLO11 models. This makes YOLOE extremely versatile—you can use it as a traditional detector for maximum speed and then switch to open-vocabulary mode only when needed.

### What types of prompts can I use with YOLOE?

YOLOE supports three types of prompts:

1. **Text prompts**: Specify object classes using natural language (e.g., "person", "traffic light", "bird scooter")
2. **Visual prompts**: Provide reference images of objects you want to detect
3. **Internal vocabulary**: Use YOLOE's built-in vocabulary of 1200+ categories without external prompts

This flexibility allows you to adapt YOLOE to various scenarios without retraining the model, making it particularly useful for dynamic environments where detection requirements change frequently.

### How does YOLOE handle instance segmentation?

YOLOE integrates instance segmentation directly into its architecture by extending the detection head with a mask prediction branch. This approach is similar to YOLOv8-Seg but works for any prompted object class. Segmentation masks are automatically included in inference results and can be accessed via `results[0].masks`. This unified approach eliminates the need for separate detection and segmentation models, streamlining workflows for applications requiring pixel-precise object boundaries.

### How does YOLOE handle inference with custom prompts?

Similar to [YOLO-World](https://docs.ultralytics.com/models/yolo-world/), YOLOE supports a "prompt-then-detect" strategy that utilizes an offline vocabulary to enhance efficiency. Custom prompts like captions or specific object categories are pre-encoded and stored as offline vocabulary embeddings. This approach streamlines the detection process without requiring retraining. You can dynamically set these prompts within the model to tailor it to specific detection tasks:

```python
from ultralytics import YOLO

# Initialize a YOLOE model
model = YOLO("yoloe-s.pt")

# Define custom classes
model.set_classes(["person", "bus"])

# Execute prediction on an image
results = model.predict("path/to/image.jpg")

# Show results
results[0].show()
```
