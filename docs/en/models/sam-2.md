---
comments: true
description: SAM 2 is Meta's Segment Anything Model for real-time promptable segmentation in images and videos. Learn its architecture, SA-V dataset, benchmarks and usage.
keywords: SAM 2, SAM 2.1, SAM-2, Segment Anything, video segmentation, image segmentation, promptable segmentation, zero-shot segmentation, instance segmentation, interactive segmentation, real-time segmentation, SAM 2 vs YOLO, SAM 2 vs SAM 3, SA-V dataset, Meta, Ultralytics
---

# SAM 2: Segment Anything Model 2

!!! note "SAM Evolution"

    SAM 2 builds upon the original [SAM](sam.md) with video segmentation capabilities. For Promptable Concept Segmentation with text and image exemplar prompts, see [SAM 3](sam-3.md).

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/inference-with-meta-sam-and-sam2-using-ultralytics-python-package.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Inference with Segment Anything 2 In Colab"></a>

SAM 2, the successor to Meta's Segment Anything Model (SAM), is a unified model for promptable object segmentation in both images and videos. Its single promptable architecture supports real-time processing and [zero-shot](https://www.ultralytics.com/glossary/zero-shot-learning) generalization.

!!! tip "SAM 2 on Ultralytics Platform"

    SAM 2.1 models power the [smart annotation feature](https://www.ultralytics.com/annotate) on [Ultralytics Platform](https://platform.ultralytics.com), enabling click-based segmentation for fast dataset labeling. See the [annotation guide](../platform/data/annotation.md) for details.

![SAM 2 example segmentation results](https://cdn.ul.run/i/28aed60b863ee7f03833312fcd2830e7.avif)

## Key Features

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yXQPLMrNX2s"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Inference with Meta's SAM2 using Ultralytics | Step-by-Step Guide
</p>

### Unified Model Architecture

SAM 2 combines the capabilities of image and video segmentation in a single model. This unification simplifies deployment and allows for consistent performance across different media types. It uses a flexible [prompt-based](https://www.ultralytics.com/glossary/visual-prompting) interface, enabling users to specify objects of interest through various prompt types, such as points, bounding boxes, or masks.

### Real-Time Performance

The model achieves real-time inference speeds, processing approximately 44 frames per second. This makes SAM 2 suitable for applications requiring immediate feedback, such as video editing and augmented reality.

### Zero-Shot Generalization

SAM 2 can segment objects it has never encountered before, demonstrating strong zero-shot generalization. This is particularly useful in diverse or evolving visual domains where pre-defined categories may not cover all possible objects.

### Interactive Refinement

Users can iteratively refine the segmentation results by providing additional prompts, allowing for precise control over the output. This interactivity is essential for refining results in applications like video annotation or [medical imaging](https://www.ultralytics.com/glossary/medical-image-analysis).

### Advanced Handling of Visual Challenges

SAM 2 includes mechanisms to manage common video segmentation challenges, such as object occlusion and reappearance. It uses a memory mechanism to keep track of objects across frames, ensuring continuity even when objects are temporarily obscured or exit and re-enter the scene.

For a deeper understanding of SAM 2's architecture and capabilities, explore the [SAM 2 research paper](https://arxiv.org/abs/2408.00714).

## Performance and Technical Details

According to Meta, SAM 2 outperforms previous models on several metrics:

| Metric                                                                                     | SAM 2         | Previous SOTA |
| ------------------------------------------------------------------------------------------ | ------------- | ------------- |
| **Interactive Video Segmentation**                                                         | **Best**      | -             |
| **Human Interactions Required**                                                            | **3x fewer**  | Baseline      |
| **[Image Segmentation](https://www.ultralytics.com/glossary/image-segmentation) Accuracy** | **Improved**  | SAM           |
| **Inference Speed**                                                                        | **6x faster** | SAM           |

## Model Architecture

### Core Components

- **Image and Video Encoder**: Utilizes a [transformer](https://www.ultralytics.com/glossary/transformer)-based architecture to extract high-level features from both images and video frames. This component is responsible for understanding the visual content at each timestep.
- **Prompt Encoder**: Processes user-provided prompts (points, boxes, masks) to guide the segmentation task. This allows SAM 2 to adapt to user input and target specific objects within a scene.
- **Memory Mechanism**: Includes a memory encoder, [memory bank](https://www.ultralytics.com/glossary/memory-bank), and memory attention module. These components collectively store and utilize information from past frames, enabling the model to maintain consistent [object tracking](https://www.ultralytics.com/glossary/object-tracking) over time.
- **Mask Decoder**: Generates the final segmentation masks based on the encoded image features and prompts. In video, it also uses memory context to ensure accurate tracking across frames.

![SAM 2 architecture diagram showing the image encoder, memory mechanism, and mask decoder](https://cdn.ul.run/i/fef83ae8fd67f9a4461ac0a8fab2582c.avif)

### Memory Mechanism and Occlusion Handling

The memory mechanism allows SAM 2 to handle temporal dependencies and occlusions in video data. As objects move and interact, SAM 2 records their features in a memory bank. When an object becomes occluded, the model can rely on this memory to predict its position and appearance when it reappears. The occlusion head specifically handles scenarios where objects are not visible, predicting the likelihood of an object being occluded.

### Multi-Mask Ambiguity Resolution

In situations with ambiguity (e.g., overlapping objects), SAM 2 can generate multiple mask predictions. This helps represent complex scenes where a single mask might not capture the intended object.

## SA-V Dataset

The SA-V dataset, developed for SAM 2's training, is one of the largest and most diverse video segmentation datasets available. It includes:

- **51,000+ Videos**: Captured across 47 countries, providing a wide range of real-world scenarios.
- **600,000+ Mask Annotations**: Detailed spatio-temporal mask annotations, referred to as "masklets," covering whole objects and parts.
- **Dataset Scale**: It features 4.5 times more videos and 53 times more annotations than the previous largest video segmentation datasets.

## Benchmarks

### Video Object Segmentation

SAM 2 reports the following results on major video segmentation benchmarks:

| Dataset         | J&F  | J    | F    |
| --------------- | ---- | ---- | ---- |
| **DAVIS 2017**  | 82.5 | 79.8 | 85.2 |
| **YouTube-VOS** | 81.2 | 78.9 | 83.5 |

### Interactive Segmentation

On interactive segmentation tasks, SAM 2 achieves the following results:

| Dataset               | NoC@90 | AUC   |
| --------------------- | ------ | ----- |
| **DAVIS Interactive** | 1.54   | 0.872 |

## Installation

SAM 2 is included in the `ultralytics` package. Install it with the following command; SAM 2 weights download automatically on first use.

```bash
pip install ultralytics
```

## How to Use SAM 2: Versatility in Image and Video Segmentation

The following table details the available SAM 2 models, their pretrained weights, supported tasks, and compatibility with different operating modes like [Training](../modes/train.md), [Validation](../modes/val.md), [Inference](../modes/predict.md), and [Export](../modes/export.md).

| Model Type    | Pretrained Weights                                                                        | Tasks Supported                              | Training | Validation | Inference | Export |
| ------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------- | -------- | ---------- | --------- | ------ |
| SAM 2 tiny    | [sam2_t.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_t.pt)     | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |
| SAM 2 small   | [sam2_s.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_s.pt)     | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |
| SAM 2 base    | [sam2_b.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_b.pt)     | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |
| SAM 2 large   | [sam2_l.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_l.pt)     | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |
| SAM 2.1 tiny  | [sam2.1_t.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt) | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |
| SAM 2.1 small | [sam2.1_s.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_s.pt) | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |
| SAM 2.1 base  | [sam2.1_b.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt) | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |
| SAM 2.1 large | [sam2.1_l.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_l.pt) | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |

### SAM 2 Prediction Examples

SAM 2 can be used across a broad range of tasks, including real-time video editing, medical imaging, and autonomous systems. Its ability to segment both static and dynamic visual data makes it a versatile tool for researchers and developers.

#### Segment with Prompts

!!! example "Segment with Prompts"

    Use prompts to segment specific objects in images or videos.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM("sam2.1_b.pt")

        # Display model information (optional)
        model.info()

        # Run inference with bboxes prompt
        results = model("path/to/image.jpg", bboxes=[100, 100, 200, 200])

        # Run inference with single point
        results = model(points=[900, 370], labels=[1])

        # Run inference with multiple points
        results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

        # Run inference with multiple points prompt per object
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # Run inference with negative points prompt
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

#### Segment Everything

!!! example "Segment Everything"

    Segment the entire image or video content without specific prompts.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM("sam2.1_b.pt")

        # Display model information (optional)
        model.info()

        # Run inference
        model("path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # Run inference with a SAM 2 model
        yolo predict model=sam2.1_b.pt source=path/to/video.mp4
        ```

- This example demonstrates how SAM 2 can be used to segment the entire content of an image or video if no prompts (bboxes/points/masks) are provided.

#### Segment Video and Track objects

!!! example "Segment Video"

    Segment the entire video content with specific prompts and track objects.

    === "Python"

        ```python
        from ultralytics.models.sam import SAM2VideoPredictor

        # Create SAM2VideoPredictor
        overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "imgsz": 1024, "model": "sam2_b.pt"}
        predictor = SAM2VideoPredictor(overrides=overrides)

        # Run inference with single point
        results = predictor(source="test.mp4", points=[920, 470], labels=[1])

        # Run inference with multiple points
        results = predictor(source="test.mp4", points=[[920, 470], [909, 138]], labels=[1, 1])

        # Run inference with multiple points prompt per object
        results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 1]])

        # Run inference with negative points prompt
        results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 0]])
        ```

## Dynamic Interactive Segment and Track

`SAM2DynamicInteractivePredictor` is a training-free extension of SAM 2 that enables dynamic interaction with multiple frames and continual learning capabilities. This predictor supports real-time prompt updates and memory management for improved tracking performance across a sequence of images. Compared to the original SAM 2, `SAM2DynamicInteractivePredictor` rebuilds the inference flow to make the best use of pretrained SAM 2 models without requiring additional training.

![SAM 2 dynamic interactive segmentation and tracking example](https://cdn.ul.run/i/3673b03a20f4606445fbbbb6d358ac74.avif)

### Key Features

It offers three enhancements:

1. **Dynamic Interactive**: Add prompts for new, untracked instances in later frames at any time during video processing
2. **Continual Learning**: Add new prompts for existing instances to improve the model performance over time
3. **Independent Multi-Image Support**: Process multiple independent images (not necessarily from a video sequence) with memory sharing and cross-image object tracking

### Core Capabilities

- **Prompt Flexibility**: Accepts bounding boxes, points, and masks as prompts
- **Memory Bank Management**: Maintains a dynamic memory bank to store object states across frames
- **Multi-Object Tracking**: Supports tracking multiple objects simultaneously with individual object IDs
- **Real-Time Updates**: Allows adding new prompts during inference without reprocessing previous frames
- **Independent Image Processing**: Process standalone images with shared memory context for cross-image object consistency

!!! example "Dynamic Object Addition"

    === "Python"

        ```python
        from ultralytics.models.sam import SAM2DynamicInteractivePredictor

        # Create SAM2DynamicInteractivePredictor
        overrides = {"conf": 0.01, "task": "segment", "mode": "predict", "imgsz": 1024, "model": "sam2_t.pt", "save": False}
        predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

        # Define a category by box prompt
        predictor(source="image1.jpg", bboxes=[[100, 100, 200, 200]], obj_ids=[0], update_memory=True)

        # Detect this particular object in a new image
        results = predictor(source="image2.jpg")

        # Add new category with a new object ID
        results = predictor(
            source="image4.jpg",
            bboxes=[[300, 300, 400, 400]],  # New object
            obj_ids=[1],  # New object ID
            update_memory=True,  # Add to memory
        )
        # Perform inference
        results = predictor(source="image5.jpg")

        # Add refinement prompts to the same category to boost performance
        # This helps when object appearance changes significantly
        results = predictor(
            source="image6.jpg",
            points=[[150, 150]],  # Refinement point
            labels=[1],  # Positive point
            obj_ids=[1],  # Same object ID
            update_memory=True,  # Update memory with new information
        )
        # Perform inference on new image
        results = predictor(source="image7.jpg")
        ```

!!! note

    The `SAM2DynamicInteractivePredictor` is designed to work with SAM 2 models, and supports adding and refining categories with all the [box, point, and mask prompts](#sam-2-prediction-examples) that SAM 2 supports natively. It is particularly useful for scenarios where objects appear or change over time, such as in video annotation or interactive editing tasks.

#### Arguments

| Name            | Default Value | Data Type   | Description                                 |
| --------------- | ------------- | ----------- | ------------------------------------------- |
| `max_obj_num`   | `3`           | `int`       | The preset maximum number of categories     |
| `update_memory` | `False`       | `bool`      | Whether to update memory with new prompts   |
| `obj_ids`       | `None`        | `List[int]` | List of object IDs corresponding to prompts |

### Use Cases

`SAM2DynamicInteractivePredictor` is ideal for:

- **Video annotation workflows** where new objects appear during the sequence
- **Interactive video editing** requiring real-time object addition and refinement
- **Surveillance applications** with dynamic object tracking needs
- **Medical imaging** for tracking anatomical structures across time series
- **Autonomous systems** requiring adaptive object detection and tracking
- **Multi-image datasets** for consistent object segmentation across independent images
- **Image collection analysis** where objects need to be tracked across different scenes
- **Cross-domain segmentation** leveraging memory from diverse image contexts
- **Semi-automatic annotation** for efficient dataset creation with minimal manual intervention

## SAM Comparison vs YOLO

Here we compare Meta's SAM 2 models, including the smallest SAM2-t variant, with Ultralytics segmentation models, including [YOLO26](yolo26.md):

| Model                                                                                          | Size<br><sup>(MB)</sup> | Parameters<br><sup>(M)</sup> | Speed (CPU)<br><sup>(ms/im)</sup> |
| ---------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| [Meta SAM-b](sam.md)                                                                           | 375                     | 93.7                         | 41703                             |
| Meta SAM2-b                                                                                    | 162                     | 80.8                         | 28867                             |
| Meta SAM2-t                                                                                    | 78.1                    | 38.9                         | 23430                             |
| [MobileSAM](mobile-sam.md)                                                                     | 40.7                    | 10.1                         | 23802                             |
| [FastSAM-s](fast-sam.md) with YOLOv8 [backbone](https://www.ultralytics.com/glossary/backbone) | 23.9                    | 11.8                         | 58.0                              |
| Ultralytics [YOLOv8n-seg](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n-seg)     | 7.1 (11.0x smaller)     | 3.4 (11.4x less)             | 24.8 (945x faster)                |
| Ultralytics [YOLO11n-seg](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n-seg)     | **6.2** (12.6x smaller) | 2.9 (13.4x less)             | **24.3** (964x faster)            |
| Ultralytics [YOLO26n-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-seg)     | 6.7 (11.7x smaller)     | **2.7** (14.4x less)         | 25.2 (930x faster)                |

This comparison demonstrates the substantial differences in model sizes and speeds between SAM variants and YOLO segmentation models. While SAM provides unique automatic segmentation capabilities, YOLO models, particularly YOLOv8n-seg, YOLO11n-seg and YOLO26n-seg, are significantly smaller, faster, and more computationally efficient.

SAM speeds measured with PyTorch, YOLO speeds measured with ONNX Runtime. Tests run on a 2025 Apple M4 Air with 16GB of RAM using `torch==2.10.0`, `ultralytics==8.4.31`, and `onnxruntime==1.24.4`. To reproduce this test:

!!! example

    === "Python"

        ```python
        from ultralytics import ASSETS, SAM, YOLO, FastSAM

        # Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
        for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
            model = SAM(file)
            model.info()
            model(ASSETS)

        # Profile FastSAM-s
        model = FastSAM("FastSAM-s.pt")
        model.info()
        model(ASSETS)

        # Profile YOLO models (ONNX)
        for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt", "yolo26n-seg.pt"]:
            model = YOLO(file_name)
            model.info()
            onnx_path = model.export(format="onnx", dynamic=True, nms=False)  # YOLO26 NMS-free head; no-op for YOLOv8/YOLO11
            model = YOLO(onnx_path)
            model(ASSETS)
        ```

## Auto-Annotation: Efficient Dataset Creation

Auto-annotation pairs SAM 2 with a pretrained detection model to generate segmentation datasets quickly, without extensive manual labeling. Ultralytics Platform also offers browser-based auto-annotation with hosted vision models; compare their detection accuracy, cost and speed on the [VLM leaderboard](https://www.ultralytics.com/benchmarks/vlm).

### How to Auto-Annotate with SAM 2

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/M7xWw4Iodhg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Auto Annotation with Meta's Segment Anything 2 Model using Ultralytics | Data Labeling
</p>

To auto-annotate your dataset with a [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) detector and SAM 2, follow this example:

!!! example "Auto-Annotation Example"

    ```python
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(data="path/to/images", det_model="yolo26x.pt", sam_model="sam2_b.pt")
    ```

{% include "macros/sam-auto-annotate.md" %}

This function turns a folder of unlabeled images into a segmentation dataset with minimal manual effort.

## Limitations

Despite its strengths, SAM 2 has certain limitations:

- **Tracking Stability**: SAM 2 may lose track of objects during extended sequences or significant viewpoint changes.
- **Object Confusion**: The model can sometimes confuse similar-looking objects, particularly in crowded scenes.
- **Efficiency with Multiple Objects**: Segmentation efficiency decreases when processing multiple objects simultaneously due to the lack of inter-object communication.
- **Detail [Accuracy](https://www.ultralytics.com/glossary/accuracy)**: May miss fine details, especially with fast-moving objects. Additional prompts can partially address this issue, but temporal smoothness is not guaranteed.

## Citations and Acknowledgments

If you use SAM 2 in your research or development work, please cite it using the following reference:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @article{ravi2024sam2,
          title={SAM 2: Segment Anything in Images and Videos},
          author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
          journal={arXiv preprint},
          year={2024}
        }
        ```

We thank Meta AI for releasing this model and dataset to the AI community.

## FAQ

### What is SAM 2 and how does it improve upon the original Segment Anything Model (SAM)?

SAM 2, the successor to Meta's Segment Anything Model (SAM), is a unified, promptable model for object segmentation in both images and videos. SAM 2 offers several improvements over the original SAM, including:

- **Unified Model Architecture**: Combines image and video segmentation capabilities in a single model.
- **Real-Time Performance**: Processes approximately 44 frames per second, making it suitable for applications requiring immediate feedback.
- **Zero-Shot Generalization**: Segments objects it has never encountered before, useful in diverse visual domains.
- **Interactive Refinement**: Allows users to iteratively refine segmentation results by providing additional prompts.
- **Advanced Handling of Visual Challenges**: Manages common video segmentation challenges like object occlusion and reappearance.

For more details on SAM 2's architecture and capabilities, explore the [SAM 2 research paper](https://arxiv.org/abs/2408.00714).

### How can I use SAM 2 for real-time video segmentation?

SAM 2 supports real-time video segmentation through its promptable interface and fast inference. Here's a basic example:

!!! example "Segment with Prompts"

    Use prompts to segment specific objects in images or videos.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM("sam2_b.pt")

        # Display model information (optional)
        model.info()

        # Segment with bounding box prompt
        results = model("path/to/image.jpg", bboxes=[100, 100, 200, 200])

        # Segment with point prompt
        results = model("path/to/image.jpg", points=[150, 150], labels=[1])
        ```

For more comprehensive usage, refer to the [How to Use SAM 2](#how-to-use-sam-2-versatility-in-image-and-video-segmentation) section.

### What datasets are used to train SAM 2, and how do they enhance its performance?

SAM 2 is trained on the [SA-V dataset](#sa-v-dataset), one of the largest and most diverse video segmentation datasets available. The SA-V dataset includes:

- **51,000+ Videos**: Captured across 47 countries, providing a wide range of real-world scenarios.
- **600,000+ Mask Annotations**: Detailed spatio-temporal mask annotations, referred to as "masklets," covering whole objects and parts.
- **Dataset Scale**: Features 4.5 times more videos and 53 times more annotations than previous largest datasets, offering unprecedented diversity and complexity.

This dataset allows SAM 2 to achieve strong results on major video segmentation benchmarks and enhances its zero-shot generalization capabilities.

### How does SAM 2 handle occlusions and object reappearances in video segmentation?

SAM 2 includes a memory mechanism to manage temporal dependencies and occlusions in video data. The memory mechanism consists of:

- **Memory Encoder and Memory Bank**: Stores features from past frames.
- **Memory Attention Module**: Utilizes stored information to maintain consistent object tracking over time.
- **Occlusion Head**: Specifically handles scenarios where objects are not visible, predicting the likelihood of an object being occluded.

This mechanism ensures continuity even when objects are temporarily obscured or exit and re-enter the scene. For more details, refer to the [Memory Mechanism and Occlusion Handling](#memory-mechanism-and-occlusion-handling) section.

### How does SAM 2 compare to other segmentation models like YOLO26?

SAM 2 models, such as Meta's SAM2-t and SAM2-b, offer powerful zero-shot segmentation capabilities but are significantly larger and slower compared to YOLO models. For instance, [YOLO26n-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-seg) is approximately **24 times smaller** and over **1145 times faster** than SAM2-b on CPU. While SAM 2 excels in versatile, prompt-based, and zero-shot segmentation scenarios, YOLO26 is optimized for speed, efficiency, and real-time applications with NMS-free end-to-end inference, making it better suited for deployment in resource-constrained environments.
