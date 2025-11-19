---
comments: true
description: Discover SAM 2, the next generation of Meta's Segment Anything Model, supporting real-time promptable segmentation in both images and videos with state-of-the-art performance. Learn about its key features, datasets, and how to use it.
keywords: SAM 2, SAM 2.1, Segment Anything, video segmentation, image segmentation, promptable segmentation, zero-shot performance, SA-V dataset, Ultralytics, real-time segmentation, AI, machine learning
---

# SAM 2: Segment Anything Model 2

!!! note "SAM Evolution"

    SAM 2 builds upon the original [SAM](sam.md) with video segmentation capabilities. For the upcoming next-generation model with enhanced memory and 2× faster processing, see [SAM 3](sam-3.md).

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/inference-with-meta-sam-and-sam2-using-ultralytics-python-package.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Inference with Segment Anything 2 In Colab"></a>

SAM 2, the successor to Meta's [Segment Anything Model (SAM)](sam.md), is a cutting-edge tool designed for comprehensive object segmentation in both images and videos. It excels in handling complex visual data through a unified, promptable model architecture that supports real-time processing and zero-shot generalization.

![SAM 2 Example Results](https://github.com/ultralytics/docs/releases/download/0/sa-v-dataset.avif)

## Key Features

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yXQPLMrNX2s"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Inference with Meta's SAM2 using Ultralytics | Step-by-Step Guide 🎉
</p>

### Unified Model Architecture

SAM 2 combines the capabilities of image and video segmentation in a single model. This unification simplifies deployment and allows for consistent performance across different media types. It leverages a flexible prompt-based interface, enabling users to specify objects of interest through various prompt types, such as points, bounding boxes, or masks.

### Real-Time Performance

The model achieves real-time inference speeds, processing approximately 44 frames per second. This makes SAM 2 suitable for applications requiring immediate feedback, such as video editing and augmented reality.

### Zero-Shot Generalization

SAM 2 can segment objects it has never encountered before, demonstrating strong zero-shot generalization. This is particularly useful in diverse or evolving visual domains where pre-defined categories may not cover all possible objects.

### Interactive Refinement

Users can iteratively refine the segmentation results by providing additional prompts, allowing for precise control over the output. This interactivity is essential for fine-tuning results in applications like video annotation or medical imaging.

### Advanced Handling of Visual Challenges

SAM 2 includes mechanisms to manage common video segmentation challenges, such as object occlusion and reappearance. It uses a sophisticated memory mechanism to keep track of objects across frames, ensuring continuity even when objects are temporarily obscured or exit and re-enter the scene.

For a deeper understanding of SAM 2's architecture and capabilities, explore the [SAM 2 research paper](https://arxiv.org/abs/2408.00714).

## Performance and Technical Details

SAM 2 sets a new benchmark in the field, outperforming previous models on various metrics:

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
- **Memory Mechanism**: Includes a memory encoder, memory bank, and memory attention module. These components collectively store and utilize information from past frames, enabling the model to maintain consistent [object tracking](https://www.ultralytics.com/glossary/object-tracking) over time.
- **Mask Decoder**: Generates the final segmentation masks based on the encoded image features and prompts. In video, it also uses memory context to ensure accurate tracking across frames.

![SAM 2 Architecture Diagram](https://github.com/ultralytics/docs/releases/download/0/sam2-architecture-diagram.avif)

### Memory Mechanism and Occlusion Handling

The memory mechanism allows SAM 2 to handle temporal dependencies and occlusions in video data. As objects move and interact, SAM 2 records their features in a memory bank. When an object becomes occluded, the model can rely on this memory to predict its position and appearance when it reappears. The occlusion head specifically handles scenarios where objects are not visible, predicting the likelihood of an object being occluded.

### Multi-Mask Ambiguity Resolution

In situations with ambiguity (e.g., overlapping objects), SAM 2 can generate multiple mask predictions. This feature is crucial for accurately representing complex scenes where a single mask might not sufficiently describe the scene's nuances.

## SA-V Dataset

The SA-V dataset, developed for SAM 2's training, is one of the largest and most diverse video segmentation datasets available. It includes:

- **51,000+ Videos**: Captured across 47 countries, providing a wide range of real-world scenarios.
- **600,000+ Mask Annotations**: Detailed spatio-temporal mask annotations, referred to as "masklets," covering whole objects and parts.
- **Dataset Scale**: It features 4.5 times more videos and 53 times more annotations than previous largest datasets, offering unprecedented diversity and complexity.

## Benchmarks

### Video Object Segmentation

SAM 2 has demonstrated superior performance across major video segmentation benchmarks:

| Dataset         | J&F  | J    | F    |
| --------------- | ---- | ---- | ---- |
| **DAVIS 2017**  | 82.5 | 79.8 | 85.2 |
| **YouTube-VOS** | 81.2 | 78.9 | 83.5 |

### Interactive Segmentation

In interactive segmentation tasks, SAM 2 shows significant efficiency and accuracy:

| Dataset               | NoC@90 | AUC   |
| --------------------- | ------ | ----- |
| **DAVIS Interactive** | 1.54   | 0.872 |

## Installation

To install SAM 2, use the following command. All SAM 2 models will automatically download on first use.

```bash
pip install ultralytics
```

## How to Use SAM 2: Versatility in Image and Video Segmentation

The following table details the available SAM 2 models, their pre-trained weights, supported tasks, and compatibility with different operating modes like [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md).

| Model Type    | Pre-trained Weights                                                                       | Tasks Supported                              | Inference | Validation | Training | Export |
| ------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| SAM 2 tiny    | [sam2_t.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_t.pt)     | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 2 small   | [sam2_s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_s.pt)     | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 2 base    | [sam2_b.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt)     | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 2 large   | [sam2_l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_l.pt)     | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 2.1 tiny  | [sam2.1_t.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 2.1 small | [sam2.1_s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_s.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 2.1 base  | [sam2.1_b.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 2.1 large | [sam2.1_l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |

### SAM 2 Prediction Examples

SAM 2 can be utilized across a broad spectrum of tasks, including real-time video editing, medical imaging, and autonomous systems. Its ability to segment both static and dynamic visual data makes it a versatile tool for researchers and developers.

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

#### Segment Video and Track objects

!!! example "Segment Video"

    Segment the entire video content with specific prompts and track objects.

    === "Python"

        ```python
        from ultralytics.models.sam import SAM2VideoPredictor

        # Create SAM2VideoPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
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

- This example demonstrates how SAM 2 can be used to segment the entire content of an image or video if no prompts (bboxes/points/masks) are provided.

## Dynamic Interactive Segment and Track

SAM2DynamicInteractivePredictor is an advanced training-free extension of SAM2 that enables dynamic interaction with multiple frames and continual learning capabilities. This predictor supports real-time prompt updates and memory management for improved tracking performance across a sequence of images. Compared to the original SAM2, SAM2DynamicInteractivePredictor rebuilds the inference flow to make the best use of pre-trained SAM2 models without requiring additional training.

![SAM 2 Example Results](https://github.com/ultralytics/assets/releases/download/v0.0.0/sam2-interative-sample.avif)

### Key Features

It offers three significant enhancements:

1. **Dynamic Interactive**: Add new prompts for merging/untracked new instances in following frames anytime during video processing
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
        overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_t.pt", save=False)
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

    The `SAM2DynamicInteractivePredictor` is designed to work with SAM2 models, and support adding/refining categories by all the [box/point/mask prompts](#sam-2-prediction-examples) natively that SAM2 supports. It is particularly useful for scenarios where objects appear or change over time, such as in video annotation or interactive editing tasks.

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

## SAM 2 Comparison vs YOLO

Here we compare Meta's SAM 2 models, including the smallest SAM2-t variant, with Ultralytics smallest segmentation model, [YOLO11n-seg](../tasks/segment.md):

| Model                                                                                          | Size<br><sup>(MB)</sup> | Parameters<br><sup>(M)</sup> | Speed (CPU)<br><sup>(ms/im)</sup> |
| ---------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| [Meta SAM-b](sam.md)                                                                           | 375                     | 93.7                         | 49401                             |
| Meta SAM2-b                                                                                    | 162                     | 80.8                         | 31901                             |
| Meta SAM2-t                                                                                    | 78.1                    | 38.9                         | 25997                             |
| [MobileSAM](mobile-sam.md)                                                                     | 40.7                    | 10.1                         | 25381                             |
| [FastSAM-s](fast-sam.md) with YOLOv8 [backbone](https://www.ultralytics.com/glossary/backbone) | 23.7                    | 11.8                         | 55.9                              |
| Ultralytics [YOLOv8n-seg](yolov8.md)                                                           | **6.7** (11.7x smaller) | **3.4** (11.4x less)         | **24.5** (1061x faster)           |
| Ultralytics [YOLO11n-seg](yolo11.md)                                                           | **5.9** (13.2x smaller) | **2.9** (13.4x less)         | **30.1** (864x faster)            |

This comparison demonstrates the substantial differences in model sizes and speeds between SAM variants and YOLO segmentation models. While SAM provides unique automatic segmentation capabilities, YOLO models, particularly YOLOv8n-seg and YOLO11n-seg, are significantly smaller, faster, and more computationally efficient.

Tests run on a 2025 Apple M4 Pro with 24GB of RAM using `torch==2.6.0` and `ultralytics==8.3.90`. To reproduce this test:

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

        # Profile YOLO models
        for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
            model = YOLO(file_name)
            model.info()
            model(ASSETS)
        ```

## Auto-Annotation: Efficient Dataset Creation

Auto-annotation is a powerful feature of SAM 2, enabling users to generate segmentation datasets quickly and accurately by leveraging pre-trained models. This capability is particularly useful for creating large, high-quality datasets without extensive manual effort.

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

To auto-annotate your dataset using SAM 2, follow this example:

!!! example "Auto-Annotation Example"

    ```python
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam2_b.pt")
    ```

{% include "macros/sam-auto-annotate.md" %}

This function facilitates the rapid creation of high-quality segmentation datasets, ideal for researchers and developers aiming to accelerate their projects.

## Limitations

Despite its strengths, SAM 2 has certain limitations:

- **Tracking Stability**: SAM 2 may lose track of objects during extended sequences or significant viewpoint changes.
- **Object Confusion**: The model can sometimes confuse similar-looking objects, particularly in crowded scenes.
- **Efficiency with Multiple Objects**: Segmentation efficiency decreases when processing multiple objects simultaneously due to the lack of inter-object communication.
- **Detail [Accuracy](https://www.ultralytics.com/glossary/accuracy)**: May miss fine details, especially with fast-moving objects. Additional prompts can partially address this issue, but temporal smoothness is not guaranteed.

## Citations and Acknowledgments

If SAM 2 is a crucial part of your research or development work, please cite it using the following reference:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{ravi2024sam2,
          title={SAM 2: Segment Anything in Images and Videos},
          author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
          journal={arXiv preprint},
          year={2024}
        }
        ```

We extend our gratitude to Meta AI for their contributions to the AI community with this groundbreaking model and dataset.

## FAQ

### What is SAM 2 and how does it improve upon the original Segment Anything Model (SAM)?

SAM 2, the successor to Meta's [Segment Anything Model (SAM)](sam.md), is a cutting-edge tool designed for comprehensive object segmentation in both images and videos. It excels in handling complex visual data through a unified, promptable model architecture that supports real-time processing and zero-shot generalization. SAM 2 offers several improvements over the original SAM, including:

- **Unified Model Architecture**: Combines image and video segmentation capabilities in a single model.
- **Real-Time Performance**: Processes approximately 44 frames per second, making it suitable for applications requiring immediate feedback.
- **Zero-Shot Generalization**: Segments objects it has never encountered before, useful in diverse visual domains.
- **Interactive Refinement**: Allows users to iteratively refine segmentation results by providing additional prompts.
- **Advanced Handling of Visual Challenges**: Manages common video segmentation challenges like object occlusion and reappearance.

For more details on SAM 2's architecture and capabilities, explore the [SAM 2 research paper](https://arxiv.org/abs/2408.00714).

### How can I use SAM 2 for real-time video segmentation?

SAM 2 can be utilized for real-time video segmentation by leveraging its promptable interface and real-time inference capabilities. Here's a basic example:

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

SAM 2 is trained on the SA-V dataset, one of the largest and most diverse video segmentation datasets available. The SA-V dataset includes:

- **51,000+ Videos**: Captured across 47 countries, providing a wide range of real-world scenarios.
- **600,000+ Mask Annotations**: Detailed spatio-temporal mask annotations, referred to as "masklets," covering whole objects and parts.
- **Dataset Scale**: Features 4.5 times more videos and 53 times more annotations than previous largest datasets, offering unprecedented diversity and complexity.

This extensive dataset allows SAM 2 to achieve superior performance across major video segmentation benchmarks and enhances its zero-shot generalization capabilities. For more information, see the [SA-V Dataset](#sa-v-dataset) section.

### How does SAM 2 handle occlusions and object reappearances in video segmentation?

SAM 2 includes a sophisticated memory mechanism to manage temporal dependencies and occlusions in video data. The memory mechanism consists of:

- **Memory Encoder and Memory Bank**: Stores features from past frames.
- **Memory Attention Module**: Utilizes stored information to maintain consistent object tracking over time.
- **Occlusion Head**: Specifically handles scenarios where objects are not visible, predicting the likelihood of an object being occluded.

This mechanism ensures continuity even when objects are temporarily obscured or exit and re-enter the scene. For more details, refer to the [Memory Mechanism and Occlusion Handling](#memory-mechanism-and-occlusion-handling) section.

### How does SAM 2 compare to other segmentation models like YOLO11?

SAM 2 models, such as Meta's SAM2-t and SAM2-b, offer powerful zero-shot segmentation capabilities but are significantly larger and slower compared to YOLO11 models. For instance, YOLO11n-seg is approximately **13 times smaller** and over **860 times faster** than SAM2-b. While SAM 2 excels in versatile, prompt-based, and zero-shot segmentation scenarios, YOLO11 is optimized for speed, efficiency, and real-time applications, making it better suited for deployment in resource-constrained environments.
