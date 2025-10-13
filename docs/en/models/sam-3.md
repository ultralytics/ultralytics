---
comments: true
description: Discover SAM 3, Meta's next evolution of the Segment Anything Model, introducing Promptable Concept Segmentation with text and image exemplar prompts for detecting all instances of visual concepts across images and videos.
keywords: SAM 3, Segment Anything, SAM3, video segmentation, image segmentation, concept segmentation, promptable AI, SA-Co dataset, Meta, Ultralytics, computer vision, AI, machine learning, open vocabulary
---

!!! note "Coming Soon ⚠️"

    🚧 SAM 3 models have not yet been publicly released by Meta. The information below is based on the research paper submitted to ICLR 2026.  
    Model downloads and final benchmarks will be available following Meta's official release.

# SAM 3: Segment Anything with Concepts

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/inference-with-meta-sam3-using-ultralytics.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Run SAM 3 in Colab"></a>

**SAM 3** (Segment Anything Model 3) represents Meta's next-generation foundation model for **Promptable Concept Segmentation (PCS)**. Building upon [SAM 2](sam-2.md), SAM 3 introduces a fundamentally new capability: detecting, segmenting, and tracking **all instances** of a visual concept specified by text prompts, image exemplars, or both. Unlike previous SAM versions that segment single objects per prompt, SAM 3 can find and segment every occurrence of a concept appearing anywhere in images or videos.

![SAM 3 Concept Segmentation](https://github.com/ultralytics/assets/releases/download/v0.0.0/sam3-concept.avif)

## Overview

SAM 3 achieves a **2× performance gain** over existing systems in Promptable Concept Segmentation while maintaining and improving SAM 2's capabilities for interactive visual segmentation. The model excels at open-vocabulary segmentation, allowing users to specify concepts using simple noun phrases (e.g., "yellow school bus", "striped cat") or by providing example images of the target object.

### What is Promptable Concept Segmentation (PCS)?

The PCS task takes a **concept prompt** as input and returns segmentation masks with unique identities for **all matching object instances**. Concept prompts can be:

- **Text**: Simple noun phrases like "red apple" or "person wearing a hat"
- **Image exemplars**: Bounding boxes around example objects (positive or negative)
- **Combined**: Both text and image exemplars together for precise control

This differs from traditional visual prompts (points, boxes, masks) which segment only a single specific object instance.

### Key Performance Metrics

| Metric                           | SAM 3 Achievement                                           |
| -------------------------------- | ----------------------------------------------------------- |
| **LVIS Zero-Shot Mask AP**       | **47.0** (vs previous best 38.5, +22% improvement)          |
| **SA-Co Benchmark**              | **2× better** than existing systems                         |
| **Inference Speed (H200 GPU)**   | **30 ms** per image with 100+ detected objects              |
| **Video Performance**            | Near real-time for ~5 concurrent objects                    |
| **MOSEv2 VOS Benchmark**         | **60.1 J&F** (+25.5% over SAM 2.1, +17% over prior SOTA)   |
| **Interactive Refinement**       | **+18.6 CGF1** improvement after 3 exemplar prompts         |
| **Human Performance Gap**        | Achieves **88%** of estimated lower bound on SA-Co/Gold     |

## Architecture

SAM 3 consists of a **detector** and **tracker** that share a Perception Encoder (PE) vision backbone. This decoupled design avoids task conflicts while enabling both image-level detection and video-level tracking.

### Core Components

- **Detector**: DETR-based architecture for image-level concept detection
  - Text encoder for noun phrase prompts
  - Exemplar encoder for image-based prompts
  - Fusion encoder to condition image features on prompts
  - Novel **presence head** that decouples recognition ("what") from localization ("where")
  - Mask head for generating instance segmentation masks

- **Tracker**: Memory-based video segmentation inherited from SAM 2
  - Prompt encoder, mask decoder, memory encoder
  - Memory bank for storing object appearance across frames
  - Temporal disambiguation strategies for crowded scenes

- **Presence Token**: A learned global token that predicts whether the target concept is present in the image/frame, significantly boosting detection accuracy by separating recognition from localization tasks.

![SAM 3 Architecture](https://github.com/ultralytics/assets/releases/download/v0.0.0/sam3-architecture-diagram.avif)

### Key Innovations

1. **Decoupled Recognition and Localization**: The presence head predicts concept presence globally, while proposal queries focus only on localization, avoiding conflicting objectives.

2. **Unified Concept and Visual Prompts**: Supports both PCS (concept prompts) and PVS (visual prompts like SAM 2's clicks/boxes) in a single model.

3. **Interactive Exemplar Refinement**: Users can add positive or negative image exemplars to iteratively refine results, with the model generalizing to similar objects rather than just correcting individual instances.

4. **Temporal Disambiguation**: Uses masklet detection scores and periodic re-prompting to handle occlusions, crowded scenes, and tracking failures in video.

## SA-Co Dataset

SAM 3 is trained on **Segment Anything with Concepts (SA-Co)**, Meta's largest and most diverse segmentation dataset to date.

### Training Data

| Dataset Component | Description                                                                 | Scale                                  |
| ----------------- | --------------------------------------------------------------------------- | -------------------------------------- |
| **SA-Co/HQ**      | High-quality human-annotated image data from 4-phase data engine           | 5.2M images, 4M unique noun phrases    |
| **SA-Co/SYN**     | Synthetic dataset labeled by AI without human involvement                   | 38M noun phrases, 1.4B masks           |
| **SA-Co/EXT**     | 15 external datasets enriched with hard negatives                           | Varies by source                       |
| **SA-Co/VIDEO**   | Video annotations with temporal tracking                                    | 52.5K videos, 24.8K unique noun phrases |

### Benchmark Data

The **SA-Co evaluation benchmark** contains **214K unique phrases** across **126K images and videos**, providing over **50× more concepts** than existing benchmarks. It includes:

- **SA-Co/Gold**: 7 domains, triple-annotated for measuring human performance bounds
- **SA-Co/Silver**: 10 domains, single human annotation
- **SA-Co/Bronze** and **SA-Co/Bio**: 9 existing datasets adapted for concept segmentation
- **SA-Co/VEval**: Video benchmark with 3 domains (SA-V, YT-Temporal-1B, SmartGlasses)

### Data Engine Innovations

SAM 3's scalable human- and model-in-the-loop data engine achieves **2× annotation throughput** through:

1. **AI Annotators**: Llama-based models propose diverse noun phrases including hard negatives
2. **AI Verifiers**: Fine-tuned MLLMs verify mask quality and exhaustivity at near-human performance
3. **Active Mining**: Focuses human effort on challenging failure cases where AI struggles
4. **Ontology-Driven**: Leverages 22.4M node SA-Co ontology based on Wikidata for concept coverage

## Installation

SAM 3 will be supported natively in the Ultralytics package upon release:

```bash
pip install ultralytics
```

Models will download automatically when first used.

## How to Use SAM 3: Versatility in Concept Segmentation

!!! warning "Preview API - Subject to Change"

    The code examples below demonstrate **intended usage patterns** based on the research paper. The actual API will be available after:
    
    1. Meta open-sources SAM 3 model weights
    2. Ultralytics integrates SAM 3 into the package
    
    Syntax and parameters may differ in the final implementation. These examples serve as a preview of expected functionality.

### Supported Tasks and Models

SAM 3 supports both Promptable Concept Segmentation (PCS) and Promptable Visual Segmentation (PVS) tasks:

| Task Type                       | Prompt Types                             | Output                                        |
| ------------------------------- | ---------------------------------------- | --------------------------------------------- |
| **Concept Segmentation (PCS)**  | Text (noun phrases), image exemplars     | All instances matching the concept            |
| **Visual Segmentation (PVS)**   | Points, boxes, masks                     | Single object instance (SAM 2 style)          |
| **Interactive Refinement**      | Add/remove exemplars or clicks iteratively | Refined segmentation with improved accuracy |

### Concept Segmentation Examples

#### Segment with Text Prompts

!!! example "Text-based Concept Segmentation"

    Find and segment all instances of a concept using a text description.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load SAM 3 model
        model = SAM("sam3.pt")

        # Segment all instances of a concept
        results = model("path/to/image.jpg", prompt="yellow school bus")

        # Works with descriptive phrases
        results = model("path/to/image.jpg", prompt="person wearing a red hat")

        # Or simple object names
        results = model("path/to/image.jpg", prompt="striped cat")
        ```

    === "CLI"

        ```bash
        # Segment all matching concepts in an image
        yolo segment model=sam3.pt source=path/to/image.jpg prompt="yellow school bus"
        ```

    !!! warning "API Preview"
        
        This example shows intended usage. Actual implementation pending Meta's release and Ultralytics integration.

#### Segment with Image Exemplars

!!! example "Image Exemplar-based Segmentation"

    Use one or more example objects to find all similar instances.

    === "Python"

        ```python
        from ultralytics import SAM

        model = SAM("sam3.pt")

        # Provide a positive example box - finds all similar objects
        results = model("path/to/image.jpg", bboxes=[100, 150, 300, 400], labels=[1])

        # Add negative examples to exclude certain instances
        results = model(
            "path/to/image.jpg",
            bboxes=[[100, 150, 300, 400], [500, 200, 600, 350]],  # Two boxes
            labels=[1, 0]  # First is positive, second is negative
        )

        # Combine text and image exemplars for precision
        results = model(
            "path/to/image.jpg",
            prompt="dog",
            bboxes=[100, 150, 300, 400],
            labels=[1]
        )
        ```

    !!! warning "API Preview"
        
        This example shows intended usage. Actual implementation pending Meta's release and Ultralytics integration.

#### Interactive Refinement

!!! example "Iterative Refinement with Exemplars"

    Progressively improve results by adding exemplar prompts based on initial output.

    === "Python"

        ```python
        from ultralytics import SAM

        model = SAM("sam3.pt")

        # Initial segmentation with text
        results = model("path/to/image.jpg", prompt="car")

        # If some cars are missed, add a positive exemplar
        results = model(
            "path/to/image.jpg",
            prompt="car",
            bboxes=[missed_car_box],
            labels=[1]  # Positive example
        )

        # If false positives appear, add negative exemplars
        results = model(
            "path/to/image.jpg",
            prompt="car",
            bboxes=[false_positive_box],
            labels=[0]  # Negative example
        )
        ```

    !!! warning "API Preview"
        
        This example shows intended usage. Actual implementation pending Meta's release and Ultralytics integration.

### Video Concept Segmentation

!!! example "Track Concepts Across Video"

    Detect and track all instances of a concept throughout a video.

    === "Python"

        ```python
        from ultralytics.models.sam import SAM3VideoPredictor

        # Create video predictor
        predictor = SAM3VideoPredictor(model="sam3.pt", imgsz=1024, conf=0.25)

        # Track all instances of a concept
        results = predictor(source="video.mp4", prompt="person wearing blue shirt")

        # Combine text with exemplar for precision
        results = predictor(
            source="video.mp4",
            prompt="kangaroo",
            bboxes=[initial_box],  # Exemplar from first frame
            labels=[1]
        )
        ```

    !!! warning "API Preview"
        
        This example shows intended usage. Actual implementation pending Meta's release and Ultralytics integration.

### Visual Prompts (SAM 2 Compatibility)

SAM 3 maintains full backward compatibility with SAM 2's visual prompting:

!!! example "SAM 2 Style Visual Prompts"

    === "Python"

        ```python
        from ultralytics import SAM

        model = SAM("sam3.pt")

        # Single point prompt (SAM 2 style)
        results = model(points=[900, 370], labels=[1])

        # Multiple points
        results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

        # Box prompt
        results = model(bboxes=[100, 150, 300, 400])
        ```

    !!! warning "API Preview"
        
        This example shows intended usage. Actual implementation pending Meta's release and Ultralytics integration.

## Performance Benchmarks

### Image Segmentation

SAM 3 achieves state-of-the-art results across multiple benchmarks:

| Benchmark                     | Metric    | SAM 3   | Previous Best | Improvement |
| ----------------------------- | --------- | ------- | ------------- | ----------- |
| **LVIS (zero-shot)**          | Mask AP   | **47.0** | 38.5          | +22.1%      |
| **SA-Co/Gold**                | CGF1      | **65.0** | 34.3 (OWLv2)  | +89.5%      |
| **COCO (zero-shot)**          | Box AP    | **53.5** | 52.2 (T-Rex2) | +2.5%       |
| **ADE-847 (semantic seg)**    | mIoU      | **14.7** | 9.2 (APE-D)   | +59.8%      |
| **PascalConcept-59**          | mIoU      | **59.4** | 58.5 (APE-D)  | +1.5%       |
| **Cityscapes (semantic seg)** | mIoU      | **65.1** | 44.2 (APE-D)  | +47.3%      |

### Video Segmentation Performance

SAM 3 shows significant improvements over SAM 2 and prior state-of-the-art across video benchmarks:

| Benchmark         | Metric | SAM 3   | SAM 2.1 L | Improvement |
| ----------------- | ------ | ------- | --------- | ----------- |
| **MOSEv2**        | J&F    | **60.1** | 47.9      | +25.5%      |
| **DAVIS 2017**    | J&F    | **92.0** | 90.7      | +1.4%       |
| **LVOSv2**        | J&F    | **88.2** | 79.6      | +10.8%      |
| **SA-V**          | J&F    | **84.6** | 78.4      | +7.9%       |
| **YTVOS19**       | J&F    | **89.6** | 89.3      | +0.3%       |

### Few-Shot Adaptation

SAM 3 excels at adapting to new domains with minimal examples:

| Benchmark          | 0-shot AP | 10-shot AP | Previous Best (10-shot) |
| ------------------ | --------- | ---------- | ----------------------- |
| **ODinW13**        | 59.9      | **71.6**   | 67.9 (gDino1.5-Pro)     |
| **Roboflow100-VL** | 14.3      | **35.7**   | 33.7 (gDino-T)          |

### Interactive Refinement Effectiveness

SAM 3's concept-based prompting with exemplars converges much faster than visual prompting:

| Prompts Added | CGF1 Score | Gain vs Text-Only | Gain vs PVS Baseline |
| ------------- | ---------- | ----------------- | -------------------- |
| Text only     | 46.4       | baseline          | baseline             |
| +1 exemplar   | 57.6       | +11.2             | +6.7                 |
| +2 exemplars  | 62.2       | +15.8             | +9.7                 |
| +3 exemplars  | **65.0**   | **+18.6**         | **+11.2**            |
| +4 exemplars  | 65.7       | +19.3             | +11.5 (plateau)      |

Exemplar-based PCS converges faster than click-based PVS by generalizing to similar objects.

### Object Counting Accuracy

SAM 3 provides accurate counting by segmenting all instances:

| Benchmark        | Accuracy | MAE  | vs Best MLLM       |
| ---------------- | -------- | ---- | ------------------ |
| **CountBench**   | **95.6%** | 0.11 | 92.4% (Gemini 2.5) |
| **PixMo-Count**  | **87.3%** | 0.22 | 88.8% (Molmo-72B)  |

## SAM 3 vs SAM 2 vs YOLO Comparison

Here we compare SAM 3's capabilities with SAM 2 and YOLO11 models:

| Capability                     | SAM 3                                      | SAM 2                   | YOLO11n-seg         |
| ------------------------------ | ------------------------------------------ | ----------------------- | ------------------- |
| **Concept Segmentation**       | ✅ All instances from text/exemplars       | ❌ Not supported        | ❌ Not supported    |
| **Visual Segmentation**        | ✅ Single instance (SAM 2 compatible)      | ✅ Single instance      | ✅ All instances    |
| **Zero-shot Capability**       | ✅ Open vocabulary                         | ✅ Geometric prompts    | ❌ Closed set       |
| **Interactive Refinement**     | ✅ Exemplars + clicks                      | ✅ Clicks only          | ❌ Not supported    |
| **Video Tracking**             | ✅ Multi-object with identities            | ✅ Multi-object         | ✅ Multi-object     |
| **LVIS Mask AP (zero-shot)**   | **47.0**                                   | N/A                     | N/A                 |
| **MOSEv2 J&F**                 | **60.1**                                   | 47.9                    | N/A                 |
| **Inference Speed (H200)**     | **30 ms** (100+ objects)                   | ~23 ms (per object)     | **2-3 ms** (image)  |
| **Model Size**                 | Large (~400+ MB expected)                  | 162 MB (base)           | **5.9 MB**          |

**Key Takeaways**:

- **SAM 3**: Best for open-vocabulary concept segmentation, finding all instances of a concept with text or exemplar prompts
- **SAM 2**: Best for interactive single-object segmentation in images and videos with geometric prompts
- **YOLO11**: Best for real-time, high-speed segmentation in resource-constrained environments with known categories

## Evaluation Metrics

SAM 3 introduces new metrics designed for the PCS task:

### Classification-Gated F1 (CGF1)

The primary metric combining localization and classification:

**CGF1 = 100 × pmF1 × IL_MCC**

Where:
- **pmF1** (Positive Macro F1): Measures localization quality on positive examples
- **IL_MCC** (Image-Level Matthews Correlation Coefficient): Measures binary classification accuracy ("is the concept present?")

### Why These Metrics?

Traditional AP metrics don't account for calibration, making models difficult to use in practice. By evaluating only predictions above 0.5 confidence, SAM 3's metrics enforce good calibration and mimic real-world usage patterns.

## Key Ablations and Insights

### Impact of Presence Head

The presence head decouples recognition from localization, providing significant improvements:

| Configuration     | CGF1  | IL_MCC | pmF1  |
| ----------------- | ----- | ------ | ----- |
| Without presence  | 57.6  | 0.77   | 74.7  |
| **With presence** | **63.3** | **0.82** | **77.1** |

The presence head provides a **+5.7 CGF1 boost** (+9.9%), primarily improving recognition ability (IL_MCC +6.5%).

### Effect of Hard Negatives

| Hard Negatives/Image | CGF1  | IL_MCC | pmF1  |
| -------------------- | ----- | ------ | ----- |
| 0                    | 31.8  | 0.44   | 70.2  |
| 5                    | 44.8  | 0.62   | 71.9  |
| **30**               | **49.2** | **0.68** | **72.3** |

Hard negatives are crucial for open-vocabulary recognition, improving IL_MCC by **54.5%** (0.44 → 0.68).

### Training Data Scaling

| Data Sources         | CGF1  | IL_MCC | pmF1  |
| -------------------- | ----- | ------ | ----- |
| External only        | 30.9  | 0.46   | 66.3  |
| External + Synthetic | 39.7  | 0.57   | 70.6  |
| External + HQ        | 51.8  | 0.71   | 73.2  |
| **All three**        | **54.3** | **0.74** | **73.5** |

High-quality human annotations (SA-Co/HQ) provide **+14.6 CGF1** over synthetic data, and **+23.4 CGF1** over external data alone.

## Applications

SAM 3's concept segmentation capability enables new use cases:

- **Content Moderation**: Find all instances of specific content types across large media libraries
- **E-commerce**: Segment all products of a certain type in catalog images
- **Medical Imaging**: Identify all occurrences of specific tissue types or abnormalities
- **Autonomous Systems**: Track all instances of traffic signs, pedestrians, or vehicles by category
- **Video Analytics**: Count and track all people wearing specific clothing or performing actions
- **Dataset Annotation**: Rapidly annotate all instances of rare object categories
- **Scientific Research**: Quantify and analyze all specimens matching specific criteria

## SAM 3 Agent: Extended Language Reasoning

SAM 3 can be combined with Multimodal Large Language Models (MLLMs) to handle complex queries requiring reasoning:

### Performance on Reasoning Tasks

| Benchmark                    | Metric | SAM 3 Agent (Gemini 2.5 Pro) | Previous Best |
| ---------------------------- | ------ | ---------------------------- | ------------- |
| **ReasonSeg (validation)**   | gIoU   | **76.0**                     | 65.0 (SoTA)   |
| **ReasonSeg (test)**         | gIoU   | **73.8**                     | 61.3 (SoTA)   |
| **OmniLabel (validation)**   | AP     | **46.7**                     | 36.5 (REAL)   |
| **RefCOCO+**                 | Acc    | **91.2**                     | 89.3 (LISA)   |

### Example Complex Queries

SAM 3 Agent can handle queries requiring reasoning:

- "People sitting down but not holding a gift box in their hands"
- "The dog closest to the camera that is not wearing a collar"
- "Red objects larger than the person's hand"

The MLLM proposes simple noun phrase queries to SAM 3, analyzes returned masks, and iterates until satisfied.

## Limitations

While SAM 3 represents a major advancement, it has certain limitations:

- **Phrase Complexity**: Best suited for simple noun phrases; long referring expressions or complex reasoning may require MLLM integration
- **Ambiguity Handling**: Some concepts remain inherently ambiguous (e.g., "small window", "cozy room")
- **Computational Requirements**: Larger and slower than specialized detection models like YOLO
- **Vocabulary Scope**: Focused on atomic visual concepts; compositional reasoning is limited without MLLM assistance
- **Rare Concepts**: Performance degrades on extremely rare or fine-grained concepts not well-represented in training data

## Citation

!!! quote ""

    === "BibTeX"
    
        ```bibtex
        @inproceedings{sam3_2025,
          title     = {SAM 3: Segment Anything with Concepts},
          author    = {Anonymous authors},
          booktitle = {Under review as a conference paper at ICLR 2026},
          year      = {2025},
          note      = {Paper under double-blind review}
        }
        ```

---

## FAQ

### When will SAM 3 be released?

SAM 3 is currently under review at ICLR 2026 (conference in 2026, review in 2025). Official models, weights, and benchmarks will be released publicly following the review process, likely in **2025-2026**. Ultralytics will provide immediate support for SAM 3 integration upon Meta's release.

### Will SAM 3 be integrated into Ultralytics?

Yes. SAM 3 will be fully supported in the Ultralytics Python package upon release, including:

- **Concept Segmentation**: Text and image exemplar prompts for finding all matching instances
- **Visual Segmentation**: Full SAM 2 compatibility for single-object segmentation
- **Video Tracking**: Multi-object tracking with concept-based initialization
- **Interactive Refinement**: Both exemplar-based and click-based refinement
- **Auto-Annotation**: Dataset creation with concept-based prompting
- **Export**: Model conversion to ONNX, TensorRT, and other formats

!!! warning "Implementation Timeline"
    
    Code examples in this documentation are preview versions showing intended usage patterns. Actual implementation will be available after Meta releases SAM 3 weights and Ultralytics completes integration.

### What is Promptable Concept Segmentation (PCS)?

PCS is a new task introduced in SAM 3 that segments **all instances** of a visual concept in an image or video. Unlike traditional segmentation that targets a specific object instance, PCS finds every occurrence of a category. For example:

- **Text prompt**: "yellow school bus" → segments all yellow school buses in the scene
- **Image exemplar**: Box around one dog → segments all dogs in the image
- **Combined**: "striped cat" + exemplar box → segments all striped cats matching the example

This differs from SAM 2's visual segmentation which segments only the single object you click on or box.

### How does SAM 3 differ from SAM 2?

SAM 3 expands on SAM 2 with several major capabilities:

| Feature                     | SAM 2                              | SAM 3                                        |
| --------------------------- | ---------------------------------- | -------------------------------------------- |
| **Task**                    | Single object per prompt           | All instances of a concept                   |
| **Prompt Types**            | Points, boxes, masks               | + Text phrases, image exemplars              |
| **Detection Capability**    | Requires external detector         | Built-in open-vocabulary detector            |
| **Recognition**             | Geometry-based only                | Text and visual recognition                  |
| **Architecture**            | Tracker only                       | Detector + Tracker with presence head        |
| **Zero-Shot Performance**   | N/A (requires visual prompts)      | 47.0 AP on LVIS, 2× better on SA-Co          |
| **Interactive Refinement**  | Clicks only                        | Clicks + exemplar generalization             |

SAM 3 maintains full backward compatibility with SAM 2's visual prompting while adding concept-based capabilities. On video segmentation, SAM 3 achieves **60.1 J&F** on MOSEv2 compared to SAM 2.1's **47.9 J&F** (+25.5% improvement).

### How can I use SAM 3 for video segmentation?

Once released, SAM 3 provides enhanced video capabilities (preview syntax):

```python
from ultralytics.models.sam import SAM3VideoPredictor

# Create predictor
predictor = SAM3VideoPredictor(model="sam3.pt")

# Track all instances of a concept across video
results = predictor(source="video.mp4", prompt="person wearing red jacket")

# Or use exemplar from first frame
results = predictor(source="video.mp4", bboxes=[100, 150, 300, 400], labels=[1])
```

!!! warning "API Preview"
    
    This code demonstrates intended usage patterns. Actual syntax may differ when SAM 3 is officially released and integrated into Ultralytics.

SAM 3 achieves **60.1 J&F** on MOSEv2 (+12.2 points over SAM 2.1) and runs at near real-time speeds for ~5 concurrent objects on H200 GPU (30 ms per frame with 100+ detections).

### What datasets are used to train SAM 3?

SAM 3 is trained on the **Segment Anything with Concepts (SA-Co)** dataset:

**Training Data**:
- **5.2M images** with **4M unique noun phrases** (SA-Co/HQ) - high-quality human annotations
- **52.5K videos** with **24.8K unique noun phrases** (SA-Co/VIDEO)
- **1.4B synthetic masks** across **38M noun phrases** (SA-Co/SYN)
- **15 external datasets** enriched with hard negatives (SA-Co/EXT)

**Benchmark Data**:
- **214K unique concepts** across **126K images/videos**
- **50× more concepts** than existing benchmarks (e.g., LVIS has ~4K concepts)
- Triple annotation on SA-Co/Gold for measuring human performance bounds

This massive scale and diversity enables SAM 3's superior zero-shot generalization across open-vocabulary concepts.

### How does SAM 3 compare to YOLO11 for segmentation?

SAM 3 and YOLO11 serve different use cases:

**SAM 3 Advantages**:
- **Open-vocabulary**: Segments any concept via text prompts without training
- **Zero-shot**: Works on new categories immediately
- **Interactive**: Exemplar-based refinement generalizes to similar objects
- **Concept-based**: Automatically finds all instances of a category
- **Accuracy**: 47.0 AP on LVIS zero-shot instance segmentation

**YOLO11 Advantages**:
- **Speed**: 10-15× faster inference (2-3ms vs 30ms per image)
- **Efficiency**: 70× smaller models (5.9MB vs ~400MB expected)
- **Resource-friendly**: Runs on edge devices and mobile
- **Real-time**: Optimized for production deployments

**Recommendation**: 
- Use **SAM 3** for flexible, open-vocabulary segmentation where you need to find all instances of concepts described by text or examples
- Use **YOLO11** for high-speed, production deployments where categories are known in advance
- Use **SAM 2** for interactive single-object segmentation with geometric prompts

### Can SAM 3 handle complex language queries?

SAM 3 is designed for simple noun phrases (e.g., "red apple", "person wearing hat"). For complex queries requiring reasoning, combine SAM 3 with an MLLM as **SAM 3 Agent**:

**Simple queries (native SAM 3)**:
- "yellow school bus"
- "striped cat"
- "person wearing red hat"

**Complex queries (SAM 3 Agent with MLLM)**:
- "People sitting down but not holding a gift box"
- "The dog closest to the camera without a collar"
- "Red objects larger than the person's hand"

SAM 3 Agent achieves **76.0 gIoU** on ReasonSeg validation (vs 65.0 previous best, +16.9% improvement) by combining SAM 3's segmentation with MLLM reasoning capabilities.

### How accurate is SAM 3 compared to human performance?

On the SA-Co/Gold benchmark with triple human annotation:

- **Human lower bound**: 74.2 CGF1 (most conservative annotator)
- **SAM 3 performance**: 65.0 CGF1
- **Achievement**: **88%** of estimated human lower bound
- **Human upper bound**: 81.4 CGF1 (most liberal annotator)

SAM 3 achieves strong performance approaching human-level accuracy on open-vocabulary concept segmentation, with the gap primarily on ambiguous or subjective concepts (e.g., "small window", "cozy room").
