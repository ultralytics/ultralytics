---
comments: true
description: Explore FastSAM, a CNN-based solution for real-time object segmentation in images. Enhanced user interaction, computational efficiency and adaptable across vision tasks.
keywords: FastSAM, machine learning, CNN-based solution, object segmentation, real-time solution, Ultralytics, vision tasks, image processing, industrial applications, user interaction
---

# Fast Segment Anything Model (FastSAM)

The Fast Segment Anything Model (FastSAM) is a novel, real-time CNN-based solution for the Segment Anything task. This task is designed to segment any object within an image based on various possible user interaction prompts. FastSAM significantly reduces computational demands while maintaining competitive performance, making it a practical choice for a variety of vision tasks.

![Fast Segment Anything Model (FastSAM) architecture overview](https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg)

## Overview

FastSAM is designed to address the limitations of the [Segment Anything Model (SAM)](sam.md), a heavy Transformer model with substantial computational resource requirements. The FastSAM decouples the segment anything task into two sequential stages: all-instance segmentation and prompt-guided selection. The first stage uses [YOLOv8-seg](../tasks/segment.md) to produce the segmentation masks of all instances in the image. In the second stage, it outputs the region-of-interest corresponding to the prompt.

## Key Features

1. **Real-time Solution:** By leveraging the computational efficiency of CNNs, FastSAM provides a real-time solution for the segment anything task, making it valuable for industrial applications that require quick results.

2. **Efficiency and Performance:** FastSAM offers a significant reduction in computational and resource demands without compromising on performance quality. It achieves comparable performance to SAM but with drastically reduced computational resources, enabling real-time application.

3. **Prompt-guided Segmentation:** FastSAM can segment any object within an image guided by various possible user interaction prompts, providing flexibility and adaptability in different scenarios.

4. **Based on YOLOv8-seg:** FastSAM is based on [YOLOv8-seg](../tasks/segment.md), an object detector equipped with an instance segmentation branch. This allows it to effectively produce the segmentation masks of all instances in an image.

5. **Competitive Results on Benchmarks:** On the object proposal task on MS COCO, FastSAM achieves high scores at a significantly faster speed than [SAM](sam.md) on a single NVIDIA RTX 3090, demonstrating its efficiency and capability.

6. **Practical Applications:** The proposed approach provides a new, practical solution for a large number of vision tasks at a really high speed, tens or hundreds of times faster than current methods.

7. **Model Compression Feasibility:** FastSAM demonstrates the feasibility of a path that can significantly reduce the computational effort by introducing an artificial prior to the structure, thus opening new possibilities for large model architecture for general vision tasks.

## Usage

### Python API

The FastSAM models are easy to integrate into your Python applications. Ultralytics provides a user-friendly Python API to streamline the process.

#### Predict Usage

To perform object detection on an image, use the `predict` method as shown below:

!!! example ""

    === "Python"
        ```python
        from ultralytics import FastSAM
        from ultralytics.models.fastsam import FastSAMPrompt

        # Define an inference source
        source = 'path/to/bus.jpg'

        # Create a FastSAM model
        model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

        # Run inference on an image
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
      
        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # Everything prompt
        ann = prompt_process.everything_prompt()

        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

        # Text prompt
        ann = prompt_process.text_prompt(text='a photo of a dog')

        # Point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
        prompt_process.plot(annotations=ann, output='./')
        ```
      
    === "CLI"
        ```bash
        # Load a FastSAM model and segment everything with it
        yolo segment predict model=FastSAM-s.pt source=path/to/bus.jpg imgsz=640
        ```

This snippet demonstrates the simplicity of loading a pre-trained model and running a prediction on an image.

#### Val Usage

Validation of the model on a dataset can be done as follows:

!!! example ""

    === "Python"
        ```python
        from ultralytics import FastSAM

        # Create a FastSAM model
        model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

        # Validate the model
        results = model.val(data='coco8-seg.yaml')
        ```

    === "CLI"
        ```bash
        # Load a FastSAM model and validate it on the COCO8 example dataset at image size 640
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

Please note that FastSAM only supports detection and segmentation of a single class of object. This means it will recognize and segment all objects as the same class. Therefore, when preparing the dataset, you need to convert all object category IDs to 0.

### FastSAM official Usage

FastSAM is also available directly from the [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) repository. Here is a brief overview of the typical steps you might take to use FastSAM:

#### Installation

1. Clone the FastSAM repository:
   ```shell
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. Create and activate a Conda environment with Python 3.9:
   ```shell
   conda create -n FastSAM python=3.9
   conda activate FastSAM
   ```

3. Navigate to the cloned repository and install the required packages:
   ```shell
   cd FastSAM
   pip install -r requirements.txt
   ```

4. Install the CLIP model:
   ```shell
   pip install git+https://github.com/openai/CLIP.git
   ```

#### Example Usage

1. Download a [model checkpoint](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing).

2. Use FastSAM for inference. Example commands:

    - Segment everything in an image:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
      ```

    - Segment specific objects using text prompt:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "the yellow dog"
      ```

    - Segment objects within a bounding box (provide box coordinates in xywh format):
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
      ```

    - Segment objects near specific points:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
      ```

Additionally, you can try FastSAM through a [Colab demo](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing) or on the [HuggingFace web demo](https://huggingface.co/spaces/An-619/FastSAM) for a visual experience.

## Citations and Acknowledgements

We would like to acknowledge the FastSAM authors for their significant contributions in the field of real-time instance segmentation:

!!! note ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

The original FastSAM paper can be found on [arXiv](https://arxiv.org/abs/2306.12156). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM). We appreciate their efforts in advancing the field and making their work accessible to the broader community.
