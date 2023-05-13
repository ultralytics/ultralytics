---
comments: true
description: Learn about the Segment Anything Model (SAM) and how it provides promptable image segmentation through an advanced architecture and the SA-1B dataset.
---

# Segment Anything Model (SAM)

## Overview

The Segment Anything Model (SAM) is a groundbreaking image segmentation model that enables promptable segmentation with real-time performance. It forms the foundation for the Segment Anything project, which introduces a new task, model, and dataset for image segmentation. SAM is designed to be promptable, allowing it to transfer zero-shot to new image distributions and tasks. The model is trained on the [SA-1B dataset](https://ai.facebook.com/datasets/segment-anything/), which contains over 1 billion masks on 11 million licensed and privacy-respecting images. SAM has demonstrated impressive zero-shot performance, often surpassing prior fully supervised results.

![Dataset sample image](https://production-media.paperswithcode.com/datasets/540cfe0d-5fe7-43a4-9e1d-4c5813ddeb3e.png)

## Key Features

- **Promptable Segmentation Task:** SAM is designed for a promptable segmentation task, enabling it to return a valid segmentation mask given any segmentation prompt, such as spatial or text information identifying an object.
- **Advanced Architecture:** SAM utilizes a powerful image encoder, a prompt encoder, and a lightweight mask decoder. This architecture enables flexible prompting, real-time mask computation, and ambiguity awareness in segmentation.
- **SA-1B Dataset:** The Segment Anything project introduces the SA-1B dataset, which contains over 1 billion masks on 11 million images. This dataset is the largest segmentation dataset to date, providing SAM with a diverse and large-scale source of data for training.
- **Zero-Shot Performance:** SAM demonstrates remarkable zero-shot performance across a range of segmentation tasks, allowing it to be used out-of-the-box with prompt engineering for various applications.

For more information about the Segment Anything Model and the SA-1B dataset, please refer to the [Segment Anything website](https://segment-anything.com) and the research paper [Segment Anything](https://arxiv.org/abs/2304.02643).

## Usage

SAM can be used for a variety of downstream tasks involving object and image distributions beyond its training data. Examples include edge detection, object proposal generation, instance segmentation, and preliminary text-to-mask prediction. By employing prompt engineering, SAM can adapt to new tasks and data distributions in a zero-shot manner, making it a versatile and powerful tool for image segmentation tasks.

```python
from ultralytics.vit import SAM

model = SAM('sam_b.pt')
model.info()  # display model information
model.predict('path/to/image.jpg')  # predict
```

## Supported Tasks

| Model Type | Pre-trained Weights | Tasks Supported       |
|------------|---------------------|-----------------------|
| sam base   | `sam_b.pt`          | Instance Segmentation |
| sam large  | `sam_l.pt`          | Instance Segmentation |

## Supported Modes

| Mode       | Supported          |
|------------|--------------------|
| Inference  | :heavy_check_mark: |
| Validation | :x:                |
| Training   | :x:                |

# Citations and Acknowledgements

If you use SAM in your research or development work, please cite the following paper:

```bibtex
@misc{kirillov2023segment,
      title={Segment Anything}, 
      author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
      year={2023},
      eprint={2304.02643},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

We would like to acknowledge Meta AI for creating and maintaining this valuable resource for the computer vision community.