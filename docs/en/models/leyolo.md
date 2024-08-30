---
comments: true
description: Discover LeYOLO, an object detection model designed for computational efficiency with innovations like efficient backbone scaling, FPAN, and DNiN detection head.
keywords: LeYOLO, real-time object detection, Ultralytics, AI, computer vision, model training, object detector
---

# LeYOLO: Lightweight and Efficient YOLO

## Overview

LeYOLO marks a significant step forward in the realm of efficient object detection, particularly for embedded and mobile-oriented AI applications. Recognizing the increasing demand for computational efficiency, LeYOLO addresses the challenges posed by newer models that prioritize speed but often overlook efficient computation (FLOP). Through innovative design choices and targeted optimizations, LeYOLO redefines the landscape of YOLO-based models, offering a solution that balances accuracy and computational efficiency.

![LeYOLO](https://github.com/user-attachments/assets/52b2d9b8-d74f-400a-925c-98881bdc8cba)

## Key Features

- **Efficient Backbone Scaling:** LeYOLO introduces an efficient backbone scaling technique inspired by inverted bottlenecks and insights from the Information Bottleneck principle. This enhancement ensures that the network remains computationally efficient while maintaining the necessary depth and complexity for high-performance object detection.

- **Fast Pyramidal Architecture Network (FPAN):** The FPAN is a novel architecture designed to enable fast multiscale feature sharing. This feature significantly reduces the computational resources required, allowing LeYOLO to perform complex detection tasks swiftly and with minimal computational overhead.

- **Decoupled Network-in-Network (DNiN) Detection Head:** LeYOLO features a decoupled detection head, the DNiN, engineered for rapid yet lightweight computations. This component excels in delivering high-speed classification and regression tasks, making LeYOLO highly effective for real-time applications.

- **Unprecedented FLOP-to-Accuracy Ratio:** LeYOLO introduces a new scaling paradigm for object detection models, optimizing the FLOP-to-accuracy ratio across various configurations. For instance, LeYOLO-Small achieves a competitive mAP score of 38.2% on the COCO validation set with just 4.5 FLOP(G), a 42% reduction in computational load compared to the latest YOLOv9-Tiny model while maintaining similar accuracy levels. This model family offers scalability from ultra-low neural network configurations (< 1 GFLOP) to more demanding setups (> 4 GFLOPs), achieving 25.2, 31.3, 35.2, 38.2, 39.3, and 41 mAP for 0.66, 1.47, 2.53, 4.51, 5.8, and 8.4 FLOP(G) respectively.

![LeYOLO Comparison](https://github.com/user-attachments/assets/f7faf438-4488-4cd9-8bde-f0d4aae7ea4f)

## Performance Metrics

| Model        | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs (G) | Latency (ms) |
| ------------ | --------------------- | -------------------- | ------------------ | --------- | ------------ |
| LeYOLONano   | 640                   | 34.3                 | 1.1                | 2.6       | 2.9          |
| LeYOLOSmall  | 640                   | 38.2                 | 1.9                | 4.5       | 3.8          |
| LeYOLOMedium | 640                   | 39.3                 | 2.4                | 5.8       | 4.9          |
| LeYOLOLarge  | 640                   | 39.2                 | 2.4                | 5.8       | 4.9          |

Latency measured on RTX 3060 GPU.

## Usage Examples

As of the time of writing, Ultralytics does not officially support LeYOLO models. However, LeYOLO is built on the Ultralytics Python package and any users interested in using LeYOLO will need to refer directly to the LeYOLO GitHub repository for installation and usage instructions.

Here is a brief overview of the typical steps you might take to use LeYOLO:

1. Visit the [LeYOLO GitHub repository](https://github.com/LilianHollard/LeYOLO).

2. Follow the instructions provided in the README file for installation. This typically involves cloning the repository, installing necessary dependencies, and setting up any necessary environment variables.

3. Once installation is complete, you can train and use the model as per the usage instructions provided in the repository. This usually involves preparing your dataset, configuring the model parameters, training the model, and then using the trained model to perform object detection.

Please note that the specific steps may vary depending on your specific use case and the current state of the LeYOLO repository. Therefore, it is strongly recommended to refer directly to the instructions provided in the LeYOLO GitHub repository.

We regret any inconvenience this may cause and will strive to update this document with usage examples for Ultralytics once support for LeYOLO is implemented.

## Citations and Acknowledgements

We would like to acknowledge the LeYOLO authors for their significant contributions in the field of real-time object detection:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{hollard2024leyolonewscalableefficient,
            title={LeYOLO, New Scalable and Efficient CNN Architecture for Object Detection},
            author={Lilian Hollard and Lucas Mohimont and Nathalie Gaveau and Luiz-Angelo Steffenel},
            year={2024},
            eprint={2406.14239},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/2406.14239},
        }
        ```

The original LeYOLO paper can be found on [arXiv](https://arxiv.org/abs/2406.14239). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/LilianHollard/LeYOLO). We appreciate their efforts in advancing the field and making their work accessible to the broader community.
