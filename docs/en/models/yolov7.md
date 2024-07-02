---
comments: true
description: Discover YOLOv7, the breakthrough real-time object detector with top speed and accuracy. Learn about key features, usage, and performance metrics.
keywords: YOLOv7, real-time object detection, Ultralytics, AI, computer vision, model training, object detector
---

# YOLOv7: Trainable Bag-of-Freebies

YOLOv7 is a state-of-the-art real-time object detector that surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS. It has the highest accuracy (56.8% AP) among all known real-time object detectors with 30 FPS or higher on GPU V100. Moreover, YOLOv7 outperforms other object detectors such as YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5, and many others in speed and accuracy. The model is trained on the MS COCO dataset from scratch without using any other datasets or pre-trained weights. Source code for YOLOv7 is available on GitHub.

![YOLOv7 comparison with SOTA object detectors](https://github.com/ultralytics/ultralytics/assets/26833433/f7605c4e-8607-428b-aca4-0c2adfceb1a2)

## Comparison of SOTA object detectors

From the results in the YOLO comparison table we know that the proposed method has the best speed-accuracy trade-off comprehensively. If we compare YOLOv7-tiny-SiLU with YOLOv5-N (r6.1), our method is 127 fps faster and 10.7% more accurate on AP. In addition, YOLOv7 has 51.4% AP at frame rate of 161 fps, while PPYOLOE-L with the same AP has only 78 fps frame rate. In terms of parameter usage, YOLOv7 is 41% less than PPYOLOE-L. If we compare YOLOv7-X with 114 fps inference speed to YOLOv5-L (r6.1) with 99 fps inference speed, YOLOv7-X can improve AP by 3.9%. If YOLOv7-X is compared with YOLOv5-X (r6.1) of similar scale, the inference speed of YOLOv7-X is 31 fps faster. In addition, in terms the amount of parameters and computation, YOLOv7-X reduces 22% of parameters and 8% of computation compared to YOLOv5-X (r6.1), but improves AP by 2.2% ([Source](https://arxiv.org/pdf/2207.02696.pdf)).

| Model                 | Params<br><sup>(M) | FLOPs<br><sup>(G) | Size<br><sup>(pixels) | FPS     | AP<sup>test / val<br>50-95 | AP<sup>test<br>50 | AP<sup>test<br>75 | AP<sup>test<br>S | AP<sup>test<br>M | AP<sup>test<br>L |
| --------------------- | ------------------ | ----------------- | --------------------- | ------- | -------------------------- | ----------------- | ----------------- | ---------------- | ---------------- | ---------------- |
| [YOLOX-S][1]          | **9.0M**           | **26.8G**         | 640                   | **102** | 40.5% / 40.5%              | -                 | -                 | -                | -                | -                |
| [YOLOX-M][1]          | 25.3M              | 73.8G             | 640                   | 81      | 47.2% / 46.9%              | -                 | -                 | -                | -                | -                |
| [YOLOX-L][1]          | 54.2M              | 155.6G            | 640                   | 69      | 50.1% / 49.7%              | -                 | -                 | -                | -                | -                |
| [YOLOX-X][1]          | 99.1M              | 281.9G            | 640                   | 58      | **51.5% / 51.1%**          | -                 | -                 | -                | -                | -                |
|                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
| [PPYOLOE-S][2]        | **7.9M**           | **17.4G**         | 640                   | **208** | 43.1% / 42.7%              | 60.5%             | 46.6%             | 23.2%            | 46.4%            | 56.9%            |
| [PPYOLOE-M][2]        | 23.4M              | 49.9G             | 640                   | 123     | 48.9% / 48.6%              | 66.5%             | 53.0%             | 28.6%            | 52.9%            | 63.8%            |
| [PPYOLOE-L][2]        | 52.2M              | 110.1G            | 640                   | 78      | 51.4% / 50.9%              | 68.9%             | 55.6%             | 31.4%            | 55.3%            | 66.1%            |
| [PPYOLOE-X][2]        | 98.4M              | 206.6G            | 640                   | 45      | **52.2% / 51.9%**          | **69.9%**         | **56.5%**         | **33.3%**        | **56.3%**        | **66.4%**        |
|                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
| [YOLOv5-N (r6.1)][3]  | **1.9M**           | **4.5G**          | 640                   | **159** | - / 28.0%                  | -                 | -                 | -                | -                | -                |
| [YOLOv5-S (r6.1)][3]  | 7.2M               | 16.5G             | 640                   | 156     | - / 37.4%                  | -                 | -                 | -                | -                | -                |
| [YOLOv5-M (r6.1)][3]  | 21.2M              | 49.0G             | 640                   | 122     | - / 45.4%                  | -                 | -                 | -                | -                | -                |
| [YOLOv5-L (r6.1)][3]  | 46.5M              | 109.1G            | 640                   | 99      | - / 49.0%                  | -                 | -                 | -                | -                | -                |
| [YOLOv5-X (r6.1)][3]  | 86.7M              | 205.7G            | 640                   | 83      | - / **50.7%**              | -                 | -                 | -                | -                | -                |
|                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
| [YOLOR-CSP][4]        | 52.9M              | 120.4G            | 640                   | 106     | 51.1% / 50.8%              | 69.6%             | 55.7%             | 31.7%            | 55.3%            | 64.7%            |
| [YOLOR-CSP-X][4]      | 96.9M              | 226.8G            | 640                   | 87      | 53.0% / 52.7%              | 71.4%             | 57.9%             | 33.7%            | 57.1%            | 66.8%            |
| [YOLOv7-tiny-SiLU][5] | **6.2M**           | **13.8G**         | 640                   | **286** | 38.7% / 38.7%              | 56.7%             | 41.7%             | 18.8%            | 42.4%            | 51.9%            |
| [YOLOv7][5]           | 36.9M              | 104.7G            | 640                   | 161     | 51.4% / 51.2%              | 69.7%             | 55.9%             | 31.8%            | 55.5%            | 65.0%            |
| [YOLOv7-X][5]         | 71.3M              | 189.9G            | 640                   | 114     | **53.1% / 52.9%**          | **71.2%**         | **57.8%**         | **33.8%**        | **57.1%**        | **67.4%**        |
|                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
| [YOLOv5-N6 (r6.1)][3] | **3.2M**           | **18.4G**         | 1280                  | **123** | - / 36.0%                  | -                 | -                 | -                | -                | -                |
| [YOLOv5-S6 (r6.1)][3] | 12.6M              | 67.2G             | 1280                  | 122     | - / 44.8%                  | -                 | -                 | -                | -                | -                |
| [YOLOv5-M6 (r6.1)][3] | 35.7M              | 200.0G            | 1280                  | 90      | - / 51.3%                  | -                 | -                 | -                | -                | -                |
| [YOLOv5-L6 (r6.1)][3] | 76.8M              | 445.6G            | 1280                  | 63      | - / 53.7%                  | -                 | -                 | -                | -                | -                |
| [YOLOv5-X6 (r6.1)][3] | 140.7M             | 839.2G            | 1280                  | 38      | - / **55.0%**              | -                 | -                 | -                | -                | -                |
|                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
| [YOLOR-P6][4]         | **37.2M**          | **325.6G**        | 1280                  | **76**  | 53.9% / 53.5%              | 71.4%             | 58.9%             | 36.1%            | 57.7%            | 65.6%            |
| [YOLOR-W6][4]         | 79.8G              | 453.2G            | 1280                  | 66      | 55.2% / 54.8%              | 72.7%             | 60.5%             | 37.7%            | 59.1%            | 67.1%            |
| [YOLOR-E6][4]         | 115.8M             | 683.2G            | 1280                  | 45      | 55.8% / 55.7%              | 73.4%             | 61.1%             | 38.4%            | 59.7%            | 67.7%            |
| [YOLOR-D6][4]         | 151.7M             | 935.6G            | 1280                  | 34      | **56.5% / 56.1%**          | **74.1%**         | **61.9%**         | **38.9%**        | **60.4%**        | **68.7%**        |
|                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
| [YOLOv7-W6][5]        | **70.4M**          | **360.0G**        | 1280                  | **84**  | 54.9% / 54.6%              | 72.6%             | 60.1%             | 37.3%            | 58.7%            | 67.1%            |
| [YOLOv7-E6][5]        | 97.2M              | 515.2G            | 1280                  | 56      | 56.0% / 55.9%              | 73.5%             | 61.2%             | 38.0%            | 59.9%            | 68.4%            |
| [YOLOv7-D6][5]        | 154.7M             | 806.8G            | 1280                  | 44      | 56.6% / 56.3%              | 74.0%             | 61.8%             | 38.8%            | 60.1%            | 69.5%            |
| [YOLOv7-E6E][5]       | 151.7M             | 843.2G            | 1280                  | 36      | **56.8% / 56.8%**          | **74.4%**         | **62.1%**         | **39.3%**        | **60.5%**        | **69.0%**        |

[1]: https://github.com/Megvii-BaseDetection/YOLOX
[2]: https://github.com/PaddlePaddle/PaddleDetection
[3]: https://github.com/ultralytics/yolov5
[4]: https://github.com/WongKinYiu/yolor
[5]: https://github.com/WongKinYiu/yolov7

## Overview

Real-time object detection is an important component in many computer vision systems, including multi-object tracking, autonomous driving, robotics, and medical image analysis. In recent years, real-time object detection development has focused on designing efficient architectures and improving the inference speed of various CPUs, GPUs, and neural processing units (NPUs). YOLOv7 supports both mobile GPU and GPU devices, from the edge to the cloud.

Unlike traditional real-time object detectors that focus on architecture optimization, YOLOv7 introduces a focus on the optimization of the training process. This includes modules and optimization methods designed to improve the accuracy of object detection without increasing the inference cost, a concept known as the "trainable bag-of-freebies".

## Key Features

YOLOv7 introduces several key features:

1. **Model Re-parameterization**: YOLOv7 proposes a planned re-parameterized model, which is a strategy applicable to layers in different networks with the concept of gradient propagation path.

2. **Dynamic Label Assignment**: The training of the model with multiple output layers presents a new issue: "How to assign dynamic targets for the outputs of different branches?" To solve this problem, YOLOv7 introduces a new label assignment method called coarse-to-fine lead guided label assignment.

3. **Extended and Compound Scaling**: YOLOv7 proposes "extend" and "compound scaling" methods for the real-time object detector that can effectively utilize parameters and computation.

4. **Efficiency**: The method proposed by YOLOv7 can effectively reduce about 40% parameters and 50% computation of state-of-the-art real-time object detector, and has faster inference speed and higher detection accuracy.

## Usage Examples

As of the time of writing, Ultralytics does not currently support YOLOv7 models. Therefore, any users interested in using YOLOv7 will need to refer directly to the YOLOv7 GitHub repository for installation and usage instructions.

Here is a brief overview of the typical steps you might take to use YOLOv7:

1. Visit the YOLOv7 GitHub repository: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).

2. Follow the instructions provided in the README file for installation. This typically involves cloning the repository, installing necessary dependencies, and setting up any necessary environment variables.

3. Once installation is complete, you can train and use the model as per the usage instructions provided in the repository. This usually involves preparing your dataset, configuring the model parameters, training the model, and then using the trained model to perform object detection.

Please note that the specific steps may vary depending on your specific use case and the current state of the YOLOv7 repository. Therefore, it is strongly recommended to refer directly to the instructions provided in the YOLOv7 GitHub repository.

We regret any inconvenience this may cause and will strive to update this document with usage examples for Ultralytics once support for YOLOv7 is implemented.

## Citations and Acknowledgements

We would like to acknowledge the YOLOv7 authors for their significant contributions in the field of real-time object detection:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2022yolov7,
          title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
          author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
          journal={arXiv preprint arXiv:2207.02696},
          year={2022}
        }
        ```

The original YOLOv7 paper can be found on [arXiv](https://arxiv.org/pdf/2207.02696.pdf). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/WongKinYiu/yolov7). We appreciate their efforts in advancing the field and making their work accessible to the broader community.

## FAQ

### What makes YOLOv7 the most accurate real-time object detector?

YOLOv7 stands out due to its superior accuracy and speed. With an accuracy of 56.8% AP and the ability to process up to 161 FPS on a GPU V100, it surpasses all known real-time object detectors. Additionally, YOLOv7 introduces features like model re-parameterization, dynamic label assignment, and efficient parameter usage, enhancing both speed and accuracy. Check out the detailed comparison in the [source paper](https://arxiv.org/pdf/2207.02696.pdf).

### How does model re-parameterization work in YOLOv7?

Model re-parameterization in YOLOv7 involves optimizing the gradient propagation path across various network layers. This strategy effectively recalibrates the training process, improving detection accuracy without increasing the inference cost. For more details, refer to the [Model Re-parameterization section](#key-features) in the documentation.

### Why should I choose YOLOv7 over YOLOv5 or other object detectors?

YOLOv7 outperforms YOLOv5 and other detectors like YOLOR, YOLOX, and PPYOLOE in both speed and accuracy. For instance, YOLOv7 achieves 127 FPS faster and 10.7% higher accuracy compared to YOLOv5-N. Furthermore, YOLOv7 effectively reduces parameters and computation while delivering higher AP scores, making it an optimal choice for real-time applications. Learn more in our [comparison table](#comparison-of-sota-object-detectors).

### What datasets is YOLOv7 trained on, and how do they impact its performance?

YOLOv7 is trained exclusively on the MS COCO dataset without using additional datasets or pre-trained weights. This robust dataset provides a wide variety of images and annotations that contribute to YOLOv7's high accuracy and generalization capabilities. Explore more about dataset formats and usage in our [datasets section](https://docs.ultralytics.com/datasets/detect/coco/).

### Are there any practical YOLOv7 usage examples available?

Currently, Ultralytics does not directly support YOLOv7 models. However, you can find detailed installation and usage instructions on the YOLOv7 GitHub repository. These steps involve cloning the repository, installing dependencies, and setting up your environment to train and use the model. Follow the [YOLOv7 GitHub repository](https://github.com/WongKinYiu/yolov7) for the latest updates. For other examples, see our [usage examples](#usage-examples) section.
