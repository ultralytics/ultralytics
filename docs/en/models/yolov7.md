---
comments: true
description: Discover YOLOv7, the breakthrough real-time object detector with top speed and accuracy. Learn about key features, usage, and performance metrics.
keywords: YOLOv7, real-time object detection, Ultralytics, AI, computer vision, model training, object detector
---

# YOLOv7: Trainable Bag-of-Freebies

YOLOv7, released in July 2022, was a significant advancement in real-time object detection at its time of release. It achieved 56.8% AP on GPU V100, setting new benchmarks when introduced. YOLOv7 outperformed contemporary object detectors such as YOLOR, YOLOX, Scaled-YOLOv4, and YOLOv5 in speed and [accuracy](https://www.ultralytics.com/glossary/accuracy). The model is trained on the MS COCO dataset from scratch without using any other datasets or pretrained weights. Source code for YOLOv7 is available on GitHub. Note that newer models like [YOLO11](yolo11.md) and [YOLO26](yolo26.md) have since achieved higher accuracy with improved efficiency.

![YOLOv7 comparison with SOTA object detectors](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov7-comparison-sota-object-detectors.avif)

## Comparison of SOTA object detectors

From the results in the YOLO comparison table we know that the proposed method has the best speed-accuracy trade-off comprehensively. If we compare YOLOv7-tiny-SiLU with YOLOv5-N (r6.1), our method is 127 fps faster and 10.7% more accurate on AP. In addition, YOLOv7 has 51.4% AP at frame rate of 161 fps, while PPYOLOE-L with the same AP has only 78 fps frame rate. In terms of parameter usage, YOLOv7 is 41% less than PPYOLOE-L.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv7"]'></canvas>

If we compare YOLOv7-X with 114 fps inference speed to YOLOv5-L (r6.1) with 99 fps inference speed, YOLOv7-X can improve AP by 3.9%. If YOLOv7-X is compared with YOLOv5-X (r6.1) of similar scale, the inference speed of YOLOv7-X is 31 fps faster. In addition, in terms the amount of parameters and computation, YOLOv7-X reduces 22% of parameters and 8% of computation compared to YOLOv5-X (r6.1), but improves AP by 2.2% ([Source](https://arxiv.org/pdf/2207.02696)).

!!! tip "Performance"

    === "Detection (COCO)"

        | Model                 | Params<br><sup>(M)</sup> | FLOPs<br><sup>(G)</sup> | Size<br><sup>(pixels)</sup> | FPS     | AP<sup>test / val<br>50-95</sup> | AP<sup>test<br>50</sup> | AP<sup>test<br>75</sup> | AP<sup>test<br>S</sup> | AP<sup>test<br>M</sup> | AP<sup>test<br>L</sup> |
        | --------------------- | ------------------ | ----------------- | --------------------- | ------- | -------------------------- | ----------------- | ----------------- | ---------------- | ---------------- | ---------------- |
        | [YOLOX-S][1]          | **9.0**           | **26.8**         | 640                   | **102** | 40.5% / 40.5%              | -                 | -                 | -                | -                | -                |
        | [YOLOX-M][1]          | 25.3              | 73.8             | 640                   | 81      | 47.2% / 46.9%              | -                 | -                 | -                | -                | -                |
        | [YOLOX-L][1]          | 54.2              | 155.6            | 640                   | 69      | 50.1% / 49.7%              | -                 | -                 | -                | -                | -                |
        | [YOLOX-X][1]          | 99.1              | 281.9            | 640                   | 58      | **51.5% / 51.1%**          | -                 | -                 | -                | -                | -                |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [PPYOLOE-S][2]        | **7.9**           | **17.4**         | 640                   | **208** | 43.1% / 42.7%              | 60.5%             | 46.6%             | 23.2%            | 46.4%            | 56.9%            |
        | [PPYOLOE-M][2]        | 23.4              | 49.9             | 640                   | 123     | 48.9% / 48.6%              | 66.5%             | 53.0%             | 28.6%            | 52.9%            | 63.8%            |
        | [PPYOLOE-L][2]        | 52.2              | 110.1            | 640                   | 78      | 51.4% / 50.9%              | 68.9%             | 55.6%             | 31.4%            | 55.3%            | 66.1%            |
        | [PPYOLOE-X][2]        | 98.4              | 206.6            | 640                   | 45      | **52.2% / 51.9%**          | **69.9%**         | **56.5%**         | **33.3%**        | **56.3%**        | **66.4%**        |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOv5-N (r6.1)][3]  | **1.9**           | **4.5**          | 640                   | **159** | - / 28.0%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-S (r6.1)][3]  | 7.2               | 16.5             | 640                   | 156     | - / 37.4%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-M (r6.1)][3]  | 21.2              | 49.0             | 640                   | 122     | - / 45.4%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-L (r6.1)][3]  | 46.5              | 109.1            | 640                   | 99      | - / 49.0%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-X (r6.1)][3]  | 86.7              | 205.7            | 640                   | 83      | - / **50.7%**              | -                 | -                 | -                | -                | -                |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOR-CSP][4]        | 52.9              | 120.4            | 640                   | 106     | 51.1% / 50.8%              | 69.6%             | 55.7%             | 31.7%            | 55.3%            | 64.7%            |
        | [YOLOR-CSP-X][4]      | 96.9              | 226.8            | 640                   | 87      | 53.0% / 52.7%              | 71.4%             | 57.9%             | 33.7%            | 57.1%            | 66.8%            |
        | [YOLOv7-tiny-SiLU][5] | **6.2**           | **13.8**         | 640                   | **286** | 38.7% / 38.7%              | 56.7%             | 41.7%             | 18.8%            | 42.4%            | 51.9%            |
        | [YOLOv7][5]           | 36.9              | 104.7            | 640                   | 161     | 51.4% / 51.2%              | 69.7%             | 55.9%             | 31.8%            | 55.5%            | 65.0%            |
        | [YOLOv7-X][5]         | 71.3              | 189.9            | 640                   | 114     | **53.1% / 52.9%**          | **71.2%**         | **57.8%**         | **33.8%**        | **57.1%**        | **67.4%**        |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOv5-N6 (r6.1)][3] | **3.2**           | **18.4**         | 1280                  | **123** | - / 36.0%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-S6 (r6.1)][3] | 12.6              | 67.2             | 1280                  | 122     | - / 44.8%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-M6 (r6.1)][3] | 35.7              | 200.0            | 1280                  | 90      | - / 51.3%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-L6 (r6.1)][3] | 76.8              | 445.6            | 1280                  | 63      | - / 53.7%                  | -                 | -                 | -                | -                | -                |
        | [YOLOv5-X6 (r6.1)][3] | 140.7             | 839.2            | 1280                  | 38      | - / **55.0%**              | -                 | -                 | -                | -                | -                |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOR-P6][4]         | **37.2**          | **325.6**        | 1280                  | **76**  | 53.9% / 53.5%              | 71.4%             | 58.9%             | 36.1%            | 57.7%            | 65.6%            |
        | [YOLOR-W6][4]         | 79.8              | 453.2            | 1280                  | 66      | 55.2% / 54.8%              | 72.7%             | 60.5%             | 37.7%            | 59.1%            | 67.1%            |
        | [YOLOR-E6][4]         | 115.8             | 683.2            | 1280                  | 45      | 55.8% / 55.7%              | 73.4%             | 61.1%             | 38.4%            | 59.7%            | 67.7%            |
        | [YOLOR-D6][4]         | 151.7             | 935.6            | 1280                  | 34      | **56.5% / 56.1%**          | **74.1%**         | **61.9%**         | **38.9%**        | **60.4%**        | **68.7%**        |
        |                       |                    |                   |                       |         |                            |                   |                   |                  |                  |                  |
        | [YOLOv7-W6][5]        | **70.4**          | **360.0**        | 1280                  | **84**  | 54.9% / 54.6%              | 72.6%             | 60.1%             | 37.3%            | 58.7%            | 67.1%            |
        | [YOLOv7-E6][5]        | 97.2              | 515.2            | 1280                  | 56      | 56.0% / 55.9%              | 73.5%             | 61.2%             | 38.0%            | 59.9%            | 68.4%            |
        | [YOLOv7-D6][5]        | 154.7             | 806.8            | 1280                  | 44      | 56.6% / 56.3%              | 74.0%             | 61.8%             | 38.8%            | 60.1%            | 69.5%            |
        | [YOLOv7-E6E][5]       | 151.7             | 843.2            | 1280                  | 36      | **56.8% / 56.8%**          | **74.4%**         | **62.1%**         | **39.3%**        | **60.5%**        | **69.0%**        |

        [1]: https://github.com/Megvii-BaseDetection/YOLOX
        [2]: https://github.com/PaddlePaddle/PaddleDetection
        [3]: https://github.com/ultralytics/yolov5
        [4]: https://github.com/WongKinYiu/yolor
        [5]: https://github.com/WongKinYiu/yolov7

## Overview

Real-time object detection is an important component in many [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) systems, including multi-[object tracking](https://www.ultralytics.com/glossary/object-tracking), autonomous driving, [robotics](https://www.ultralytics.com/glossary/robotics), and [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis). In recent years, real-time object detection development has focused on designing efficient architectures and improving the inference speed of various CPUs, GPUs, and neural processing units (NPUs). YOLOv7 supports both mobile GPU and GPU devices, from the edge to the cloud.

Unlike traditional real-time object detectors that focus on architecture optimization, YOLOv7 introduces a focus on the optimization of the training process. This includes modules and optimization methods designed to improve the accuracy of object detection without increasing the inference cost, a concept known as the "trainable bag-of-freebies".

## Key Features

YOLOv7 introduces several key features:

1. **Model Re-parameterization**: YOLOv7 proposes a planned re-parameterized model, which is a strategy applicable to layers in different networks with the concept of gradient propagation path.

2. **Dynamic Label Assignment**: The training of the model with multiple output layers presents a new issue: "How to assign dynamic targets for the outputs of different branches?" To solve this problem, YOLOv7 introduces a new label assignment method called coarse-to-fine lead guided label assignment.

3. **Extended and Compound Scaling**: YOLOv7 proposes "extend" and "compound scaling" methods for the real-time object detector that can effectively utilize parameters and computation.

4. **Efficiency**: The method proposed by YOLOv7 can effectively reduce about 40% parameters and 50% computation of state-of-the-art real-time object detector, and has faster inference speed and higher detection accuracy.

## Usage Examples

As of the time of writing, Ultralytics only supports ONNX and TensorRT inference for YOLOv7.

### ONNX Export

To use YOLOv7 ONNX model with Ultralytics:

1. (Optional) Install Ultralytics and export an ONNX model to have the required dependencies automatically installed:

    ```bash
    pip install ultralytics
    yolo export model=yolo26n.pt format=onnx
    ```

2. Export the desired YOLOv7 model by using the exporter in the [YOLOv7 repo](https://github.com/WongKinYiu/yolov7):

    ```bash
    git clone https://github.com/WongKinYiu/yolov7
    cd yolov7
    python export.py --weights yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
    ```

3. Modify the ONNX model graph to be compatible with Ultralytics using the following script:

    ```python
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    # Load the ONNX model
    model_path = "yolov7/yolov7-tiny.onnx"  # Replace with your model path
    model = onnx.load(model_path)
    graph = model.graph

    # Fix input shape to batch size 1
    input_shape = graph.input[0].type.tensor_type.shape
    input_shape.dim[0].dim_value = 1

    # Define the output of the original model
    original_output_name = graph.output[0].name

    # Create slicing nodes
    sliced_output_name = f"{original_output_name}_sliced"

    # Define initializers for slicing (remove the first value)
    start = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_start")
    end = numpy_helper.from_array(np.array([7], dtype=np.int64), name="slice_end")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_steps")

    graph.initializer.extend([start, end, axes, steps])

    slice_node = helper.make_node(
        "Slice",
        inputs=[original_output_name, "slice_start", "slice_end", "slice_axes", "slice_steps"],
        outputs=[sliced_output_name],
        name="SliceNode",
    )
    graph.node.append(slice_node)

    # Define segment slicing
    seg1_start = numpy_helper.from_array(np.array([0], dtype=np.int64), name="seg1_start")
    seg1_end = numpy_helper.from_array(np.array([4], dtype=np.int64), name="seg1_end")
    seg2_start = numpy_helper.from_array(np.array([4], dtype=np.int64), name="seg2_start")
    seg2_end = numpy_helper.from_array(np.array([5], dtype=np.int64), name="seg2_end")
    seg3_start = numpy_helper.from_array(np.array([5], dtype=np.int64), name="seg3_start")
    seg3_end = numpy_helper.from_array(np.array([6], dtype=np.int64), name="seg3_end")

    graph.initializer.extend([seg1_start, seg1_end, seg2_start, seg2_end, seg3_start, seg3_end])

    # Create intermediate tensors for segments
    segment_1_name = f"{sliced_output_name}_segment1"
    segment_2_name = f"{sliced_output_name}_segment2"
    segment_3_name = f"{sliced_output_name}_segment3"

    # Add segment slicing nodes
    graph.node.extend(
        [
            helper.make_node(
                "Slice",
                inputs=[sliced_output_name, "seg1_start", "seg1_end", "slice_axes", "slice_steps"],
                outputs=[segment_1_name],
                name="SliceSegment1",
            ),
            helper.make_node(
                "Slice",
                inputs=[sliced_output_name, "seg2_start", "seg2_end", "slice_axes", "slice_steps"],
                outputs=[segment_2_name],
                name="SliceSegment2",
            ),
            helper.make_node(
                "Slice",
                inputs=[sliced_output_name, "seg3_start", "seg3_end", "slice_axes", "slice_steps"],
                outputs=[segment_3_name],
                name="SliceSegment3",
            ),
        ]
    )

    # Concatenate the segments
    concat_output_name = f"{sliced_output_name}_concat"
    concat_node = helper.make_node(
        "Concat",
        inputs=[segment_1_name, segment_3_name, segment_2_name],
        outputs=[concat_output_name],
        axis=1,
        name="ConcatSwapped",
    )
    graph.node.append(concat_node)

    # Reshape to [1, -1, 6]
    reshape_shape = numpy_helper.from_array(np.array([1, -1, 6], dtype=np.int64), name="reshape_shape")
    graph.initializer.append(reshape_shape)

    final_output_name = f"{concat_output_name}_batched"
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[concat_output_name, "reshape_shape"],
        outputs=[final_output_name],
        name="AddBatchDimension",
    )
    graph.node.append(reshape_node)

    # Get the shape of the reshaped tensor
    shape_node_name = f"{final_output_name}_shape"
    shape_node = helper.make_node(
        "Shape",
        inputs=[final_output_name],
        outputs=[shape_node_name],
        name="GetShapeDim",
    )
    graph.node.append(shape_node)

    # Extract the second dimension
    dim_1_index = numpy_helper.from_array(np.array([1], dtype=np.int64), name="dim_1_index")
    graph.initializer.append(dim_1_index)

    second_dim_name = f"{final_output_name}_dim1"
    gather_node = helper.make_node(
        "Gather",
        inputs=[shape_node_name, "dim_1_index"],
        outputs=[second_dim_name],
        name="GatherSecondDim",
    )
    graph.node.append(gather_node)

    # Subtract from 100 to determine how many values to pad
    target_size = numpy_helper.from_array(np.array([100], dtype=np.int64), name="target_size")
    graph.initializer.append(target_size)

    pad_size_name = f"{second_dim_name}_padsize"
    sub_node = helper.make_node(
        "Sub",
        inputs=["target_size", second_dim_name],
        outputs=[pad_size_name],
        name="CalculatePadSize",
    )
    graph.node.append(sub_node)

    # Build the [2, 3] pad array:
    # 1st row -> [0, 0, 0] (no padding at the start of any dim)
    # 2nd row -> [0, pad_size, 0] (pad only at the end of the second dim)
    pad_starts = numpy_helper.from_array(np.array([0, 0, 0], dtype=np.int64), name="pad_starts")
    graph.initializer.append(pad_starts)

    zero_scalar = numpy_helper.from_array(np.array([0], dtype=np.int64), name="zero_scalar")
    graph.initializer.append(zero_scalar)

    pad_ends_name = "pad_ends"
    concat_pad_ends_node = helper.make_node(
        "Concat",
        inputs=["zero_scalar", pad_size_name, "zero_scalar"],
        outputs=[pad_ends_name],
        axis=0,
        name="ConcatPadEnds",
    )
    graph.node.append(concat_pad_ends_node)

    pad_values_name = "pad_values"
    concat_pad_node = helper.make_node(
        "Concat",
        inputs=["pad_starts", pad_ends_name],
        outputs=[pad_values_name],
        axis=0,
        name="ConcatPadStartsEnds",
    )
    graph.node.append(concat_pad_node)

    # Create Pad operator to pad with zeros
    pad_output_name = f"{final_output_name}_padded"
    pad_constant_value = numpy_helper.from_array(
        np.array([0.0], dtype=np.float32),
        name="pad_constant_value",
    )
    graph.initializer.append(pad_constant_value)

    pad_node = helper.make_node(
        "Pad",
        inputs=[final_output_name, pad_values_name, "pad_constant_value"],
        outputs=[pad_output_name],
        mode="constant",
        name="PadToFixedSize",
    )
    graph.node.append(pad_node)

    # Update the graph's final output to [1, 100, 6]
    new_output_type = onnx.helper.make_tensor_type_proto(
        elem_type=graph.output[0].type.tensor_type.elem_type, shape=[1, 100, 6]
    )
    new_output = onnx.helper.make_value_info(name=pad_output_name, type_proto=new_output_type)

    # Replace the old output with the new one
    graph.output.pop()
    graph.output.extend([new_output])

    # Save the modified model
    onnx.save(model, "yolov7-ultralytics.onnx")
    ```

4. You can then load the modified ONNX model and run inference with it in Ultralytics normally:

    ```python
    from ultralytics import ASSETS, YOLO

    model = YOLO("yolov7-ultralytics.onnx", task="detect")

    results = model(ASSETS / "bus.jpg")
    ```

### TensorRT Export

1. Follow steps 1-2 in the [ONNX Export](#onnx-export) section.

2. Install the `TensorRT` Python package:

    ```bash
    pip install tensorrt
    ```

3. Run the following script to convert the modified ONNX model to TensorRT engine:

    ```python
    from ultralytics.utils.export import export_engine

    export_engine("yolov7-ultralytics.onnx", half=True)
    ```

4. Load and run the model in Ultralytics:

    ```python
    from ultralytics import ASSETS, YOLO

    model = YOLO("yolov7-ultralytics.engine", task="detect")

    results = model(ASSETS / "bus.jpg")
    ```

## Citations and Acknowledgments

We would like to acknowledge the YOLOv7 authors for their significant contributions in the field of real-time object detection:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2022yolov7,
          title={YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
          author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
          journal={arXiv preprint arXiv:2207.02696},
          year={2022}
        }
        ```

The original YOLOv7 paper can be found on [arXiv](https://arxiv.org/pdf/2207.02696). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/WongKinYiu/yolov7). We appreciate their efforts in advancing the field and making their work accessible to the broader community.

## FAQ

### What is YOLOv7 and why is it considered a breakthrough in real-time [object detection](https://www.ultralytics.com/glossary/object-detection)?

YOLOv7, released in July 2022, was a significant real-time object detection model that achieved excellent speed and accuracy at its time of release. It surpassed contemporary models such as YOLOX, YOLOv5, and PPYOLOE in both parameters usage and inference speed. YOLOv7's distinguishing features include its model re-parameterization and dynamic label assignment, which optimize its performance without increasing inference costs. For more technical details about its architecture and comparison metrics with other state-of-the-art object detectors, refer to the [YOLOv7 paper](https://arxiv.org/pdf/2207.02696).

### How does YOLOv7 improve on previous YOLO models like YOLOv4 and YOLOv5?

YOLOv7 introduces several innovations, including model re-parameterization and dynamic label assignment, which enhance the training process and improve inference accuracy. Compared to YOLOv5, YOLOv7 significantly boosts speed and accuracy. For instance, YOLOv7-X improves accuracy by 2.2% and reduces parameters by 22% compared to YOLOv5-X. Detailed comparisons can be found in the performance table [YOLOv7 comparison with SOTA object detectors](#comparison-of-sota-object-detectors).

### Can I use YOLOv7 with Ultralytics tools and platforms?

As of now, Ultralytics only supports YOLOv7 ONNX and TensorRT inference. To run the ONNX and TensorRT exported version of YOLOv7 with Ultralytics, check the [Usage Examples](#usage-examples) section.

### How do I train a custom YOLOv7 model using my dataset?

To install and train a custom YOLOv7 model, follow these steps:

1. Clone the YOLOv7 repository:
    ```bash
    git clone https://github.com/WongKinYiu/yolov7
    ```
2. Navigate to the cloned directory and install dependencies:
    ```bash
    cd yolov7
    pip install -r requirements.txt
    ```
3. Prepare your dataset and configure the model parameters according to the [usage instructions](https://github.com/WongKinYiu/yolov7) provided in the repository.
   For further guidance, visit the YOLOv7 GitHub repository for the latest information and updates.

4. After training, you can export the model to ONNX or TensorRT for use in Ultralytics as shown in [Usage Examples](#usage-examples).

### What are the key features and optimizations introduced in YOLOv7?

YOLOv7 offers several key features that revolutionize real-time object detection:

- **Model Re-parameterization**: Enhances the model's performance by optimizing gradient propagation paths.
- **Dynamic Label Assignment**: Uses a coarse-to-fine lead guided method to assign dynamic targets for outputs across different branches, improving accuracy.
- **Extended and Compound Scaling**: Efficiently utilizes parameters and computation to scale the model for various real-time applications.
- **Efficiency**: Reduces parameter count by 40% and computation by 50% compared to other state-of-the-art models while achieving faster inference speeds.

For further details on these features, see the [YOLOv7 Overview](#overview) section.
