---
comments: true
description: Learn how to use NVIDIA DALI for GPU-accelerated preprocessing with Ultralytics YOLO models. Eliminate CPU bottlenecks by running letterbox resize, padding, and normalization on the GPU for faster TensorRT and Triton deployments.
keywords: NVIDIA DALI, GPU preprocessing, Ultralytics, YOLO, YOLO26, TensorRT, Triton Inference Server, letterbox, inference optimization, deep learning, computer vision, deployment, video processing, batch inference, DALI pipeline, CV-CUDA
---

# GPU-Accelerated Preprocessing with NVIDIA DALI

## Introduction

When deploying [Ultralytics YOLO](../models/index.md) models in production, [preprocessing](https://www.ultralytics.com/glossary/data-preprocessing) often becomes the bottleneck. While [TensorRT](../integrations/tensorrt.md) can run model [inference](../modes/predict.md) in just a few milliseconds, the CPU-based preprocessing (resize, pad, normalize) can take 2-10ms per image, especially at high resolutions. [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) (Data Loading Library) solves this by moving the entire preprocessing pipeline to the GPU.

This guide walks you through building DALI pipelines that exactly replicate Ultralytics YOLO preprocessing, integrating them with `model.predict()`, processing video streams, and deploying end-to-end with [Triton Inference Server](triton-inference-server.md).

!!! tip "Who is this guide for?"

    This guide is for engineers deploying YOLO models in production environments where CPU preprocessing is a measured bottleneck — typically [TensorRT](../integrations/tensorrt.md) deployments on NVIDIA GPUs, high-throughput video pipelines, or [Triton Inference Server](triton-inference-server.md) setups. If you're running standard inference with `model.predict()` and don't have a preprocessing bottleneck, the default CPU pipeline works well.

!!! summary "Quick Summary"

    - **Building a DALI pipeline?** Use `fn.resize(mode="not_larger")` + `fn.crop(out_of_bounds_policy="pad")` + `fn.crop_mirror_normalize` to replicate YOLO's letterbox preprocessing on GPU.
    - **Integrating with Ultralytics?** Pass the DALI output as a `torch.Tensor` to `model.predict()` — Ultralytics skips image preprocessing automatically.
    - **Deploying with Triton?** Use the DALI backend with a TensorRT ensemble for zero-CPU preprocessing.

## Why Use DALI for YOLO Preprocessing

In a typical YOLO inference pipeline, the preprocessing steps run on the CPU:

1. **Decode** the image (JPEG/PNG)
2. **Resize** while preserving aspect ratio
3. **Pad** to the target size (letterbox)
4. **Normalize** pixel values from `[0, 255]` to `[0, 1]`
5. **Convert** layout from HWC to CHW

With DALI, all these operations run on the GPU, eliminating the CPU bottleneck. This is especially valuable when:

| Scenario                                                                 | Why DALI Helps                                                                                                          |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| **Fast GPU inference**                                                   | [TensorRT](../integrations/tensorrt.md) engines with sub-millisecond inference make CPU preprocessing the dominant cost |
| **High-resolution inputs**                                               | 1080p and 4K video streams require expensive resize operations                                                          |
| **Large [batch sizes](https://www.ultralytics.com/glossary/batch-size)** | Server-side inference processing many images in parallel                                                                |
| **Limited CPU cores**                                                    | Edge devices like [NVIDIA Jetson](nvidia-jetson.md), or dense GPU servers with few CPU cores per GPU                    |

## Prerequisites

!!! warning "Linux Only"

    NVIDIA DALI supports **Linux only**. It is not available on Windows or macOS.

Install the required packages:

=== "CUDA 12.x"

    ```bash
    pip install ultralytics
    pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120
    ```

=== "CUDA 11.x"

    ```bash
    pip install ultralytics
    pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda110
    ```

**Requirements:**

- NVIDIA GPU (compute capability 5.0+ / Maxwell or newer)
- CUDA 11.0+ or 12.0+
- Python 3.10-3.14
- Linux operating system

## Understanding YOLO Preprocessing

Before building a DALI pipeline, it helps to understand exactly what Ultralytics does during preprocessing. The key class is `LetterBox` in [`ultralytics/data/augment.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py):

```python
from ultralytics.data.augment import LetterBox

letterbox = LetterBox(
    new_shape=(640, 640),  # Target size
    center=True,  # Center the image (pad equally on both sides)
    stride=32,  # Stride alignment
    padding_value=114,  # Gray padding (114, 114, 114)
)
```

The full preprocessing pipeline in [`ultralytics/engine/predictor.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py) performs these steps:

| Step | Operation                  | CPU Function                    | DALI Equivalent                               |
| ---- | -------------------------- | ------------------------------- | --------------------------------------------- |
| 1    | Letterbox resize           | `cv2.resize`                    | `fn.resize(mode="not_larger")`                |
| 2    | Centered padding           | `cv2.copyMakeBorder`            | `fn.crop(out_of_bounds_policy="pad")`         |
| 3    | BGR → RGB                  | `im[..., ::-1]`                 | `fn.decoders.image(output_type=types.RGB)`    |
| 4    | HWC → CHW + normalize /255 | `np.transpose` + `tensor / 255` | `fn.crop_mirror_normalize(std=[255,255,255])` |

The letterbox operation preserves the aspect ratio by:

1. Computing scale: `r = min(target_h / h, target_w / w)`
2. Resizing to `(round(w * r), round(h * r))`
3. Padding the remaining space with gray (`114`) to reach the target size
4. Centering the image so padding is distributed equally on both sides

## DALI Pipeline for YOLO

### Basic Pipeline (Padding at Bottom-Right)

This simpler version pads only at the bottom and right edges, which is sufficient for many deployment scenarios:

!!! example "DALI pipeline with bottom-right padding"

    ```python
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types


    @dali.pipeline_def(batch_size=8, num_threads=4, device_id=0)
    def yolo_dali_pipeline(image_dir, target_size=640):
        """DALI pipeline replicating YOLO preprocessing with bottom-right padding."""
        # Read and decode images on GPU
        jpegs, _ = fn.readers.file(file_root=image_dir, random_shuffle=False, name="Reader")
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        # Aspect-ratio-preserving resize (equivalent to LetterBox resize step)
        resized = fn.resize(
            images,
            resize_x=target_size,
            resize_y=target_size,
            mode="not_larger",  # Preserve aspect ratio, fit within target
            interp_type=types.INTERP_LINEAR,
            antialias=False,  # Match cv2.INTER_LINEAR behavior
        )

        # Pad to target size (bottom-right only)
        padded = fn.pad(
            resized,
            fill_value=114,  # YOLO padding value
            axes=(0, 1),  # Pad height and width only
            shape=[target_size, target_size],
        )

        # Normalize [0,255] -> [0,1] and convert HWC -> CHW
        output = fn.crop_mirror_normalize(
            padded,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )
        return output
    ```

### Centered Pipeline (Matching Ultralytics LetterBox)

This version exactly replicates the default Ultralytics preprocessing with centered padding, matching `LetterBox(center=True)`:

!!! example "DALI pipeline with centered padding (recommended)"

    ```python
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types


    @dali.pipeline_def(batch_size=8, num_threads=4, device_id=0)
    def yolo_dali_pipeline_centered(image_dir, target_size=640):
        """DALI pipeline replicating YOLO preprocessing with centered padding.

        Matches Ultralytics LetterBox(center=True) behavior exactly.
        """
        # Read and decode images on GPU
        jpegs, _ = fn.readers.file(file_root=image_dir, random_shuffle=False, name="Reader")
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        # Aspect-ratio-preserving resize
        resized = fn.resize(
            images,
            resize_x=target_size,
            resize_y=target_size,
            mode="not_larger",
            interp_type=types.INTERP_LINEAR,
            antialias=False,  # Match cv2.INTER_LINEAR (no antialiasing)
        )

        # Centered padding using fn.crop with out_of_bounds_policy
        # When crop size > image size, fn.crop centers the image and pads symmetrically
        padded = fn.crop(
            resized,
            crop=(target_size, target_size),
            out_of_bounds_policy="pad",
            fill_values=114,  # YOLO padding value
        )

        # Normalize and convert layout
        output = fn.crop_mirror_normalize(
            padded,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )
        return output
    ```

!!! tip "Why `fn.crop` for centered padding?"

    DALI's `fn.pad` operator only adds padding to the **right and bottom** edges. To get centered padding (matching Ultralytics `LetterBox(center=True)`), use `fn.crop` with `out_of_bounds_policy="pad"`. With the default `crop_pos_x=0.5` and `crop_pos_y=0.5`, the image is automatically centered with symmetric padding.

!!! warning "Antialias Mismatch"

    DALI's `fn.resize` enables antialiasing by default (`antialias=True`), while OpenCV's `cv2.resize` with `INTER_LINEAR` does **not** apply antialiasing. Always set `antialias=False` in DALI to match the CPU pipeline. Omitting this causes subtle numerical differences that can affect [model accuracy](https://www.ultralytics.com/glossary/accuracy).

### Running the Pipeline

!!! example "Build and run a DALI pipeline"

    ```python
    # Build and run the pipeline
    pipe = yolo_dali_pipeline_centered(image_dir="/path/to/images", target_size=640)
    pipe.build()

    # Get a batch of preprocessed images
    (output,) = pipe.run()

    # Convert to numpy or PyTorch tensors
    batch_np = output.as_cpu().as_array()  # Shape: (batch_size, 3, 640, 640)
    print(f"Output shape: {batch_np.shape}, dtype: {batch_np.dtype}")
    print(f"Value range: [{batch_np.min():.4f}, {batch_np.max():.4f}]")
    ```

## Using DALI with Ultralytics Predict

You can pass a preprocessed [PyTorch](https://www.ultralytics.com/glossary/pytorch) tensor directly to `model.predict()`. When a `torch.Tensor` is passed, Ultralytics **skips image preprocessing** (letterbox, BGR→RGB, HWC→CHW, and /255 normalization) and only performs device transfer and dtype casting before sending it to the model.

Since Ultralytics doesn't have access to the original image dimensions in this case, detection box coordinates are returned in the 640×640 letterboxed space. To map them back to original image coordinates, use [`scale_boxes`](../reference/utils/ops.md) which handles the exact rounding logic used by `LetterBox`:

```python
from ultralytics.utils.ops import scale_boxes

# boxes: tensor of shape (N, 4) in xyxy format, in 640x640 letterboxed coords
# Scale boxes from letterboxed (640, 640) back to original (orig_h, orig_w)
boxes = scale_boxes((640, 640), boxes, (orig_h, orig_w))
```

This applies to all external preprocessing paths — direct tensor input, video streams, and Triton deployment.

!!! example "DALI + Ultralytics predict"

    ```python
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    from ultralytics import YOLO

    # Load model
    model = YOLO("yolo26n.pt")

    # Create DALI iterator
    pipe = yolo_dali_pipeline_centered(image_dir="/path/to/images", target_size=640)
    pipe.build()
    dali_iter = DALIGenericIterator(pipe, ["images"], reader_name="Reader")

    # Run inference with DALI-preprocessed tensors
    for batch in dali_iter:
        images = batch[0]["images"]  # Already on GPU, shape (B, 3, 640, 640)
        results = model.predict(images, verbose=False)
        for result in results:
            print(f"Detected {len(result.boxes)} objects")
    ```

!!! tip "Zero Preprocessing Overhead"

    When you pass a `torch.Tensor` to `model.predict()`, the image preprocessing step takes ~0.004ms (essentially zero) compared to ~1-10ms with CPU preprocessing. The tensor must be in BCHW format, float32 (or float16), and normalized to `[0, 1]`. Ultralytics will still handle device transfer and dtype casting automatically.

## DALI with Video Streams

For real-time video processing, use `fn.external_source` to feed frames from any source — [OpenCV](https://www.ultralytics.com/glossary/opencv), GStreamer, or custom capture libraries:

!!! example "DALI pipeline for video stream preprocessing"

    === "Pipeline Definition"

        ```python
        import nvidia.dali as dali
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types


        @dali.pipeline_def(batch_size=1, num_threads=4, device_id=0)
        def yolo_video_pipeline(target_size=640):
            """DALI pipeline for processing video frames from external source."""
            # External source for feeding frames from OpenCV, GStreamer, etc.
            frames = fn.external_source(device="cpu", name="input")
            frames = fn.reshape(frames, layout="HWC")

            # Move to GPU and preprocess
            frames_gpu = frames.gpu()
            resized = fn.resize(
                frames_gpu,
                resize_x=target_size,
                resize_y=target_size,
                mode="not_larger",
                interp_type=types.INTERP_LINEAR,
                antialias=False,
            )
            padded = fn.crop(
                resized,
                crop=(target_size, target_size),
                out_of_bounds_policy="pad",
                fill_values=114,
            )
            output = fn.crop_mirror_normalize(
                padded,
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
            )
            return output
        ```

    === "Inference Loop (Simple OpenCV fallback)"

        ```python
        import cv2
        import numpy as np
        import torch

        from ultralytics import YOLO

        model = YOLO("yolo26n.engine")  # TensorRT model

        pipe = yolo_video_pipeline(target_size=640)
        pipe.build()

        cap = cv2.VideoCapture("video.mp4")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Feed BGR frame (convert to RGB for DALI)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pipe.feed_input("input", [np.array(frame_rgb)])
            (output,) = pipe.run()

            # Convert DALI output to torch tensor for inference.
            # This is a simple fallback path: using feed_input() with pipe.run() keeps a GPU->CPU->GPU copy.
            # For high-throughput deployments, prefer a reader-based pipeline plus DALIGenericIterator to keep data on GPU.
            tensor = torch.tensor(output.as_cpu().as_array()).to("cuda")
            results = model.predict(tensor, verbose=False)
        ```

## Triton Inference Server with DALI

For production deployment, combine DALI preprocessing with [TensorRT](../integrations/tensorrt.md) inference in [Triton Inference Server](triton-inference-server.md) using an ensemble model. This eliminates CPU preprocessing entirely — raw JPEG bytes go in, detections come out, with everything processed on the GPU.

### Model Repository Structure

```
model_repository/
├── dali_preprocessing/
│   ├── 1/
│   │   └── model.dali
│   └── config.pbtxt
├── yolo_trt/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
└── ensemble_dali_yolo/
    ├── 1/                  # Empty directory (required by Triton)
    └── config.pbtxt
```

### Step 1: Create the DALI Pipeline

Serialize the DALI pipeline for the Triton DALI backend:

!!! example "Serialize DALI pipeline for Triton"

    ```python
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types


    @dali.pipeline_def(batch_size=8, num_threads=4, device_id=0)
    def triton_dali_pipeline():
        """DALI preprocessing pipeline for Triton deployment."""
        # Input: raw encoded image bytes from Triton
        images = fn.external_source(device="cpu", name="DALI_INPUT_0")
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

        resized = fn.resize(
            images,
            resize_x=640,
            resize_y=640,
            mode="not_larger",
            interp_type=types.INTERP_LINEAR,
            antialias=False,
        )
        padded = fn.crop(
            resized,
            crop=(640, 640),
            out_of_bounds_policy="pad",
            fill_values=114,
        )
        output = fn.crop_mirror_normalize(
            padded,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )
        return output


    # Serialize pipeline to model repository
    pipe = triton_dali_pipeline()
    pipe.serialize(filename="model_repository/dali_preprocessing/1/model.dali")
    ```

### Step 2: Export YOLO to TensorRT

!!! example "Export YOLO model to TensorRT engine"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    model.export(format="engine", imgsz=640, half=True, batch=8)
    # Copy the .engine file to model_repository/yolo_trt/1/model.plan
    ```

### Step 3: Configure Triton

**dali_preprocessing/config.pbtxt:**

```protobuf
name: "dali_preprocessing"
backend: "dali"
max_batch_size: 8
input [
  {
    name: "DALI_INPUT_0"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]
output [
  {
    name: "DALI_OUTPUT_0"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
```

**yolo_trt/config.pbtxt:**

```protobuf
name: "yolo_trt"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 300, 6 ]
  }
]
```

**ensemble_dali_yolo/config.pbtxt:**

```protobuf
name: "ensemble_dali_yolo"
platform: "ensemble"
max_batch_size: 8
input [
  {
    name: "INPUT"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]
output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ 300, 6 ]
  }
]
ensemble_scheduling {
  step [
    {
      model_name: "dali_preprocessing"
      model_version: -1
      input_map {
        key: "DALI_INPUT_0"
        value: "INPUT"
      }
      output_map {
        key: "DALI_OUTPUT_0"
        value: "preprocessed_image"
      }
    },
    {
      model_name: "yolo_trt"
      model_version: -1
      input_map {
        key: "images"
        value: "preprocessed_image"
      }
      output_map {
        key: "output0"
        value: "OUTPUT"
      }
    }
  ]
}
```

!!! info "How Ensemble Mapping Works"

    The ensemble connects models through **virtual tensor names**. The `output_map` value `"preprocessed_image"` in the DALI step matches the `input_map` value `"preprocessed_image"` in the TensorRT step. These are arbitrary names that link one step's output to the next step's input — they don't need to match any model's internal tensor names.

### Step 4: Send Inference Requests

!!! info "Why `tritonclient` instead of `YOLO(\"http://...\")`?"

    Ultralytics has [built-in Triton support](triton-inference-server.md#running-inference) that handles pre/postprocessing automatically. However, it won't work with the DALI ensemble because `YOLO()` sends a preprocessed float32 tensor while the ensemble expects raw JPEG bytes. Use `tritonclient` directly for DALI ensembles, and the [built-in integration](triton-inference-server.md) for standard deployments without DALI.

!!! example "Send images to Triton ensemble"

    ```python
    import numpy as np
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Load image as raw bytes (JPEG/PNG encoded)
    image_data = np.fromfile("image.jpg", dtype="uint8")
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension

    # Create input
    input_tensor = httpclient.InferInput("INPUT", image_data.shape, "UINT8")
    input_tensor.set_data_from_numpy(image_data)

    # Run inference through the ensemble
    result = client.infer(model_name="ensemble_dali_yolo", inputs=[input_tensor])
    detections = result.as_numpy("OUTPUT")  # Shape: (1, 300, 6) -> [x1, y1, x2, y2, conf, class_id]

    # Filter by confidence (no NMS needed — YOLO26 is end-to-end)
    detections = detections[0]  # First image
    detections = detections[detections[:, 4] > 0.25]  # Confidence threshold
    print(f"Detected {len(detections)} objects")
    ```

!!! tip "Batching JPEG Images"

    When sending a batch of JPEG images to Triton, pad all encoded byte arrays to the same length (the maximum byte count in the batch). Triton requires homogeneous batch shapes for the input tensor.

## Supported Tasks

DALI preprocessing works with all YOLO tasks that use the standard `LetterBox` pipeline:

| Task                                        | Supported | Notes                                                    |
| ------------------------------------------- | --------- | -------------------------------------------------------- |
| [Detection](../tasks/detect.md)             | ✅        | Standard letterbox preprocessing                         |
| [Segmentation](../tasks/segment.md)         | ✅        | Same preprocessing as detection                          |
| [Pose Estimation](../tasks/pose.md)         | ✅        | Same preprocessing as detection                          |
| [Oriented Detection (OBB)](../tasks/obb.md) | ✅        | Same preprocessing as detection                          |
| [Classification](../tasks/classify.md)      | ❌        | Uses torchvision transforms (center crop), not letterbox |

## Limitations

- **Linux only**: DALI does not support Windows or macOS
- **NVIDIA GPU required**: No CPU-only fallback
- **Static pipeline**: Pipeline structure is defined at build time and cannot change dynamically
- **`fn.pad` is right/bottom only**: Use `fn.crop` with `out_of_bounds_policy="pad"` for centered padding
- **No rect mode**: DALI pipelines produce fixed-size outputs (e.g., 640×640). The `auto=True` rect mode that produces variable-size outputs (e.g., 384×640) is not supported. Note that while [TensorRT](../integrations/tensorrt.md) does support dynamic input shapes, a fixed-size DALI pipeline pairs naturally with a fixed-size engine for maximum throughput
- **Memory with multiple instances**: Using `instance_group` with `count` > 1 in Triton can cause high memory usage. Use the default instance group for the DALI model

## FAQ

### How does DALI preprocessing compare to CPU preprocessing speed?

The benefit depends on your pipeline. When GPU inference is already fast with [TensorRT](../integrations/tensorrt.md), CPU preprocessing at 2-10ms can become the dominant cost. DALI eliminates this bottleneck by running preprocessing on the GPU. The biggest gains are seen with high-resolution inputs (1080p, 4K), large [batch sizes](https://www.ultralytics.com/glossary/batch-size), and systems with limited CPU cores per GPU.

### Can I use DALI with PyTorch models (not just TensorRT)?

Yes. Use `DALIGenericIterator` to get preprocessed `torch.Tensor` outputs, then pass them to `model.predict()`. However, the performance benefit is greatest with [TensorRT](../integrations/tensorrt.md) models where inference is already very fast and CPU preprocessing becomes the bottleneck.

### What is the difference between `fn.pad` and `fn.crop` for padding?

`fn.pad` adds padding only to the **right and bottom** edges. `fn.crop` with `out_of_bounds_policy="pad"` centers the image and adds padding symmetrically on all sides, matching Ultralytics `LetterBox(center=True)` behavior.

### Does DALI produce pixel-identical results to CPU preprocessing?

Nearly identical. Set `antialias=False` in `fn.resize` to match OpenCV's `cv2.INTER_LINEAR`. Minor floating-point differences (< 0.001) may occur due to GPU vs CPU arithmetic, but these have no measurable impact on detection [accuracy](https://www.ultralytics.com/glossary/accuracy).

### What about CV-CUDA as an alternative to DALI?

[CV-CUDA](https://github.com/CVCUDA/CV-CUDA) is another NVIDIA library for GPU-accelerated vision processing. It provides per-operator control (like [OpenCV](https://www.ultralytics.com/glossary/opencv) but on GPU) rather than DALI's pipeline approach. CV-CUDA's `cvcuda.copymakeborder()` supports explicit per-side padding, making centered letterbox straightforward. Choose DALI for pipeline-based workflows (especially with [Triton](triton-inference-server.md)), and CV-CUDA for fine-grained operator-level control in custom inference code.
