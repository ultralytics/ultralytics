---
comments: true
description: Discover how to enhance Ultralytics YOLO model performance using Intel's OpenVINO toolkit. Boost latency and throughput efficiently.
keywords: Ultralytics YOLO, OpenVINO optimization, deep learning, model inference, throughput optimization, latency optimization, AI deployment, Intel's OpenVINO, performance tuning
---

# OpenVINO Inference Optimization for YOLO

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ecosystem.avif" alt="OpenVINO Ecosystem">

## Introduction

When deploying [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models, particularly those for [object detection](https://www.ultralytics.com/glossary/object-detection) such as Ultralytics YOLO models, achieving optimal performance is crucial. This guide delves into leveraging [Intel's OpenVINO toolkit](https://docs.ultralytics.com/integrations/openvino/) to optimize inference, focusing on latency and throughput. Whether you're working on consumer-grade applications or large-scale deployments, understanding and applying these optimization strategies will ensure your models run efficiently on various devices.

## Optimizing for Latency

Latency optimization is vital for applications requiring immediate response from a single model given a single input, typical in consumer scenarios. The goal is to minimize the delay between input and inference result. However, achieving low latency involves careful consideration, especially when running concurrent inferences or managing multiple models.

### Key Strategies for Latency Optimization:

- **Single Inference per Device:** The simplest way to achieve low latency is by limiting to one inference at a time per device. Additional concurrency often leads to increased latency.
- **Leveraging Sub-Devices:** Devices like multi-socket CPUs or multi-tile GPUs can execute multiple requests with minimal latency increase by utilizing their internal sub-devices.
- **OpenVINO Performance Hints:** Utilizing OpenVINO's `ov::hint::PerformanceMode::LATENCY` for the `ov::hint::performance_mode` property during model compilation simplifies performance tuning, offering a device-agnostic and future-proof approach.

### Managing First-Inference Latency:

- **Model Caching:** To mitigate model load and compile times impacting latency, use model caching where possible. For scenarios where caching isn't viable, CPUs generally offer the fastest model load times.
- **Model Mapping vs. Reading:** To reduce load times, OpenVINO replaced model reading with mapping. However, if the model is on a removable or network drive, consider using `ov::enable_mmap(false)` to switch back to reading.
- **AUTO Device Selection:** This mode begins inference on the CPU, shifting to an accelerator once ready, seamlessly reducing first-inference latency.

## Optimizing for Throughput

Throughput optimization is crucial for scenarios serving numerous inference requests simultaneously, maximizing [resource utilization](https://www.ultralytics.com/blog/measuring-ai-performance-to-weigh-the-impact-of-your-innovations) without significantly sacrificing individual request performance.

### Approaches to Throughput Optimization:

1. **OpenVINO Performance Hints:** A high-level, future-proof method to enhance throughput across devices using performance hints.

    ```python
    import openvino.properties.hint as hints

    config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
    compiled_model = core.compile_model(model, "GPU", config)
    ```

2. **Explicit Batching and Streams:** A more granular approach involving explicit batching and the use of streams for advanced performance tuning.

### Designing Throughput-Oriented Applications:

To maximize throughput, applications should:

- Process inputs in parallel, making full use of the device's capabilities.
- Decompose data flow into concurrent inference requests, scheduled for parallel execution.
- Utilize the Async API with callbacks to maintain efficiency and avoid device starvation.

### Multi-Device Execution:

OpenVINO's multi-device mode simplifies scaling throughput by automatically balancing inference requests across devices without requiring application-level device management.

## Real-World Performance Gains

Implementing OpenVINO optimizations with Ultralytics YOLO models can yield significant performance improvements. As demonstrated in [benchmarks](https://docs.ultralytics.com/integrations/openvino/#openvino-yolov8-benchmarks), users can experience up to 3x faster inference speeds on Intel CPUs, with even greater accelerations possible across Intel's hardware spectrum including integrated GPUs, dedicated GPUs, and VPUs.

For example, when running YOLOv8 models on Intel Xeon CPUs, the OpenVINO-optimized versions consistently outperform their PyTorch counterparts in terms of inference time per image, without compromising on [accuracy](https://www.ultralytics.com/glossary/accuracy).

## Practical Implementation

To export and optimize your Ultralytics YOLO model for OpenVINO, you can use the [export](https://docs.ultralytics.com/modes/export/) functionality:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

# Export the model to OpenVINO format
model.export(format="openvino", half=True)  # Export with FP16 precision
```

After exporting, you can run inference with the optimized model:

```python
# Load the OpenVINO model
ov_model = YOLO("yolo26n_openvino_model/")

# Run inference with performance hints for latency
results = ov_model("path/to/image.jpg", verbose=True)
```

## Conclusion

Optimizing Ultralytics YOLO models for latency and throughput with OpenVINO can significantly enhance your application's performance. By carefully applying the strategies outlined in this guide, developers can ensure their models run efficiently, meeting the demands of various deployment scenarios. Remember, the choice between optimizing for latency or throughput depends on your specific application needs and the characteristics of the deployment environment.

For more detailed technical information and the latest updates, refer to the [OpenVINO documentation](https://docs.openvino.ai/2024/index.html) and [Ultralytics YOLO repository](https://github.com/ultralytics/ultralytics). These resources provide in-depth guides, tutorials, and community support to help you get the most out of your deep learning models.

---

Ensuring your models achieve optimal performance is not just about tweaking configurations; it's about understanding your application's needs and making informed decisions. Whether you're optimizing for [real-time responses](https://www.ultralytics.com/blog/real-time-inferences-in-vision-ai-solutions-are-making-an-impact) or maximizing throughput for large-scale processing, the combination of Ultralytics YOLO models and OpenVINO offers a powerful toolkit for developers to deploy high-performance AI solutions.

## FAQ

### How do I optimize Ultralytics YOLO models for low latency using OpenVINO?

Optimizing Ultralytics YOLO models for low latency involves several key strategies:

1. **Single Inference per Device:** Limit inferences to one at a time per device to minimize delays.
2. **Leveraging Sub-Devices:** Utilize devices like multi-socket CPUs or multi-tile GPUs which can handle multiple requests with minimal latency increase.
3. **OpenVINO Performance Hints:** Use OpenVINO's `ov::hint::PerformanceMode::LATENCY` during model compilation for simplified, device-agnostic tuning.

For more practical tips on optimizing latency, check out the [Latency Optimization section](#optimizing-for-latency) of our guide.

### Why should I use OpenVINO for optimizing Ultralytics YOLO throughput?

OpenVINO enhances Ultralytics YOLO model throughput by maximizing device resource utilization without sacrificing performance. Key benefits include:

- **Performance Hints:** Simple, high-level performance tuning across devices.
- **Explicit Batching and Streams:** Fine-tuning for advanced performance.
- **Multi-Device Execution:** Automated inference load balancing, easing application-level management.

Example configuration:

```python
import openvino.properties.hint as hints

config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
compiled_model = core.compile_model(model, "GPU", config)
```

Learn more about throughput optimization in the [Throughput Optimization section](#optimizing-for-throughput) of our detailed guide.

### What is the best practice for reducing first-inference latency in OpenVINO?

To reduce first-inference latency, consider these practices:

1. **Model Caching:** Use model caching to decrease load and compile times.
2. **Model Mapping vs. Reading:** Use mapping (`ov::enable_mmap(true)`) by default but switch to reading (`ov::enable_mmap(false)`) if the model is on a removable or network drive.
3. **AUTO Device Selection:** Utilize AUTO mode to start with CPU inference and transition to an accelerator seamlessly.

For detailed strategies on managing first-inference latency, refer to the [Managing First-Inference Latency section](#managing-first-inference-latency).

### How do I balance optimizing for latency and throughput with Ultralytics YOLO and OpenVINO?

Balancing latency and throughput optimization requires understanding your application needs:

- **Latency Optimization:** Ideal for real-time applications requiring immediate responses (e.g., consumer-grade apps).
- **Throughput Optimization:** Best for scenarios with many concurrent inferences, maximizing resource use (e.g., large-scale deployments).

Using OpenVINO's high-level performance hints and multi-device modes can help strike the right balance. Choose the appropriate [OpenVINO Performance hints](https://docs.ultralytics.com/integrations/openvino/#openvino-performance-hints) based on your specific requirements.

### Can I use Ultralytics YOLO models with other AI frameworks besides OpenVINO?

Yes, Ultralytics YOLO models are highly versatile and can be integrated with various AI frameworks. Options include:

- **TensorRT:** For NVIDIA GPU optimization, follow the [TensorRT integration guide](https://docs.ultralytics.com/integrations/tensorrt/).
- **CoreML:** For Apple devices, refer to our [CoreML export instructions](https://docs.ultralytics.com/integrations/coreml/).
- **[TensorFlow](https://www.ultralytics.com/glossary/tensorflow).js:** For web and Node.js apps, see the [TF.js conversion guide](https://docs.ultralytics.com/integrations/tfjs/).

Explore more integrations on the [Ultralytics Integrations page](https://docs.ultralytics.com/integrations/).
