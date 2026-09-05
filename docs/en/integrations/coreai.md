---
comments: true
description: Learn about Apple's Core AI framework and .aimodel format, how Core AI differs from Core ML, and how to export YOLO26 models with format=coreai.
keywords: Apple Core AI, CoreAI, aimodel, Core ML comparison, CoreML, mlpackage, Apple Neural Engine, on-device inference, YOLO26, iOS 27, macOS 27
---

# Apple Core AI Integration

!!! warning "Core AI export requires macOS 26 or later on Apple silicon"

    `coreai-core` publishes `macosx_26_0_arm64` wheels only, so export runs on Apple silicon Macs. The exported `.aimodel` runs on iOS 27 and macOS 27. The Ultralytics iOS and Flutter SDKs do not load `.aimodel` assets yet — use [Core ML](coreml.md) for those.

[Core AI](https://developer.apple.com/core-ai/) is Apple's new framework for running neural networks directly on Apple silicon. It introduces the `.aimodel` model format, a modern Swift inference API, PyTorch-based conversion tools, ahead-of-time compilation, model specialization, and dedicated debugging and profiling tools.

Apple describes Core AI as the next evolution of on-device AI execution and the inference framework behind on-device Apple Intelligence. It is designed for current neural network architectures, from compact vision models to large generative models, and can schedule work across the CPU, GPU, and Apple Neural Engine (ANE).

Core AI is a new deployment path rather than a new name for Core ML. The frameworks use different model formats, conversion tools, runtime APIs, and application-integration patterns.

## Core AI and Core ML Compared

| Capability                       | Core AI                                                                              | Core ML                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Model artifact                   | `.aimodel`                                                                           | `.mlpackage` or `.mlmodel`                                                                |
| Ultralytics export               | Available with `format=coreai`                                                       | Available with `format=coreml`                                                            |
| Apple runtime API                | `AIModel`, `InferenceFunction`, and `NDArray`                                        | `MLModel`, often through `VNCoreMLModel` and `VNCoreMLRequest`                            |
| Conversion workflow              | PyTorch `torch.export` through `coreai-torch`                                        | TorchScript conversion through `coremltools`                                              |
| Primary focus                    | Modern neural networks and generative AI                                             | Broad machine learning deployment, including neural and non-neural models                 |
| Image integration                | Applications prepare tensors or use Core AI image descriptors and buffers            | Direct integration with the Vision framework for image scaling, orientation, and requests |
| Hardware                         | CPU, GPU, and Apple Neural Engine                                                    | CPU, GPU, and Apple Neural Engine                                                         |
| Model preparation                | Specialization at installation or first use, with optional ahead-of-time compilation | Xcode or on-device model compilation                                                      |
| Custom operations                | Custom Core AI lowerings and Metal kernels                                           | Core ML custom layers and supported MIL operations                                        |
| Deployment availability          | New Apple operating-system generation; currently beta                                | Broad support across existing Apple operating systems                                     |
| Ultralytics iOS and Flutter SDKs | Not yet supported                                                                    | Fully supported                                                                           |

Core ML remains the appropriate choice when an application needs broad device coverage, Vision framework integration, or model types such as decision trees and tabular pipelines. Apple continues to support Core ML and directs developers with non-neural model types to it.

## How the Core AI Format Works

The Core AI authoring workflow starts from a PyTorch model:

```text
PyTorch model
    ↓ torch.export
ExportedProgram
    ↓ coreai-torch
Core AI program
    ↓ optimize and save
.aimodel
    ↓ specialize or compile ahead of time
Apple silicon executable
```

Apple's [`coreai-torch`](https://github.com/apple/coreai-torch) package converts a `torch.export.ExportedProgram` by lowering PyTorch ATen operations into Core AI operations. Unsupported operations can be implemented with a custom lowering or custom Metal kernel.

The resulting `.aimodel` is an unspecialized model asset. When an application prepares the model, Core AI specializes it for the target device. Applications can let this happen on first use, request specialization earlier, or ship an ahead-of-time compiled model to reduce initial loading time.

In Swift, applications load the asset with the Core AI framework, select an inference function, provide typed `NDArray` inputs, and receive named outputs. This is different from wrapping a Core ML model in a Vision request, so adopting Core AI requires an application runtime designed for `.aimodel` assets.

For implementation details, see Apple's documentation for [`AIModel`](https://developer.apple.com/documentation/coreai/aimodel), [model specialization and caching](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching), and [ahead-of-time compilation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time).

## Exporting YOLO26 Models to Core AI

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="coreai")  # creates 'yolo26n.aimodel'
        model.export(format="coreai", quantize=16)  # FP16 asset

        # Run the exported model
        coreai_model = YOLO("yolo26n.aimodel")
        results = coreai_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=coreai             # creates 'yolo26n.aimodel'
        yolo export model=yolo26n.pt format=coreai quantize=16 # FP16 asset

        yolo predict model=yolo26n.aimodel source='https://ultralytics.com/images/bus.jpg'
        ```

For the full argument list see [Export mode](../modes/export.md). The graph is static: it is traced at
the `imgsz` given to `export`, so predict on that same size. Ultralytics metadata travels inside the
asset's own `metadata.json`, so class names, stride and task survive the round trip.

### Choosing the head

With `nms=False`, YOLO26 exports its end-to-end head, which selects detections inside the graph. Core AI has
no top-k primitive, so that selection lowers to a full sort and is charged a fixed cost at the Apple
Neural Engine partition boundary — about 1.7 ms, regardless of `max_det`. Exporting with
`nms=None` emits the raw `(1, 84, 8400)` predictions instead and leaves non-maximum suppression
to the predictor:

```bash
yolo export model=yolo26n.pt format=coreai nms=None quantize=16
```

On an iPhone 17 Pro running iOS 27.0, YOLO26n at 640 measures 3.01 ms with the head in the graph and
1.28 ms without it (FP16, ahead-of-time compiled, three interleaved blocks of 50 iterations). Both
round-trip through `YOLO(...)` for inference. Use `nms=False` when a single graph call must return
finished detections, or keep the default `nms=None` for external NMS.

On iOS 27 or macOS 27, an application would then load and run the exported asset through Apple's Core AI Swift API. Exported assets use the entrypoint `main`, take a single `images` input of shape `[batch, 3, imgsz, imgsz]`, and return `output0`:

```swift
import CoreAI

let modelURL = Bundle.main.url(forResource: "yolo26n", withExtension: "aimodel")!
let model = try await AIModel(contentsOf: modelURL)
guard let function = try model.loadFunction(named: "main") else {
    throw AppError.missingInferenceFunction
}

let outputs = try await function.run(inputs: ["images": imageTensor])
```

Unlike the current [Core ML and Vision workflow](coreml.md#deploying-exported-yolo26-coreml-models), the future Core AI path will need to define image preprocessing, `NDArray` construction, model metadata, and output decoding in the [Ultralytics iOS SDK](https://github.com/ultralytics/yolo-ios-app). Apple provides current API details in the [Core AI framework documentation](https://developer.apple.com/documentation/coreai) and working model examples in the [Core AI models repository](https://github.com/apple/coreai-models).

## Advantages of Core AI

Core AI offers several promising advantages for future Ultralytics deployment:

- **Modern PyTorch export path:** Conversion starts from `torch.export`, preserving a more expressive PyTorch graph than the tracing workflow used by many existing exporters.
- **Fine-grained runtime control:** Applications can manage specialization, compiled-model caches, inference functions, memory, and compute placement.
- **Advanced model support:** Stateful execution, dynamic shapes, multiple functions in one artifact, and custom Metal kernels are designed for modern vision and generative architectures.
- **Dedicated developer tools:** The Core AI Debugger can inspect graphs and tensor values and trace them back to the originating Python code. Xcode and Instruments provide runtime profiling.
- **Zero-copy opportunities:** Core AI exposes storage and buffer controls intended to reduce copies between camera, graphics, and inference workloads.
- **Apple-silicon optimization:** Device specialization lets Apple optimize a model for the CPU, GPU, and Neural Engine available on the specific device.
- **Flexible compression:** Apple's Core AI Optimization tools support quantization, palettization, and pruning, including low-bit weight formats.

These capabilities could be particularly useful for future YOLO models with dynamic execution, larger multimodal components, or custom operations that do not map cleanly to existing Core ML operations.

## Current Disadvantages and Limitations

Core AI is not currently a replacement for the production Core ML path:

- **New operating systems required:** The public framework targets the iOS 27 and macOS 27 generation, while Core ML supports a much larger installed base.
- **Beta software:** Apple's Core AI framework and parts of its Python toolchain are still preliminary and may change before their stable releases.
- **Narrower export environment:** `coreai-torch` currently requires Python 3.11 or newer but below 3.14, plus recent PyTorch versions, which is much narrower than Ultralytics' supported Python and PyTorch range.
- **Export runs on macOS only:** `coreai-core` publishes `macosx_26_0_arm64` wheels only, so `format=coreai` needs an Apple silicon Mac on macOS 26 or later.
- **No Ultralytics application runtime yet:** The official [YOLO iOS app](https://github.com/ultralytics/yolo-ios-app) and [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) currently load Core ML artifacts through `MLModel` and Vision.
- **Application migration required:** A `.aimodel` cannot be substituted for an `.mlpackage`; model loading, preprocessing, inference calls, metadata handling, and output decoding need a Core AI implementation.
- **Limited production evidence:** Performance, power use, first-run specialization time, accuracy, and compression need validation across the supported YOLO task and device matrix.
- **No NMS pipeline:** Core ML can package an NMS stage for older YOLO detection models. Core AI exports raw one-to-many predictions by default; use `nms=False` for YOLO26's NMS-free head. Embedded NMS (`nms=True`) and `dynamic=True` are not supported. `coreai-torch` has no lowering for `torchvision::nms`, so NMS stays on the host.
- **Fixed input size:** The exported graph is traced at one `imgsz` and has no dynamic shapes, so predict at the size it was exported with.
- **FP16 assets can abort on load:** Some FP16 `.aimodel` assets fail to load their Apple Neural Engine program and MPSGraph raises a failed assertion, which ends the process rather than falling back. This happens inside Apple's runtime, before any Ultralytics code runs, and the same asset loads with a CPU-only specialization. Prefer FP32 until it is fixed upstream.

## Which Apple Format Should You Use?

Use **Core ML today** when you need:

- Deployment across current and older Apple operating systems
- Integration with the Ultralytics iOS or Flutter SDK
- Vision framework image handling
- Tested FP16 and INT8 YOLO deployment
- Embedded NMS for compatible legacy detection models

Evaluate **Core AI** when you can require iOS 27 or macOS 27 and need:

- The newest Apple on-device neural network runtime
- Explicit specialization and cache management
- Advanced dynamic or stateful model execution
- Custom Core AI operations or Metal kernels
- Detailed Core AI graph debugging and runtime profiling

Core ML and Core AI are expected to coexist while applications transition. Supporting Core AI does not immediately remove the need for Core ML because their deployment targets and application contracts differ.

## Ultralytics Roadmap

The dedicated `coreai` export target is implemented: export and numerical validation cover the supported YOLO26 task models and run continuously in Ultralytics CI on macOS 26, and FP16 latency is measured on-device. The remaining roadmap before Core AI reaches parity with the Core ML path:

1. Core AI model loading and preprocessing in the Ultralytics iOS SDK.
2. Flutter integration and a compatibility strategy for devices below iOS 27.
3. Stable Apple framework and conversion-tool releases (the iOS 27 and macOS 27 generation is currently in beta).
4. Memory, power, and specialization benchmarks across the supported device matrix.

Core ML remains the recommended target for applications that need the Ultralytics iOS or Flutter SDK or coverage below iOS 27; follow the [Ultralytics roadmap](https://www.ultralytics.com/roadmap) and release notes for the remaining items.

## Additional Resources

- [Apple Core AI overview](https://developer.apple.com/core-ai/)
- [Core AI framework documentation](https://developer.apple.com/documentation/coreai)
- [Core AI PyTorch Extensions](https://apple.github.io/coreai-torch/)
- [Core AI Optimization](https://apple.github.io/coreai-optimization/)
- [Apple Core AI models repository](https://github.com/apple/coreai-models)
- [Ultralytics Core ML integration](coreml.md)

## FAQ

### Can Ultralytics export YOLO models to `.aimodel`?

Yes. Export with `model.export(format="coreai")` or `yolo export format=coreai` on an Apple silicon Mac running macOS 26 or later; the exported `.aimodel` runs on iOS 27 and macOS 27. For the Ultralytics iOS and Flutter SDKs, and for operating systems below that generation, export Core ML `.mlpackage` files with `format="coreml"`.

### Is Core AI replacing Core ML?

Not immediately. Core AI is Apple's newer path for modern neural networks, while Core ML remains supported and provides broader operating-system coverage, Vision integration, and non-neural model support.

### Can I rename an `.mlpackage` to `.aimodel`?

No. They contain different model representations and are loaded by different frameworks. Conversion must start from the source model through the appropriate Apple toolchain.

### Will the Ultralytics Core AI integration replace `format=coreml`?

The initial integration is expected to coexist with Core ML. Any future replacement decision depends on operating-system adoption, stable tooling, performance, and downstream iOS and Flutter support.
