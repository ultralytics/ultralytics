---
comments: true
description: Learn about Apple's Core AI framework and .aimodel format, how Core AI differs from Core ML, and the planned Ultralytics integration.
keywords: Apple Core AI, CoreAI, aimodel, Core ML comparison, CoreML, mlpackage, Apple Neural Engine, on-device inference, YOLO26, iOS 27, macOS 27
---

# Apple Core AI Integration

!!! warning "Core AI export is not yet available in Ultralytics"

    Ultralytics does **not currently support** `format=coreai` or direct export to Apple's `.aimodel` format. For production deployment on Apple devices today, use the supported [Core ML integration](coreml.md). Core AI support is planned for Q4 2026, after iOS 27 and macOS 27 become generally available.

[Core AI](https://developer.apple.com/core-ai/) is Apple's new framework for running neural networks directly on Apple silicon. It introduces the `.aimodel` model format, a modern Swift inference API, PyTorch-based conversion tools, ahead-of-time compilation, model specialization, and dedicated debugging and profiling tools.

Apple describes Core AI as the next evolution of on-device AI execution and the inference framework behind on-device Apple Intelligence. It is designed for current neural network architectures, from compact vision models to large generative models, and can schedule work across the CPU, GPU, and Apple Neural Engine (ANE).

Core AI is a new deployment path rather than a new name for Core ML. The frameworks use different model formats, conversion tools, runtime APIs, and application-integration patterns.

## Core AI and Core ML Compared

| Capability                       | Core AI                                                                              | Core ML                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Model artifact                   | `.aimodel`                                                                           | `.mlpackage` or `.mlmodel`                                                                |
| Ultralytics export               | Planned                                                                              | Available with `format=coreml`                                                            |
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

## Future Ultralytics Usage

!!! danger "Planned examples — these commands do not work yet"

    The following examples illustrate the intended integration and are **not available in the current Ultralytics release**. Use [`format=coreml`](coreml.md#exporting-yolo26-models-to-coreml) for a supported Apple export today.

After the planned integration ships, the [Python API](../usage/python.md) is expected to export a [YOLO26](../models/yolo26.md) model to `.aimodel` with a dedicated format value:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="coreai")  # Planned: creates yolo26n.aimodel
```

The equivalent planned [CLI](../usage/cli.md) command is:

```bash
yolo export model=yolo26n.pt format=coreai # Planned: not yet available
```

The final arguments, supported [YOLO tasks](../tasks/index.md), precision options, and dynamic-shape behavior will be documented in [Export mode](../modes/export.md) after the exporter is implemented and validated.

On iOS 27 or macOS 27, an application would then load and run the exported asset through Apple's Core AI Swift API. The function and tensor names below are illustrative; the supported Ultralytics output contract will be published with the exporter:

```swift
import CoreAI

let modelURL = Bundle.main.url(forResource: "yolo26n", withExtension: "aimodel")!
let model = try await AIModel(contentsOf: modelURL)
guard let function = try model.loadFunction(named: "main") else {
    throw AppError.missingInferenceFunction
}

let outputs = try await function.run(inputs: ["image": imageTensor])
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
- **Narrower export environment:** `coreai-torch` currently requires Python 3.11 or newer and recent PyTorch versions, which is much narrower than Ultralytics' supported Python and PyTorch range.
- **No current Ultralytics command:** `yolo export format=coreai` is not implemented, tested, or covered by Ultralytics' compatibility guarantees.
- **No Ultralytics application runtime yet:** The official [YOLO iOS app](https://github.com/ultralytics/yolo-ios-app) and [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) currently load Core ML artifacts through `MLModel` and Vision.
- **Application migration required:** A `.aimodel` cannot be substituted for an `.mlpackage`; model loading, preprocessing, inference calls, metadata handling, and output decoding need a Core AI implementation.
- **Limited production evidence:** Performance, power use, first-run specialization time, accuracy, and compression need validation across the supported YOLO task and device matrix.
- **No established legacy NMS pipeline:** Core ML can package an NMS stage for older YOLO detection models. The first Core AI integration is expected to focus on NMS-free YOLO26 models.

## Which Apple Format Should You Use?

Use **Core ML today** when you need:

- A supported Ultralytics export command
- Deployment across current and older Apple operating systems
- Integration with the Ultralytics iOS or Flutter SDK
- Vision framework image handling
- Tested FP16 and INT8 YOLO deployment
- Embedded NMS for compatible legacy detection models

Evaluate **Core AI in the future** when you can require iOS 27 or macOS 27 and need:

- The newest Apple on-device neural network runtime
- Explicit specialization and cache management
- Advanced dynamic or stateful model execution
- Custom Core AI operations or Metal kernels
- Detailed Core AI graph debugging and runtime profiling

Core ML and Core AI are expected to coexist while applications transition. Supporting Core AI does not immediately remove the need for Core ML because their deployment targets and application contracts differ.

## Ultralytics Roadmap

Ultralytics plans to evaluate a dedicated `coreai` export target in Q4 2026, after iOS 27 and macOS 27 are generally available. The initial work is expected to focus on NMS-free YOLO26 models and the `.aimodel` format while retaining Core ML for established Apple deployment targets.

Before Core AI can become a supported export format, the integration needs:

1. Export and numerical validation across detection, instance segmentation, semantic segmentation, classification, pose, and oriented bounding boxes.
2. FP16 and quantized accuracy testing against PyTorch and Core ML baselines.
3. On-device latency, memory, power, and specialization benchmarks.
4. Core AI model loading and preprocessing in the Ultralytics iOS SDK.
5. Flutter integration and a compatibility strategy for devices below iOS 27.
6. Stable Apple framework and conversion-tool releases.

Follow the [Ultralytics roadmap](https://www.ultralytics.com/roadmap) and release notes for availability. Until support ships, commands or third-party patches that produce `.aimodel` files are experimental and outside the supported Ultralytics export matrix.

## Additional Resources

- [Apple Core AI overview](https://developer.apple.com/core-ai/)
- [Core AI framework documentation](https://developer.apple.com/documentation/coreai)
- [Core AI PyTorch Extensions](https://apple.github.io/coreai-torch/)
- [Core AI Optimization](https://apple.github.io/coreai-optimization/)
- [Apple Core AI models repository](https://github.com/apple/coreai-models)
- [Ultralytics Core ML integration](coreml.md)

## FAQ

### Can Ultralytics export YOLO models to `.aimodel` today?

No. Ultralytics currently supports Apple's Core ML `.mlpackage` format through `model.export(format="coreml")`. A native Core AI export target is planned but is not yet part of the supported exporter.

### Is Core AI replacing Core ML?

Not immediately. Core AI is Apple's newer path for modern neural networks, while Core ML remains supported and provides broader operating-system coverage, Vision integration, and non-neural model support.

### Can I rename an `.mlpackage` to `.aimodel`?

No. They contain different model representations and are loaded by different frameworks. Conversion must start from the source model through the appropriate Apple toolchain.

### Will the Ultralytics Core AI integration replace `format=coreml`?

The initial integration is expected to coexist with Core ML. Any future replacement decision depends on operating-system adoption, stable tooling, performance, and downstream iOS and Flutter support.
