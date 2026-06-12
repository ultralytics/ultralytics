---
comments: true
description: Learn how to contribute a new hardware export and runtime integration to Ultralytics YOLO, including exporter pipelines, AutoBackend wiring, dependencies, docs, and CI testing.
keywords: Ultralytics, YOLO, integration guide, export integration, hardware integration, AutoBackend, exporter, contribute, edge AI, accelerator, runtime, partner integration
---

# Contribute a New Hardware Export Integration

This guide outlines the requirements and best practices for hardware vendors, AI accelerator companies, and community contributors who want to integrate a new export pipeline and runtime into the Ultralytics ecosystem. It ensures seamless addition of custom compilation toolkits and runtime environments while preserving Ultralytics' core principles of maintainability, ease of use, and minimal dependencies.

If you are looking for examples of completed integrations, the [OpenVINO](openvino.md), [Sony IMX500](sony-imx500.md), and [ExecuTorch](executorch.md) pages are good references for the structure, writing style, and level of detail expected.

## Core Integration Principles

A successful integration follows three principles that apply to every export format already shipping in Ultralytics.

### Maintainability First

- Clean, well-documented code that the Ultralytics team can maintain long-term.
- Minimal code footprint within the core Ultralytics repository.
- Clear separation between integration-specific and core functionality.
- Standardized integration patterns across all formats.

### Zero-Impact on Core Users

- Integration dependencies are strictly optional.
- Core Ultralytics functionality remains unaffected when the integration's packages are not installed.
- No performance degradation for users who do not use the integration.

### Minimal Dependency Footprint

- Integrations should introduce zero new core dependencies.
- All integration-specific dependencies must be available as optional extras.
- Lightweight runtime with lazy loading of heavy components.
- Clear dependency isolation to prevent conflicts.

## Integration Architecture

Ultralytics YOLO provides several core modes for working with models, all documented in the [official documentation](../modes/index.md):

1. **[Train](../modes/train.md)**: Functionality to train and fine-tune Ultralytics YOLO models from scratch or with transfer learning.
2. **[Val](../modes/val.md)**: Assessment tools to evaluate the quality and performance of trained models on validation datasets.
3. **[Predict](../modes/predict.md)**: The inference engine that processes visual data to generate predictions. This mode supports the vast majority of export formats.
4. **[Track](../modes/track.md)**: Object tracking across multiple frames in video sequences, built on top of predict mode.
5. **[Benchmark](../modes/benchmark.md)**: Performance evaluation framework that assesses model speed and accuracy across various export formats in real-world scenarios.
6. **[Export](../modes/export.md)**: Model conversion tools that transform trained models into different deployment formats for various platforms and devices.

All modes are accessible through both the Python API and CLI:

!!! example "Ultralytics modes"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.train()  # Train mode
        model.val()  # Validation mode
        model.predict()  # Predict / inference mode
        model.track()  # Tracking mode
        model.benchmark()  # Benchmark mode
        model.export()  # Export mode
        ```

    === "CLI"

        ```bash
        yolo train model=yolo26n.pt
        yolo val model=yolo26n.pt
        yolo predict model=yolo26n.pt
        yolo track model=yolo26n.pt
        yolo benchmark model=yolo26n.pt
        yolo export model=yolo26n.pt
        ```

For a complete hardware integration, two core modes are primarily affected:

1. **Export Mode**: Where compilation toolkits convert YOLO models into hardware-optimized formats.
2. **Predict Mode**: Where runtime engines enable optimized inference on specialized hardware.

!!! note "Val, Track, and Benchmark are downstream of Predict"

    Val, Track, and Benchmark modes consume the Predict pipeline through `AutoBackend`, so they work automatically once Predict is correctly integrated — provided the backend preserves model metadata (`names`, `task`, `stride`) and produces output tensors that match the expected YOLO format.

This guide focuses on these two integration points.

## Export Pipeline Integration

Compilation toolkits are integrated into export mode, enabling users to convert YOLO models to hardware-specific formats. New export integrations should follow the same pattern as existing ones.

!!! example "Target user-facing API"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="format", arg1="arg", arg2="arg")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=format arg1=arg arg2=arg
        ```

Every export pipeline in Ultralytics resides in the [`ultralytics/engine/exporter.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py) module, which contains all logic for validation, settings, and format-specific implementations. The `Exporter` class is the primary interface that orchestrates this functionality across all export formats.

### Exporter Class Architecture

The `Exporter` class has the following responsibilities:

1. **Model Validation**: Ultralytics models and tasks have varying compatibility across export formats. To provide a smooth export experience, the system should validate format-specific compatibility upfront and clearly communicate any limitations to users before they attempt the export process.
2. **Argument Validation**: Each export format supports specific arguments (quantization levels, optimization settings). The principle is to minimize new arguments and reuse existing ones.
3. **Model Modification**: Changes to model heads or outputs should be minimal and non-invasive to the PyTorch model, applying external modifications when possible.
4. **Exception Handling**: All potential export failures must be properly handled with clear error messages.
5. **Calibration Data**: Exports requiring quantization data must use Ultralytics' [`get_int8_calibration_dataloader`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py) helper.

### Export Method Implementation Pattern

Each export format is encapsulated in a single `@try_export`-decorated method on the `Exporter` class. Keep these methods thin — heavy compilation logic lives in a helper module under `ultralytics/utils/export/partner.py` so `exporter.py` stays readable. The `@try_export` decorator handles export timing, success/failure logging, and validation that the output file is non-empty.

```python
@try_export
def export_partner_format(self, prefix=colorstr("Partner Format:")):
    """Export YOLO model to Partner's optimized format."""
    from ultralytics.utils.export.partner import torch2partner

    return torch2partner(
        model=self.model,
        output_dir=str(self.file).replace(self.file.suffix, "_partner_model/"),
        metadata=self.metadata,
        dataset=self.get_int8_calibration_dataloader(prefix),
        prefix=prefix,
    )
```

The method should return the output path. See `export_imx`, `export_rknn`, and `export_executorch` in [`exporter.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py) for live references that follow this pattern.

The helper module (e.g. `ultralytics/utils/export/partner.py`) is where the actual compilation lives — dependency imports, calibration handling, partner SDK calls, metadata writing (`YAML.save(Path(output_dir) / "metadata.yaml", metadata)`), and any external model wrapping.

#### Integration Registration

Register the new format in `export_formats()`. The `Arguments` list is the **only** arg-validation hook required: the generic `validate_args()` function automatically rejects any non-default export arg the user passes that is not on this list.

```python
def export_formats():
    """Return a dictionary of Ultralytics YOLO export formats."""
    x = [
        # ... existing formats ...
        ["Partner Format", "partner_format", "_partner_model", True, True, ["batch", "half", "int8", "nms"]],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))
```

Choose a unique `Suffix` (e.g. `"_partner_model"` or `".partner"`) — this same suffix drives runtime auto-detection (see [Format Detection](#format-detection)).

### Main Export Flow Integration

`Exporter.__call__` auto-dispatches to `export_<format>()` methods by name. For non-TensorFlow formats the dispatch is a single line at the end of `__call__`:

```python
# In Exporter.__call__:
if is_tf_format:
    # ... existing TF cascade ...
else:
    f = getattr(self, f"export_{fmt}")()
```

No manual flag wiring is required to register a new export method.

If your format needs pre-validation — mutually exclusive args, hardware checks, or default-arg coercion — add it inside `__call__` next to the existing format-specific blocks (search for `if fmt == "imx":` for an example):

```python
if fmt == "partner_format":
    if not self.args.int8:
        LOGGER.warning("Partner format requires int8=True, setting int8=True.")
        self.args.int8 = True
    if model.task not in {"detect", "segment"}:
        raise ValueError("Partner format only supports detection and segmentation models.")
```

#### Argument Validation Framework

Argument validation is generic — do not add per-format branches to `validate_args()`. Each format declares its supported argument names in the `Arguments` column of `export_formats()`, and `validate_args()` rejects any non-default export arg that is not on that list. To add support for a new arg combination, extend the `Arguments` list for your format entry.

### Model Modification Guidelines

When model modifications are necessary, follow these principles:

1. **Reuse Existing Wrappers**: Prefer the wrappers already shipped in `ultralytics/engine/exporter.py` and `ultralytics/utils/export/` over rolling your own.
2. **External Wrappers**: When a custom wrapper is unavoidable, build a `torch.nn.Module` that composes the model rather than mutating its internals.
3. **Temporary Changes**: Apply modifications only during export.
4. **Minimal Impact**: Make the smallest possible changes to achieve export compatibility.

For embedded NMS post-processing, use the existing [`NMSModel`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py) class — every NMS-capable export in the codebase (`export_torchscript`, `export_onnx`, `export_openvino`, `export_engine`, etc.) wraps the model with it via `NMSModel(self.model, self.args)`. Reuse it rather than reimplementing NMS post-processing in a new wrapper.

For graph-level rewrites (e.g. swapping ops the target compiler doesn't support), see the existing patterns: `FXModel` for IMX, `tf_wrapper` for TFLite/EdgeTPU, and `executorch_wrapper` for ExecuTorch — all under `ultralytics/utils/export/`.

### Error Handling Standards

The `@try_export` decorator already wraps every export method with timing, success/failure logging, and output validation — do not reimplement this layer.

Inside the helper module (`ultralytics/utils/export/<format>.py`), catch only the dependency-specific exceptions you can recover from or re-raise with actionable context. For everything else, let the exception propagate so `@try_export` can log it cleanly.

```python
# Inside ultralytics/utils/export/partner.py
from ultralytics.utils.checks import check_requirements


def torch2partner(model, output_dir, metadata, dataset=None, prefix=""):
    """Compile a YOLO model to Partner format."""
    check_requirements("partner-compiler>=2.0.0")
    import partner_compiler

    # Re-raise dependency-specific errors with an install hint
    try:
        compiled = partner_compiler.compile(model, output_dir=output_dir)
    except partner_compiler.CompilationError as e:
        raise RuntimeError(f"{prefix} compilation failed: {e}\nCheck model compatibility and arguments") from e

    # Other failures (I/O, OOM, etc.) propagate naturally to @try_export
    return str(output_dir)
```

### Dependency Management

!!! warning "Dependencies must integrate cleanly with the Ultralytics ecosystem"

    Every dependency added by a new integration must install without conflicts on top of `ultralytics`. This protects every user of the package, not just users of the new integration.

#### Standards

- **Distribution**: dependencies must be available on PyPI or via standard channels (Python wheels, system package managers, downloadable binary installers).
- **Minimal Footprint**: keep the number of sub-dependencies as small as possible. Lightweight runtime, lazy loading of heavy components.
- **No Conflicts**: dependencies must not pin versions that clash with `ultralytics` core requirements.
- **Cross-platform**: installation should succeed on Linux, macOS, and Windows (or scoped with platform markers when not).
- **Version Stability**: prefer permissive version bounds — see the version pinning tip below.

#### Adding Optional Dependencies to `pyproject.toml`

Register your integration's dependencies under `[project.optional-dependencies]` in [`pyproject.toml`](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml). Add packages used by the export pipeline to the existing `export` extras group; runtime-only packages can go in `export` as well unless they introduce a heavyweight dependency that would bloat that group, in which case open a discussion to add a dedicated extras group.

```toml
[project.optional-dependencies]
export = [
    # ... existing entries ...
    "partner-compiler>=2.0.0; platform_system == 'Linux'",  # Partner export compiler
    "partner-runtime-sdk>=2.0.0; platform_system == 'Linux'",  # Partner runtime SDK
]
```

Use platform markers (`platform_system`, `platform_machine`, `python_version`) to scope dependencies that don't install cleanly across all environments. Keep version bounds permissive (`>=`, optional `<X.0.0` upper bound only when a known incompatibility exists).

After editing, contributors install your integration with:

```bash
pip install "ultralytics[export]"
```

#### Validation Process

Before opening a PR:

1. **Dependency audit**: provide a complete dependency tree analysis.
2. **Conflict test**: demonstrate no conflicts with a clean `ultralytics` installation.
3. **Cross-platform test**: installation must succeed on at least one of Linux or macOS (Windows is also tested in CI and recommended where applicable).
4. **Version compatibility**: confirm dependencies work across all supported Python versions (3.8 to 3.12).
5. **Long-term stability**: demonstrate dependency maintenance commitments.

#### Common Blockers

Integrations cannot be merged if dependencies:

- Conflict with any existing Ultralytics dependencies.
- Require complex manual installation procedures.
- Have excessive sub-dependency trees.
- Require system-level modifications that could affect other software.

!!! tip "Version pinning"

    Avoid strict equality constraints (`==`) unless absolutely necessary. Prefer flexible version ranges with `>=` (and `<X.0.0` upper bounds only when a known incompatibility exists). For example, rather than `torch==2.5`, use `torch>=2.5` if your library is compatible with newer versions. Strict pinning forces unnecessary reinstallations and creates dependency conflicts for users who already have compatible newer versions installed.

## Runtime Integration (Predict Mode)

Runtime integration occurs through the [`ultralytics/nn/autobackend.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py) module, which provides the `AutoBackend` class responsible for dynamic backend selection and inference execution. This class serves as the universal interface that abstracts various inference engines and hardware backends.

### Runtime Dependencies

Runtime dependency management is as critical as export dependency management. Every runtime dependency must be thoroughly validated to ensure seamless integration without conflicts. Runtime integrations have additional complexity due to hardware-specific requirements and deployment scenarios.

#### Hardware Compatibility Requirements

- **Host Machine Fallback**: If the target hardware is not available on the host machine, users must still be able to validate model performance through intermediate models or performance emulators.
- **Emulation Support**: Provide accuracy and speed estimation capabilities without requiring physical hardware deployment.
- **Cross-platform compatibility**: For development environments where applicable.

#### Validation Without Hardware

- Provide intermediate format validation (e.g., ONNX Runtime simulation).
- Enable accuracy testing through software emulation or reference implementations.

### AutoBackend Class Architecture

`AutoBackend` is a thin dispatcher: it identifies the model format from the file path and delegates inference to a per-format backend class registered in `AutoBackend._BACKEND_MAP`. The actual runtime code lives in dedicated modules under [`ultralytics/nn/backends/`](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/backends), one file per format.

Adding a new runtime integration is a four-step process:

1. **Implement** a backend class in `ultralytics/nn/backends/<format>.py` that extends `BaseBackend`.
2. **Register** it in `AutoBackend._BACKEND_MAP`.
3. **Match** the suffix you used in `export_formats()` so format detection picks it up automatically.
4. **Update** the FP16, NHWC, and GPU-capable format sets in `AutoBackend.__init__()` if your runtime supports any of them.

### Runtime Integration Implementation Pattern

#### Backend Class Implementation

A backend extends [`BaseBackend`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/base.py) and implements two methods: `load_model()` and `forward()`. The base class already handles common attributes (`device`, `fp16`, `stride`, `names`, `task`, `imgsz`, `end2end`, `dynamic`, `metadata`) and provides `apply_metadata()` to populate them from the `metadata.yaml` saved during export.

```python
# ultralytics/nn/backends/partner.py

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class PartnerBackend(BaseBackend):
    """Partner runtime inference backend."""

    def load_model(self, weight: str | Path) -> None:
        """Load the partner model from a directory or single file."""
        LOGGER.info(f"Loading {weight} for Partner runtime inference...")
        check_requirements("partner-runtime-sdk>=2.0.0")

        import partner_runtime_sdk

        w = Path(weight)
        if w.is_dir():
            model_file = next(w.rglob("*.partner"))
            metadata_file = w / "metadata.yaml"
        else:
            model_file = w
            metadata_file = w.parent / "metadata.yaml"

        self.model = partner_runtime_sdk.Model(str(model_file), device=self.device)
        self.model.load()

        if metadata_file.exists():
            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor):
        """Run inference using the partner runtime."""
        outputs = self.model.predict(im.cpu().numpy())
        if isinstance(outputs, list):
            return [torch.from_numpy(x).to(self.device) for x in outputs]
        return torch.from_numpy(outputs).to(self.device)
```

See [`ultralytics/nn/backends/executorch.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/executorch.py), [`openvino.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/openvino.py), and [`tensorrt.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/tensorrt.py) for live references that follow this pattern.

#### Backend Registration

Register the new backend in `AutoBackend._BACKEND_MAP`. The dictionary key must match the `Argument` value used in `export_formats()`.

```python
# In ultralytics/nn/autobackend.py

from .backends.partner import PartnerBackend


class AutoBackend(nn.Module):
    _BACKEND_MAP = {
        # ... existing backends ...
        "partner_format": PartnerBackend,
    }
```

If the format supports FP16 inference, expects NHWC inputs, or runs on a CUDA GPU, also extend the relevant sets inside `AutoBackend.__init__()`:

```python
# Add only if FP16 inference is supported by the runtime
fp16 &= format in {"pt", "torchscript", "onnx", "openvino", "engine", "triton"}

# Add only if the runtime expects NHWC tensors instead of NCHW
self.nhwc = format in {"coreml", "saved_model", "pb", "tflite", "edgetpu", "rknn"}

# Add only if the runtime supports CUDA — otherwise device=cuda is silently downgraded to CPU
if (
    isinstance(device, torch.device)
    and torch.cuda.is_available()
    and device.type != "cpu"
    and format not in {"pt", "torchscript", "engine", "onnx", "paddle"}
):
    device = torch.device("cpu")
```

#### Format Detection

Format auto-detection is driven entirely by the `Suffix` column of `export_formats()`. The shared [`AutoBackend._model_type()`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py) method walks the registered suffixes and returns the first match, so picking a unique suffix in `export_formats()` (e.g. `"_partner_model"` or `".partner"`) is enough — no edits to `_model_type()` are required.

## Documentation

This section explains how to create documentation pages for new integrations in the Ultralytics package. Integration documentation pages live in [`docs/en/integrations/`](https://github.com/ultralytics/ultralytics/tree/main/docs/en/integrations) and follow a specific structure to ensure consistency and usability.

### File Structure

Integration documentation pages should be placed in:

```text
docs/en/integrations/
├── index.md                  # Main integrations overview page
├── your-integration.md       # Your new integration page
├── existing-integration.md   # Example of an existing integration
└── ...
```

### Page Structure Template

#### Page Header (Front Matter)

Every integration page must start with YAML front matter:

```yaml
---
description: Brief description of the integration for SEO and meta tags
keywords: keyword1, keyword2, keyword3, Ultralytics, YOLO, integration
---
```

#### Page Title and Introduction

Begin every integration guide with a clear value proposition and selection criteria. Start with why users should choose this integration over alternatives, including specific use cases, benefits, and any limitations or requirements that might influence their decision.

```markdown
# Integration Name

<img width="1024" src="https://path/to/integration-banner-image.png" alt="Integration Name banner">

[Brief description of what the integration does and why it's useful. Include a
link to the integration's main website if applicable.]

This integration enables [specific functionality] with Ultralytics YOLO models,
providing [key benefits].
```

#### Key Features Section

```markdown
## Key Features

- **Feature 1**: Description of what this feature provides
- **Feature 2**: Description of what this feature provides
- **Feature 3**: Description of what this feature provides
- **Easy Integration**: Simple setup process with minimal configuration required
```

#### Installation Section

Keep installation and setup sections comprehensive yet streamlined. Minimize steps wherever possible — if an integration requires extensive documentation to explain, it is likely too complex for users to adopt. Prioritize conciseness, thoroughly test all instructions, and verify functionality in the target environment before publishing.

````markdown
## Installation

### Prerequisites

Before using this integration, ensure you have:

- Python 3.8 or higher
- Ultralytics package installed
- [Any other specific requirements]

### Install Required Packages

```bash
pip install ultralytics integration-package-name
```

### Additional Setup (if required)

If the integration requires additional configuration:

```bash
# Example additional setup commands
export INTEGRATION_API_KEY="your-api-key"
```
````

#### Usage Examples Section

The usage section should clearly demonstrate how to implement the integration with practical examples. Focus on common use cases and provide working code snippets that users can easily adapt.

````markdown
## Usage

```python
from ultralytics import YOLO

# Load a YOLO model
model = YOLO("yolo26n.pt")

# Export model
path = model.export(format="format")

export_model = YOLO(path)

# Use the integration with your model
results = export_model.predict(source="path/to/image.jpg")
```
````

#### Export Arguments

All export function arguments must include comprehensive documentation with explicit typing and default values.

```markdown
The integration supports the following configuration options:

| Parameter | Type   | Default     | Description           |
| --------- | ------ | ----------- | --------------------- |
| `param1`  | `str`  | `"default"` | Description of param1 |
| `param2`  | `int`  | `100`       | Description of param2 |
| `param3`  | `bool` | `True`      | Description of param3 |
```

#### Benchmarks

All model exports must include a benchmark section using standardized test configurations to guide users in selecting optimal export formats based on accuracy and speed requirements. Benchmarks must be conducted using the same dataset consistently across all export formats, utilizing default Ultralytics configurations to maintain consistency across formats including PyTorch, ONNX, TensorRT, OpenVINO, CoreML, TensorFlow variants, and mobile formats like NCNN and MNN. Test both CPU and GPU performance where supported, measure key metrics including mAP50-95 [accuracy](https://www.ultralytics.com/glossary/accuracy), inference time (ms/image), FPS, model file size, and memory usage, then present results in a standardized table showing export status (✅/⚠️/❌), performance deltas versus the PyTorch baseline, and clear recommendations for when each format is optimal.

### Documentation Standards

#### Writing Style

1. **Clear and Concise**: Use simple, straightforward language.
2. **Active Voice**: Prefer active voice over passive voice.
3. **Consistent Terminology**: Use consistent terms throughout the document.
4. **User-Focused**: Write from the user's perspective.

#### Code Examples

1. **Complete Examples**: Provide runnable code examples.
2. **Comments**: Include helpful comments in code blocks.
3. **Error Handling**: Show proper error handling where relevant.
4. **Real Paths**: Use realistic file paths and examples.

#### Links and References

1. **Internal Links**: Use relative links to other documentation pages.
2. **External Links**: Include links to official integration documentation.
3. **Glossary Links**: Link to the [Ultralytics glossary](https://www.ultralytics.com/glossary) for technical terms when appropriate.

#### Images and Media

1. **Banner Image**: Include a banner image at the top (1024px width recommended).
2. **Screenshots**: Use screenshots to illustrate complex setup steps.
3. **Alt Text**: Always include descriptive alt text for images.

#### SEO and Metadata

1. **Meta Description**: Write a compelling meta description (150–160 characters).
2. **Keywords**: Include relevant keywords in the front matter.
3. **Headers**: Use proper header hierarchy (H1, H2, H3).
4. **Internal Linking**: Link to related pages within the documentation.

### Example Integration Documentation

Reference existing integration documentation pages as examples:

- [OpenVINO Integration](openvino.md)
- [Sony IMX500 Integration](sony-imx500.md)
- [ExecuTorch Integration](executorch.md)

These pages demonstrate the structure, writing style, and level of detail expected for integration documentation.

## Testing

All new integrations must be incorporated into the Ultralytics CI/CD pipeline to ensure continuous validation and prevent regressions. This automated testing framework is essential for long-term maintenance and reliability.

### Ultralytics CI/CD Testing Framework

Ultralytics maintains an automated testing infrastructure that validates all integrations:

- **Continuous Integration**: All export and runtime functionality is tested in the CI pipeline.
- **Cross-Platform Validation**: Automated testing for Linux, macOS, and Windows environments.
- **Python Compatibility**: Testing across all supported Python versions (3.8–3.12).
- **Functional Unit Tests**: Comprehensive unit testing for both export pipelines and runtime integrations.
- **Regression Testing**: Automated regression testing to prevent performance and functionality degradation.
- **Performance Benchmarking**: Automated performance testing and regression detection.
- **Dependency Validation**: Testing of optional dependency installation and conflict detection.

### Testing Requirements

Contributors must provide and maintain:

1. **CI Support**: Assistance with CI configuration for integration-specific testing requirements.
2. **Response Commitment**: Prompt response to CI test failures and regression issues.

#### Testing Categories Covered

- **Export Pipeline Testing**: Validation of export functionality, format conversion, and error handling.
- **Runtime Integration Testing**: AutoBackend integration, format detection, and inference validation.
- **Cross-Platform Compatibility**: Testing across different operating systems and Python versions.
- **Performance Regression**: Continuous monitoring of export times, inference speed, and memory usage.
- **Accuracy Preservation**: Validation that the new format maintains model accuracy standards.

#### Required Pre-Submission Verification

Before submitting your integration PR, verify the export and runtime end-to-end by running [Val mode](../modes/val.md) against the exported model on the appropriate dataset. This is the only way to confirm the [Accuracy Preservation](#accuracy-preservation) targets in the Performance Standards section are actually met.

!!! example "Verify accuracy with Val mode"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported model
        model = YOLO("yolo26n_partner_model")

        # Run validation and confirm mAP / accuracy is within tolerance
        metrics = model.val(data="coco8.yaml")
        print(metrics.box.map)  # mAP50-95
        ```

    === "CLI"

        ```bash
        yolo val model=yolo26n_partner_model data=coco8.yaml
        ```

Include the Val results (mAP / Top-1 / Top-5 against the PyTorch baseline) in the PR description so reviewers can verify the integration meets the accuracy bar without re-running the export themselves.

This testing approach ensures that integrations maintain high quality standards and reliability within the Ultralytics ecosystem.

## Performance Standards

### Export Time Performance

- **Baseline Requirement**: Export time without a calibration dataset must not exceed 5× the time of an equivalent ONNX export.
- **Large Model Handling**: For models >100 MB, export time should not exceed 5 minutes.
- **Progress Reporting**: Implement progress indicators for exports exceeding 60 seconds.
- **Memory Efficiency**: Peak memory usage during export must not exceed 5× the base model size, excluding calibration dataset loading.
- **Calibration Memory Management**: The calibration dataset should never be fully loaded into memory. Instead, data should be read sequentially, one sample at a time, to prevent unexpected out-of-memory errors.

### Accuracy Preservation

- **Baseline Accuracy**: Maintain ≥95% of the original model accuracy (mAP / Top-1 / Top-5).
- **Quantization Tolerance**: For quantized formats, maintain ≥85% of the original.

### Export Consistency Standards

- **Reproducible Results**: Ensure consistent outputs across runs.
- **Cross-Platform Consistency**: Results should be consistent across different platforms (where supported).
- **Version Stability**: Maintain accuracy across minor version updates.

### Resource Utilization

- **CPU Usage**: Efficient CPU utilization without excessive overhead.
- **GPU Memory**: Optimal GPU memory usage for accelerated formats.
- **I/O Efficiency**: Minimize disk I/O during model loading and inference.

## Maintenance and Support

### Long-term Maintenance

- Contributors commit to maintaining their integration for the duration of the agreement.
- Regular updates for new Ultralytics releases.
- Security patch responsibility for integration-specific dependencies.

### Support Channels

- Clear escalation path for integration problems.
- Response time commitments for critical issues.

### Deprecation Policy

- Two-month notice for any breaking changes.
- Migration guides for version updates.
- Backward compatibility maintenance when possible.

## Quality Assurance Checklist

Use this checklist as a final review before opening a pull request for a new integration.

### Pre-Integration Requirements

- [ ] **Zero Impact Validation**: Core Ultralytics functionality unaffected by the integration.
- [ ] **Dependency Compatibility**: All dependencies compatible with existing Ultralytics requirements.
- [ ] **Optional Installation**: All integration dependencies installable as optional extras.
- [ ] **Python Compatibility**: Support for Python ≥ 3.8.

### Export Pipeline Requirements

- [ ] **Export Method Implementation**: Complete export method following the standardized pattern.
- [ ] **Argument Validation**: Integration with the existing argument validation framework.
- [ ] **Model Compatibility**: Clear model type and task compatibility definitions.
- [ ] **Error Handling**: Comprehensive exception handling with helpful error messages.
- [ ] **Metadata Integration**: Proper metadata saving and loading following Ultralytics conventions.
- [ ] **Quantization Support**: INT8 / FLOAT16 calibration using the Ultralytics base dataloader when applicable.

### Runtime Pipeline Requirements

- [ ] **AutoBackend Integration**: Proper integration into the `AutoBackend` class.
- [ ] **Device Management**: Intelligent device selection and fallback mechanisms.
- [ ] **Output Tensors**: `forward()` returns torch tensors matching the YOLO output shape contract (the `Results` object is constructed downstream by the predictor — backends only emit tensors).
- [ ] **Metadata Handling**: Consistent metadata extraction and processing.
- [ ] **Hardware Fallback**: Graceful handling when target hardware is unavailable.
- [ ] **Val Mode Verified**: `model.val()` runs end-to-end against the exported model with mAP / accuracy within the targets specified in [Performance Standards](#performance-standards), and Val results are included in the PR description.

### Documentation Requirements

- [ ] **Installation Guide**: Step-by-step installation instructions for all scenarios.
- [ ] **Export Documentation**: Complete export functionality documentation with examples.
- [ ] **Runtime Documentation**: Runtime usage documentation.
- [ ] **API Reference**: Complete function and class documentation.
- [ ] **Performance Documentation**: Benchmark results and performance characteristics.
- [ ] **Troubleshooting Guide**: Common issues and solutions for hardware.
- [ ] **Emulation Guide**: Detailed documentation on emulation capabilities and limitations.
- [ ] **Hardware Requirements**: Clear hardware requirements and compatibility matrix.

### Dependency Management Requirements

- [ ] **Dependency Audit**: Complete dependency tree analysis provided.
- [ ] **Conflict Analysis**: No conflicts with existing Ultralytics dependencies.
- [ ] **Minimal Footprint**: Minimal sub-dependency requirements.
- [ ] **Standard Distribution**: Available through PyPI or standard package managers.
- [ ] **Emulation Dependencies**: Lightweight emulation dependencies for target platforms.

### Long-term Support Requirements

- [ ] **Maintenance Commitment**: Maintenance commitment for the duration of the agreement.
- [ ] **Update Responsibility**: Clear update process for new Ultralytics releases.
- [ ] **Security Commitment**: Responsibility for security patches.
- [ ] **Documentation Maintenance**: Commitment to keep documentation current.
- [ ] **Performance Monitoring**: Ongoing performance regression monitoring.

### Code Quality Requirements

- [ ] **Type Hints**: Type annotations for public interfaces.
- [ ] **Error Messages**: Clear, actionable error messages.
- [ ] **Logging Integration**: Use `LOGGER` from `ultralytics.utils` with a `colorstr("Format:")` prefix; do not use bare `print()`.
- [ ] **Unit Tests**: Tests added for the export pipeline and runtime backend (see [`tests/`](https://github.com/ultralytics/ultralytics/tree/main/tests) for examples).
- [ ] **Docstrings**: Google-style docstrings on all public functions and classes (per the [Contributing Guide](../help/contributing.md#google-style-docstrings)).

## Conclusion

This guide enables hardware vendors and community contributors to successfully integrate new export pipelines and runtime software into Ultralytics while preserving the ecosystem's core values of maintainability, ease of use, and minimal dependencies. Following these guidelines results in integrations that enhance the user experience while preserving the reliability and performance of the Ultralytics platform.

For questions or clarification on any aspect of this guide, please reach out through the [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues) page or refer to the broader [Contributing Guide](../help/contributing.md).
