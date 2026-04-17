---
comments: true
description: Export YOLO models to torch.export ExportedProgram (.pt2) format for PyTorch-native interoperability, custom lowering pipelines, and research workflows.
keywords: Ultralytics, YOLO, ExportedProgram, torch.export, pt2, model export, PyTorch, research, custom lowering, model serialization
---

# Export YOLO Models to ExportedProgram (.pt2) Format

The [torch.export](https://docs.pytorch.org/docs/stable/export.html) ExportedProgram format (`.pt2`) captures a PyTorch model as a self-contained, graph-based intermediate representation (IR). It preserves the full program semantics before any backend lowering, making it the standard entry point for downstream tooling in the PyTorch ecosystem.

This guide explains how to export Ultralytics YOLO models to ExportedProgram format.

## When to Use ExportedProgram

ExportedProgram is useful when you need:

- **Custom lowering pipelines**: Feed the exported program into your own `torch.export`-based compiler or optimizer before targeting a specific backend.
- **Research and analysis**: Inspect the graph IR, run custom passes, or benchmark individual operators.
- **PyTorch ecosystem interop**: Use the standardized `.pt2` artifact with any tool that consumes `torch.export.ExportedProgram`.

!!! tip

    If your goal is on-device mobile/edge deployment, consider [ExecuTorch](executorch.md) instead, which lowers the ExportedProgram further into an optimized `.pte` file.

## Exporting YOLO Models to ExportedProgram

### Installation

ExportedProgram export requires PyTorch 2.9 or higher. No extra dependencies beyond PyTorch are required:

!!! tip "Installation"

    === "CLI"

        ```bash
        pip install ultralytics
        ```

### Usage

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO model
        model = YOLO("yolo11n.pt")

        # Export to ExportedProgram format
        model.export(format="exported_program")  # creates 'yolo11n.pt2'
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=exported_program  # creates 'yolo11n.pt2'
        ```

### Export Arguments

| Argument | Type            | Default | Description                                |
| -------- | --------------- | ------- | ------------------------------------------ |
| `imgsz`  | `int` or `list` | `640`   | Image size for model input (height, width) |
| `batch`  | `int`           | `1`     | Batch size for export                      |
| `device` | `str`           | `'cpu'` | Device to use for export (`'cpu'` or `'cuda'`) |

### Output

The export produces a single `.pt2` file with metadata embedded inside:

```text
yolo11n.pt2              # ExportedProgram with embedded metadata
```

## Using Exported Models

Load the `.pt2` file with standard PyTorch APIs:

```python
import torch

ep = torch.export.load("yolo11n.pt2")

# Inspect the graph
print(ep.graph_module.graph)

# Run inference
output = ep.module()(torch.randn(1, 3, 640, 640))
```

## FAQ

### How does ExportedProgram differ from TorchScript?

TorchScript (`torch.jit.script` / `torch.jit.trace`) is a legacy serialization path. `torch.export` is its successor: it produces a sound, whole-graph capture with stronger guarantees and is the recommended path for new projects.
