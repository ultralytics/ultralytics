---
title: YAML2ModelGraph Architecture Visualization for YOLO Models
comments: true
description: Generate SVG architecture diagrams from Ultralytics YOLO model YAML files with YAML2ModelGraph.
keywords: YAML2ModelGraph, YOLO, model visualization, architecture diagram, Ultralytics, SVG
---

# Visualize YOLO Architectures with YAML2ModelGraph

[YAML2ModelGraph](https://github.com/WangQvQ/YAML2ModelGraph) is a community tool that converts
[Ultralytics YOLO](../models/index.md) model YAML files into SVG architecture diagrams. It can help document custom
models or inspect their backbone, neck, and head structure.

<p align="center">
  <img width="48%" src="https://raw.githubusercontent.com/WangQvQ/YAML2ModelGraph/main/svg/graph_paper.svg" alt="Single-head YOLO architecture rendered by YAML2ModelGraph">
  <img width="48%" src="https://raw.githubusercontent.com/WangQvQ/YAML2ModelGraph/main/svg/multi/graph_paper.svg" alt="Multi-head YOLO architecture rendered by YAML2ModelGraph">
</p>

## Installation

Clone the project and install its PyYAML dependency:

```bash
git clone https://github.com/WangQvQ/YAML2ModelGraph.git
cd YAML2ModelGraph
pip install pyyaml
```

## Usage

Generate a diagram from the included YOLO26 example:

```bash
python main.py examples/yolo26.yaml output.svg
```

Use `--head multi` to draw separate scale-specific head nodes, or select one of the included themes:

```bash
python main.py examples/yolo26.yaml output.svg --head multi --theme paper_ryb
```

Available themes include `paper`, `paper_ryb`, `candy`, `dark`, `ocean`, `retro`, `blueprint`, `forest`, and
`journal`. The generated SVG requires no Graphviz installation.

## Node Metadata

Edit `DISPLAY_CONFIG` in YAML2ModelGraph's `main.py` to choose which details appear in each node:

```python
DISPLAY_CONFIG = {
    "show_channels": True,
    "show_repeats": True,
    "show_stride": True,
    "show_args": False,
}
```

Custom YAML module names are rendered automatically. Theme colors, shapes, and typography are configured in the
tool's `themes.py`.

## Limitations

YAML2ModelGraph specially aligns standard `Detect` heads. Other task heads such as `OBB`, `Pose`, and `Segment` are
currently rendered as generic neck-lane nodes. See the
[YAML2ModelGraph repository](https://github.com/WangQvQ/YAML2ModelGraph) for current support and usage details.
