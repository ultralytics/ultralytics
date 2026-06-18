---
comments: true
description: Generate beautiful SVG architecture diagrams from Ultralytics YOLO YAML model configs. Visualize Backbone, Neck, and Head modules with 9 themes and dual head display modes.
keywords: Ultralytics, YOLO26, YAML2ModelGraph, model visualization, architecture diagram, SVG, model architecture, Backbone, Neck, Head, visualization tool
---

# Visualize YOLO Model Architectures with YAML2ModelGraph 🎨

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/WangQvQ/YAML2ModelGraph/main/svg/graph_paper.svg" alt="YAML2ModelGraph overview showing YOLO model architecture diagrams in Paper and Dark themes">
</p>

[YAML2ModelGraph](https://github.com/WangQvQ/YAML2ModelGraph) is an open-source tool that converts [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) YAML model configuration files into publication-ready SVG architecture diagrams. It automatically parses the `backbone` and `head` sections, identifies module types, and renders a clear three-lane layout — [Backbone](https://www.ultralytics.com/glossary/backbone), [Neck](https://www.ultralytics.com/glossary/feature-pyramid-network), and [Head](https://www.ultralytics.com/glossary/object-detection) — with color-coded nodes and connection paths.

## Why Visualize Model Architectures?

Understanding a model's structure is essential for research, education, and communication. While textual summaries provide numbers, a visual diagram reveals the **spatial relationships** between layers at a glance:

- **Research papers** — Produce clean, consistent architecture figures without manual drawing
- **Teaching** — Help students understand how feature maps flow through Backbone, Neck, and Head
- **Model comparison** — Place diagrams of different model variants side by side for quick comparison
- **Documentation** — Add architecture visuals to project wikis, slides, and reports

YAML2ModelGraph fills a gap in the YOLO ecosystem: Ultralytics provides textual model summaries via `model_info()`, but does not include a built-in diagram generator. This tool works directly with the standard YAML config format, requiring no model export or inference — just the YAML file.

## Key Features

- **9 built-in themes** — From academic paper style (Paper, Journal, Paper RYB) to modern looks (Candy, Ocean, Dark) and specialized styles (Blueprint, Forest, Retro)
- **Dual head display modes** — Single head (`single`, default) or triple head (`multi`) showing P3/8, P4/16, and P5/32 detection heads as separate nodes
- **Smart layout engine** — Automatic Backbone vertical stacking, Neck multi-column folding, and Head alignment
- **Multiple [connection](https://www.ultralytics.com/glossary/feature-extraction) styles** — Vertical straight lines, Bézier curves, Manhattan routing, and dashed cross-module links
- **Rich node metadata** — Displays channel counts, stride multipliers, and repeat counts on each node
- **Zero heavy dependencies** — Requires only `pyyaml`; pure Python with SVG output

## Installation

!!! tip "Install YAML2ModelGraph"

    ```bash
    git clone https://github.com/WangQvQ/YAML2ModelGraph.git
    cd YAML2ModelGraph
    pip install pyyaml
    ```

YAML2ModelGraph requires Python 3.8 or later and has no GPU or deep learning framework dependencies. It works entirely offline — no model weights or inference needed, only the YAML configuration file.

For detailed Ultralytics installation instructions, see the [Ultralytics Quickstart guide](../quickstart.md).

## Quick Start

Generate a diagram from any YOLO YAML config with a single command:

!!! example "Generate Architecture Diagram"

    === "Single Head (Default)"

        ```bash
        python main.py examples/yolov8.yaml output.svg --theme paper
        ```

    === "Triple Head"

        ```bash
        python main.py examples/yolov8.yaml output.svg --theme paper --head multi
        ```

    === "Dark Theme"

        ```bash
        python main.py examples/yolov8.yaml output.svg --theme dark --head multi
        ```

The tool reads your YAML file, identifies all modules, computes layout positions, and writes a self-contained SVG file that can be opened in any browser or embedded in documents.

## Theme Gallery

YAML2ModelGraph ships with 9 carefully designed themes. Each theme defines colors, gradients, fonts, and type-specific strip colors for module nodes.

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/WangQvQ/YAML2ModelGraph/main/svg/graph_paper_ryb.svg" alt="All 9 YAML2ModelGraph themes: Paper, Candy, Dark, Ocean, Retro, Blueprint, Forest, Paper RYB, and Journal">
</p>

| Theme | Style | Best For |
| :---- | :---- | :------- |
| `paper` | Black & white, Times New Roman | IEEE / CVPR / thesis figures (default) |
| `candy` | Morandi palette, rounded corners | Presentations, blog posts, posters |
| `dark` | Dark background, high contrast | Screen demos, dark mode reading |
| `ocean` | Blue tones, fresh and professional | Business reports, tech whitepapers |
| `retro` | Warm beige (Gruvbox style) | Long reading sessions, retro aesthetics |
| `blueprint` | Deep blue, white fine lines, CAD font | Engineering diagrams, architecture docs |
| `forest` | Green palette, natural feel | Eco/lightweight theme emphasis |
| `paper_ryb` | Red-yellow-blue, low saturation | Clear Backbone/Neck/Head distinction |
| `journal` | Minimalist cool, near-invisible bg | Springer / Nature style figures |

## Head Display Modes

YOLO models typically use a single `Detect` module that receives feature maps from multiple scales (P3, P4, P5). YAML2ModelGraph offers two ways to visualize this:

<p align="center">
  <img width="45%" src="https://raw.githubusercontent.com/WangQvQ/YAML2ModelGraph/main/svg/graph_paper.svg" alt="Single head mode showing one Detect node">
  &nbsp;&nbsp;
  <img width="45%" src="https://raw.githubusercontent.com/WangQvQ/YAML2ModelGraph/main/svg/multi/graph_paper.svg" alt="Triple head mode showing three separate Detect nodes for P3, P4, and P5">
</p>

| Mode | Parameter | Description |
| :--- | :-------- | :---------- |
| **Single** | `--head single` | One `Detect` node — compact and simple (default) |
| **Triple** | `--head multi` | Three nodes: `Detect (P3/8)`, `Detect (P4/16)`, `Detect (P5/32)` — shows the multi-scale detection structure explicitly |

In triple head mode, each detection head connects to its corresponding feature map source, and the nodes are stacked from the bottom of the diagram, aligned with the lowest Backbone/Neck module.

## Supported Models

YAML2ModelGraph works with any model that follows the standard Ultralytics YAML configuration format with `backbone:` and `head:` sections. Tested models include:

| Model | YAML | Special Modules |
| :---- | :--- | :-------------- |
| [YOLO26](../models/yolo26.md) | `yolo26.yaml` | Conv, C3k2, SPPF, C2PSA, Concat, Upsample, Detect |
| [YOLO11](../models/yolo11.md) | `yolo11.yaml` | Conv, C3k2, SPPF, C2PSA, Concat, Upsample, Detect |
| [YOLO12](../models/yolo12.md) | `yolo12.yaml` | Conv, C3k2, A2C2f, Concat, Upsample, Detect |
| [YOLOv8](../models/yolov8.md) | `yolov8.yaml` | Conv, C2f, SPPF, Concat, Upsample, Detect |
| [YOLOv9](../models/yolov9.md) | `yolov9s.yaml` | Conv, ELAN1, AConv, RepNCSPELAN4, SPPELAN, Detect |

The tool recognizes module types including `Conv`, `C2f`, `C3k2`, `Concat`, `Upsample`, `Detect`, `SPPF`, `C2PSA`, `A2C2f`, `ELAN1`, `AConv`, `RepNCSPELAN4`, `SPPELAN`, and more. Unrecognized modules are rendered with a default color strip.

## Configuration Options

### Command-Line Arguments

| Argument | Required | Default | Description |
| :------- | :------- | :------ | :---------- |
| `model.yaml` | Yes | — | Input YAML model configuration file |
| `output.svg` | No | `yolo_graph.svg` | Output SVG file path |
| `--theme` | No | `paper` | Theme name (see [Theme Gallery](#theme-gallery)) |
| `--head` | No | `single` | Head display mode: `single` or `multi` |

### Node Information Display

Control which metadata appears on each node by editing the `DISPLAY_CONFIG` dictionary in `main.py`:

```python
DISPLAY_CONFIG = {
    "show_channels": True,  # Channel counts (e.g., 64->128 or 128c)
    "show_repeats":  True,  # Repeat count (e.g., n=3)
    "show_stride":   True,  # Stride multiplier (e.g., /32x)
    "show_args":     False, # Detailed args (e.g., a:3,2) — disable if text overflows
}
```

Set `show_args` to `True` for detailed architecture analysis, or leave it `False` for cleaner diagrams suitable for papers and presentations.

## How It Works

YAML2ModelGraph follows a three-phase pipeline:

1. **Parse** — Reads the YAML file, concatenates `backbone` and `head` sequences, extracts module type, source indices, channel counts, and stride for each layer. Channel tracking handles special cases like `Concat` (sums input channels) and `Upsample` (channel passthrough).

2. **Layout** — Assigns each module to a swim lane (Backbone = column 0, Neck = column 1, Head = column 2). Backbone nodes stack vertically; Neck nodes fold into multiple columns when they exceed the Backbone height; Head nodes align to their source positions.

3. **Render** — Builds an SVG with gradient-filled rectangles, colored type strips, text labels, and connection paths. Four routing strategies (`vertical_straight`, `manhattan`, `detour_right`, `standard`) handle different spatial relationships between connected nodes.

## Summary

YAML2ModelGraph provides a lightweight, dependency-free way to generate publication-quality architecture diagrams from Ultralytics YOLO YAML configs. With 9 themes, dual head display modes, and automatic layout, it serves researchers, educators, and engineers who need clear visual documentation of model architectures.

For more details, visit the [YAML2ModelGraph GitHub repository](https://github.com/WangQvQ/YAML2ModelGraph). For other Ultralytics integrations, see the [Integrations overview](./index.md).

## FAQ

### Can I use YAML2ModelGraph with custom YOLO models?

Yes. Any model that follows the standard Ultralytics YAML format with `backbone:` and `head:` sections works. Simply point the tool at your custom YAML file. Unrecognized module types will be rendered with a default color strip labeled with the raw module name.

### How do I add a new theme?

Define a new theme dictionary in `themes.py` with `colors`, `gradients`, `type_colors`, and `font` keys, then register it in the `THEMES` dictionary. See the existing themes for reference — each is a self-contained dictionary of approximately 20 lines.

### Does it support YOLO segmentation or pose models?

YAML2ModelGraph focuses on the model architecture defined in YAML configs. It visualizes the `Detect` head and its inputs. Segmentation and pose models that share the same YAML backbone/head structure will render correctly, though task-specific head details (e.g., keypoint counts) are not displayed.

### Why are some module nodes gray?

Modules like `C3k2`, `A2C2f`, `SPPF`, and `C2PSA` fall through to the default `"Other"` color in the `type_colors` theme dictionary. To give them distinct colors, add entries for these module names in your theme definition in `themes.py`.

### Can I embed the SVG in LaTeX or PowerPoint?

Yes. SVG files can be inserted directly into PowerPoint and modern LaTeX workflows (using the `svg` package with Inkscape conversion, or the `includesvg` command). For older LaTeX setups, convert the SVG to PDF first using a tool like Inkscape or `rsvg-convert`.
