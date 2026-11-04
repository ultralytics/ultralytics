# YOLO Architecture Explorer

An interactive, single-file HTML explorer for the Ultralytics detection configs. It is generated from the code, not drawn by hand. Nothing in it is specific to one YOLO version: configs, blocks, block roles and train/deploy differences are all discovered from the models, so a new version under `ultralytics/cfg/models/` shows up on the next build.

- **Discovery.** Every config under `ultralytics/cfg/models/*/` that Ultralytics' `guess_model_task` reports as detection (or as unknown, so new head types are still tried) is built at every scale. Configs that cannot run on an image alone, such as text-prompted heads, are listed with their error instead of a diagram.
- **Two modes: Training and Deployment.** Training is `model.train()`. Deployment is `model.fuse()` + `eval()`, which is what `predict()` and `export()` run. The build compares the modules that actually run in each mode, plus helper calls and tensor ops, and reports what is removed, replaced or added, with indices generalized to `*`. A branch that stays in the model but is not called at deployment counts as removed. Examples: `model.*.cv1.bn` (BatchNorm folded), `…cv1.conv1 + …cv1.conv2 → …cv1.conv` (re-parameterised branches), `model.*.one2one_cv2` (a head branch unused at deployment), and helper calls or ops that run in only one mode. Δ marks layers whose wiring changes; inside cards, blocks that exist in only one mode are outlined.
- **One diagram per config file.** A compact overview strip (backbone pyramid, neck chips, outputs) sits above the full diagram. In the full diagram, rows are grouped by resolution, so neck nodes line up with the backbone stage they connect to. Parallel backbones and link layers get their own columns. Shapes are written with the scale constants (`80×80×512×w×(1+r)`, `n=3×d`), and a scale table gives `d`, `w`, `max_channels`, `r`, params and GFLOPs for each mode.
- **Click to expand, recursively.** Clicking a block opens its real dataflow as a card beside the diagram, and clicking inside a card goes one level deeper, down to `Conv2d → BatchNorm2d → SiLU`. Repeats collapse to one `×n` node, and shapes are block-relative (`h×w×0.5c_out`, `0.5(n+2)c_out`). Scale-dependent wirings get a variant switch. Helper-method calls (e.g. `forward_head()`) and mixed `nn.Sequential` branches fold into one node each, and pure reshape ops are hidden unless you untick **simple**. Hover any block or line to see numbers. The mouse wheel scrolls, and Ctrl/⌘ + wheel zooms.
- **Blocks & evolution.** A block × version matrix shows where each block type first appears. Per-version summaries list new block types and new wirings of existing blocks, both found from structure keys. Each block has its docstring, its `__init__`/`forward` source, and every distinct internal structure it has. Block colors (conv, CSP, pooling, attention, downsampling, head, …) are also derived from structure.

## What it shows across versions

Everything below comes from the build, not from hand-written notes, so it stays correct as configs change:

- **New blocks per version.** YOLOv9 introduces RepNCSPELAN4, ADown and SPPELAN; YOLOv10 adds SCDown, C2fCIB, PSA and RepVGGDW; YOLO11 adds C3k2 and C2PSA; YOLO12 adds A2C2f area attention. The **Blocks & evolution** tab opens each block's docstring, source and internal structure, and a matrix shows which versions use it.
- **New wiring of existing blocks.** When a version reuses a block with a different internal structure (for example YOLO26's C3k2, SPPF and Detect), the version summary lists it.
- **Training vs deployment cost.** The scale table gives parameters and GFLOPs for both modes. For example YOLOv10n goes from 2.78 M parameters and 8.6 GFLOPs in training to 2.30 M and 6.7 GFLOPs deployed, and YOLO26n from 2.57 M / 6.0 to 2.41 M / 5.4, because the head's training-only `one2one` branch stops running and BatchNorm is folded.
- **Re-parameterisation.** RepConv and RepVGGDW blocks (YOLOv9, YOLOv10, RT-DETR) train with parallel branches; switch to Deployment and open the block to see them merged into one conv.

## Build

Requires PyTorch 2.3 or newer (`TorchFunctionMode` and pytree key paths), which the tracer checks on import.

```bash
python3 build_explorer.py                 # all detection configs -> yolo_architecture_explorer.html
python3 build_explorer.py --only 'yolo11' # regex on config stems for a quick run
```

The script records one 640×640 forward pass per scale and mode with `trace_graph.py`, which uses global module hooks plus a `TorchFunctionMode` to capture every module call and tensor op with its real shapes. A full run takes about three minutes on CPU. Open the HTML file in a browser; block internals are laid out with [dagre](https://github.com/dagrejs/dagre), which loads from cdnjs.

Deep links: `#yolov9c@2.0/5` opens layer 2's card and the block at node 5 inside it, `#yolov10n~deploy@23.0` does the same in Deployment mode, `#yolov9c/2.0` opens the full-screen view, and `#evo=A2C2f` opens a block in the catalog.
