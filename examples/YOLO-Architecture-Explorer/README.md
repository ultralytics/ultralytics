# YOLO Architecture Explorer

An interactive, single-file HTML explorer for the Ultralytics detection configs. It is generated from the code, not drawn by hand. Nothing in it is specific to one YOLO version: configs, blocks, block roles and train/deploy differences are all discovered from the models, so a new version under `ultralytics/cfg/models/` shows up on the next build.

- **Discovery.** Every config under `ultralytics/cfg/models/*/` that Ultralytics' `guess_model_task` reports as detection (or as unknown, so new head types are still tried) is built at every scale. Configs that cannot run on an image alone, such as text-prompted heads, are listed with their error instead of a diagram.
- **Two modes: Training and Deployment.** Training is `model.train()`. Deployment is `model.fuse()` + `eval()`, which is what `predict()` and `export()` run. The build compares the modules that actually run in each mode, plus helper calls and tensor ops, and reports what is removed, replaced or added, with indices generalised to `*`. A branch that stays in the model but is not called at deployment counts as removed. Examples: `model.*.cv1.bn` (BatchNorm folded), `…cv1.conv1 + …cv1.conv2 → …cv1.conv` (re-parameterised branches), `model.*.one2one_cv2` (a head branch unused at deployment), and helper calls or ops that run in only one mode. Δ marks layers whose wiring changes; inside cards, blocks that exist in only one mode are outlined.
- **One diagram per config file.** A compact overview strip (backbone pyramid, neck chips, outputs) sits above the full diagram. In the full diagram, rows are grouped by resolution, so neck nodes line up with the backbone stage they connect to. Parallel backbones and link layers get their own columns. Shapes are written with the scale constants (`80×80×512×w×(1+r)`, `n=3×d`), and a scale table gives `d`, `w`, `max_channels`, `r`, params and GFLOPs for each mode.
- **Click to expand, recursively.** Clicking a block opens its real dataflow as a card beside the diagram, and clicking inside a card goes one level deeper, down to `Conv2d → BatchNorm2d → SiLU`. Repeats collapse to one `×n` node, and shapes are block-relative (`h×w×0.5c_out`, `0.5(n+2)c_out`). Scale-dependent wirings get a variant switch. Helper-method calls (e.g. `forward_head()`) and mixed `nn.Sequential` branches fold into one node each, and pure reshape ops are hidden unless you untick **simple**. Hover any block or line to see numbers. The mouse wheel scrolls, and Ctrl/⌘ + wheel zooms.
- **Blocks & evolution.** A block × version matrix shows where each block type first appears. Per-version summaries list new block types and new wirings of existing blocks, both found from structure keys. Each block has its docstring, its `__init__`/`forward` source, and every distinct internal structure it has. Block colours (conv, CSP, pooling, attention, downsampling, head, …) are also derived from structure.

## Build

```bash
python3 build_explorer.py                  # all detection configs -> yolo_architecture_explorer.html
python3 build_explorer.py --only 'yolo11'  # regex on config stems for a quick run
```

The script records one 640×640 forward pass per scale and mode with `trace_graph.py`, which uses global module hooks plus a `TorchFunctionMode` to capture every module call and tensor op with its real shapes. A full run takes about three minutes on CPU. Open the HTML file in a browser; block internals are laid out with [dagre](https://github.com/dagrejs/dagre), which loads from cdnjs.

Deep links: `#yolov9c@2.0/5` opens layer 2's card and the block at node 5 inside it, `#yolov10n~deploy@23.0` does the same in Deployment mode, `#yolov9c/2.0` opens the full-screen view, and `#evo=A2C2f` opens a block in the catalog.
