import argparse
import re
import types
from copy import deepcopy
from pathlib import Path

import torch

from ultralytics import RTDETR
from ultralytics.engine.exporter import Exporter


def parse_args():
    p = argparse.ArgumentParser(description="Export RT-DETR-family checkpoints with optional deploy conversion.")
    p.add_argument("weights", type=str, help="Path to RT-DETR-family .pt checkpoint.")
    p.add_argument("--format", type=str, default="onnx", help="Export format.")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset.")
    p.add_argument("--device", default=0, help="CUDA device id or 'cpu'.")
    p.add_argument("--batch", type=int, default=1, help="Batch size.")
    p.add_argument("--half", action="store_true", help="FP16 export.")
    p.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX graph. Upstream DEIMv2 keeps this disabled by default.",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional experiment stem. Output names also include imgsz, precision, and export tags.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to the weights directory.",
    )
    p.add_argument("--workspace", type=float, default=None, help="TensorRT workspace size in GB.")
    p.add_argument(
        "--export-eval-idx",
        type=int,
        default=None,
        help="Override decoder eval_idx only for export. Example: 3 keeps decoder layers 0..3 during export.",
    )
    p.add_argument(
        "--export-num-queries",
        type=int,
        default=None,
        help="Override head num_queries only for export. Example: 100 exports with num_queries=100.",
    )
    p.add_argument(
        "--no-fp32-attn",
        action="store_true",
        help="Disable DINOv3-safe fp32 attention pinning for TRT fp16 builds. "
        "Without this flag, attention MatMul/Softmax and norm internals are "
        "forced to fp32 to avoid overflow (NVIDIA/TensorRT#4723).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable experimental debug TensorRT overrides and add a debug tag to output names.",
    )
    p.add_argument(
        "--opt-level",
        type=int,
        default=3,
        choices=range(0, 6),
        help="TensorRT builder optimization level (0-5). Default 3 = TensorRT's own default. "
        "Lower levels use less aggressive layer fusion, which can avoid FP16 overflow "
        "(NaN) in some graphs (e.g. truncated-decoder exports) at a possible latency cost. "
        "When != 3 an 'opt<N>' tag is added to output names.",
    )
    p.add_argument(
        "--ropefix",
        action="store_true",
        default=False,
        help="Pre-compute DINOv3 RoPE sin/cos as constant buffers (default ON). "
        "Required for onnxsim --simplify to succeed and removes one source of "
        "FP16 instability (trig + division). See DEIMv2 export_onnx.py.",
    )
    p.set_defaults(simplify=False)
    return p.parse_args()


def apply_ropefix(deploy_model, imgsz):
    """Replace DINOv3 RoPE forward with a constant lookup at the eval feature-map size.

    Mirrors upstream DEIMv2 ``--ropefix`` (tools/deployment/export_onnx.py).
    Collapses the per-block rope subgraph (arange/meshgrid/sub/sin/cos/div) into
    two Constants in the exported ONNX, which:
      - lets onnxsim --simplify succeed (otherwise crashes on Sub_1_output_0)
      - removes a known fp16 instability source (trig + division)
      - shrinks the graph TRT has to optimize → reduces Myelin tactic flakiness
    """
    # Path: deploy_model.model[0]  (DEIMDINOv3STAs wrapper)
    #     .m                       (DINOv3STAs adapter)
    #     .dinov3                  (DinoVisionTransformer or Windowed)
    try:
        backbone_wrapper = deploy_model.model[0]
        sta_adapter = backbone_wrapper.m
        dinov3 = sta_adapter.dinov3
        rope = dinov3.rope_embed
    except AttributeError as e:
        print(f"[ropefix] no DINOv3 rope_embed found ({e}) — skipping")
        return

    # Refuse on windowed configs: rope is called with both windowed and global
    # (Hw, Ww) vs (H, W); a single constant table would be wrong for one path.
    from ultralytics.nn.backbones.dinov3 import WindowedDinoVisionTransformer
    if isinstance(dinov3, WindowedDinoVisionTransformer):
        print("[ropefix] WindowedDinoVisionTransformer detected — RoPE is called with "
              "both windowed and global sizes; a single constant table would be wrong. "
              "Skipping ropefix. Use --no-ropefix to silence.")
        return

    patch_size = dinov3.patch_embed.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    H = imgsz // patch_size
    W = imgsz // patch_size
    rope.eval()
    with torch.no_grad():
        sin, cos = rope(H=H, W=W)
    rope.register_buffer("rope_sin_const", sin)
    rope.register_buffer("rope_cos_const", cos)

    def _const_forward(self, *, H=None, W=None):
        return (self.rope_sin_const, self.rope_cos_const)

    rope.forward = types.MethodType(_const_forward, rope)
    print(f"[ropefix] precomputed RoPE at H={H}, W={W} (patch_size={patch_size})")


def build_engine_fp16(
    onnx_path,
    engine_path,
    workspace=None,
    half=True,
    shape=(1, 3, 640, 640),
    fp32_attn=True,
    debug=False,
    opt_level=3,
):
    """Build a TensorRT fp16 engine with DINOv3-safe precision overrides.

    DINOv3 ViT backbones use decomposed self-attention (MatMul → Softmax) in
    TRT 10.x.  The QK^T MatMul output materialises as fp16 and overflows for
    attention logit spreads > ~11 (d_k=64), producing NaN that propagates
    through the entire decoder.  See NVIDIA/TensorRT#4723.

    When ``fp32_attn=True`` (default) this builder pins every Softmax, every
    attention-path MatMul, and every norm-internal Reduce/Pow/Unary/Elementwise
    to fp32 while keeping the rest of the graph in fp16. When ``debug=True``,
    decoder residual Add layers are pinned to fp32 as an experimental override.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    ws = int((workspace or 4) * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)
    # Builder optimization level controls fusion aggressiveness. TRT default is 3;
    # lowering it (e.g. 0-2) yields less aggressive Myelin fusion, which avoids an
    # FP16 attention overflow seen on truncated-decoder (eidx) exports.
    config.builder_optimization_level = opt_level

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("PARSE ERR:", parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    if half and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if half and fp32_attn:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        # Keep the numerically sensitive ops in fp32, but leave attention
        # MatMul in fp16 for this minimal mixed-precision trial.
        norm_re = re.compile(r"/(?:norm\d*|gateway/norm)(?:/|$)")
        attn_re = re.compile(r"/attn/|/self_attn/|/cross_attn/")
        pow_re = re.compile(r"/Pow(?:_|$)", re.IGNORECASE)
        sqrt_re = re.compile(r"/Sqrt(?:_|$)", re.IGNORECASE)
        head_re = re.compile(r"/dec_(score|bbox)_head")
        lqe_re = re.compile(r"/lqe_layers")
        integral_re = re.compile(r"/integral(?:_\d+)?/")
        # Minimal classification-head pin: only the Linear (MatMul + bias Add)
        # of each traced dec_score_head. PyTorch ONNX export only emits the
        # heads that are actually executed, so this hits at most 4 ops total
        # (dec_score_head.0 for pre_scores + dec_score_head.<eval_idx> for the
        # final scores). Targets the class-id flip drift identified by per-tensor
        # probe analysis; bbox-head drift is buffered by the integral softmax.
        score_head_re = re.compile(r"/dec_score_head\.\d+/(MatMul|Add)$")
        # Layer types that actually do fp16 math. Shape/constant/gather/
        # unsqueeze/concat ops carry metadata; pinning them is a no-op at
        # best and can force spurious reformat layers at worst.
        compute_types = {
            trt.LayerType.MATRIX_MULTIPLY,
            trt.LayerType.CONVOLUTION,
            trt.LayerType.ELEMENTWISE,
            trt.LayerType.ACTIVATION,
            trt.LayerType.SOFTMAX,
            trt.LayerType.REDUCE,
            trt.LayerType.UNARY,
            trt.LayerType.NORMALIZATION,
            trt.LayerType.SCALE,
        }
        # All 24 decoder residual Adds (4 layers × 6 Adds)
        RESIDUAL_ADD_RE = re.compile(r"/model\.22/decoder/layers\.\d+/(?:gateway/)?Add(?:_\d+)?$")
        # The .clamp(-65504, 65504) safety valve before each decoder norm3.
        CLIP_RE = re.compile(r"/model\.22/decoder/layers\.\d+/Clip(?:_\d+)?$")

        n_pinned = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name or ""
            pin = False
            if layer.type == trt.LayerType.SOFTMAX:
                pin = True
            if layer.type == trt.LayerType.NORMALIZATION:
                pin = True
            # Uncomment this narrower score-matmul override if T4 still needs
            # extra stabilization without pinning every attention MatMul.
            # elif layer.type == trt.LayerType.MATRIX_MULTIPLY and attn_re.search(name):
            #     pin = True
            elif norm_re.search(name) and layer.type == trt.LayerType.REDUCE:
                pin = True
            elif norm_re.search(name) and layer.type == trt.LayerType.UNARY and sqrt_re.search(name):
                pin = True
            elif norm_re.search(name) and layer.type == trt.LayerType.ELEMENTWISE and pow_re.search(name):
                pin = True
            # elif debug and RESIDUAL_ADD_RE.search(name) and layer.type == trt.LayerType.ELEMENTWISE:
            #     pin = True
            # # Decoder norm3 safety clamp — when fused with surrounding ops by
            # # Myelin, the clamp can be reordered AFTER an fp16 cast and become
            # # ineffective. Pin it to ensure clamp runs in fp32 BEFORE narrowing.
            # elif debug and CLIP_RE.search(name) and layer.type == trt.LayerType.ACTIVATION:
            #     pin = True
            # Priority A: classification path only. Per-tensor probe analysis
            # showed dec_score_head's MatMul + bias Add carries the drift that
            # flips argmax on borderline classes (~0.3 mAP). dec_bbox_head drift
            # is much larger in magnitude but absorbed by integral softmax, so
            # we DO NOT pin it. Score head Linear is at most 4 ops in ONNX
            # (dec_score_head.0 for pre_scores + dec_score_head.<eval_idx> for
            # final scores). 15× cheaper than the broader head_re (62 ops).
            # DISABLED: empirically did not recover the mAP gap from PyTorch fp16.
            # elif debug and score_head_re.search(name) and layer.type in {trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE}:
            #     pin = True
            # Pin the FULL DEIMRMSNorm body (Pow + ReduceMean + Add(eps) + Sqrt +
            # Div + 2× Mul) for exact PyTorch .half() parity. The default rules
            # above only cover Pow/ReduceMean/Sqrt; adding this catches the
            # remaining Add(eps), Div(rsqrt), and the two elementwise Muls
            # (x × inv_rms and × scale) — all cheap ELEMENTWISE ops. Per RMSNorm
            # site: ~7 ops vs 3 with the default rules.
            elif debug and norm_re.search(name) and layer.type in compute_types:
                pin = True
            # Broader head_re (dec_score_head + dec_bbox_head MLPs, 62 ops on XL)
            # — keep available but commented as the minimal score_head_re is
            # the data-driven choice from per-tensor probe analysis.
            # elif debug and head_re.search(name) and layer.type in compute_types:
            #     pin = True
            # Priority B: LQE adds quality scores to classification logits.
            # Drift here shifts class-id rankings on borderline cases.
            # elif debug and lqe_re.search(name) and layer.type in compute_types:
            #     pin = True
            # Priority C: distribution integral (softmax + project) turns
            # reg_max bins into continuous bbox distances. Less impactful than
            # head/LQE since high-conf box drift is already <1e-3.
            # elif debug and integral_re.search(name) and layer.type in compute_types:
            #     pin = True
            if pin:
                layer.precision = trt.float32
                for o in range(layer.num_outputs):
                    layer.set_output_type(o, trt.float32)
                n_pinned += 1
        print(f"DINOv3-safe: pinned {n_pinned} attention/norm layers to fp32")

    print(f"Building {'FP16' if half else 'FP32'} engine → {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed — check logs above")

    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    print(f"Saved {engine_path}")


def build_output_paths(args):
    """Build experiment-specific output paths to avoid clobbering previous exports."""
    weights_path = Path(args.weights)
    outdir = Path(args.outdir) if args.outdir else weights_path.parent
    rope_tag = "rope" if args.ropefix else "norope"
    sim_tag = "sim" if args.simplify else "nosim"
    extra_tags = [f"imgsz{args.imgsz}"]
    if args.export_eval_idx is not None:
        extra_tags.append(f"eidx{args.export_eval_idx}")
    if args.export_num_queries is not None:
        extra_tags.append(f"nq{args.export_num_queries}")
    if args.format == "engine" and args.half:
        extra_tags.append("nofp32attn" if args.no_fp32_attn else "fp32attn")
    if args.debug:
        extra_tags.append("debug")
    if args.format == "engine" and args.opt_level != 3:
        extra_tags.append(f"opt{args.opt_level}")
    suffix = f"_{'_'.join(extra_tags)}" if extra_tags else ""
    base_stem = Path(args.name).stem if args.name else f"{weights_path.stem}_op{args.opset}_{sim_tag}_{rope_tag}"
    base_stem = f"rtdetr_{base_stem}"
    stem = f"{base_stem}{suffix}"

    # Name the intermediate ONNX after the requested precision so artifact sets stay easy to compare.
    onnx_precision = "fp16" if args.half else "fp32"
    engine_precision = "fp16" if args.half else "fp32"

    onnx_path = outdir / f"{stem}_{onnx_precision}.onnx"
    engine_path = outdir / f"{stem}_{engine_precision}.engine"
    return onnx_path, engine_path


def apply_export_eval_idx_override(deploy_model, export_eval_idx):
    """Apply an export-only decoder eval_idx override for RT-DETR-family models."""
    head = deploy_model.model[-1] if hasattr(deploy_model, "model") and len(deploy_model.model) else None
    decoder_modules = []
    if head is not None and hasattr(head, "decoder") and hasattr(head.decoder, "eval_idx") and hasattr(head.decoder, "num_layers"):
        decoder_modules.append(head.decoder)
    else:
        decoder_modules = [m for m in deploy_model.modules() if hasattr(m, "eval_idx") and hasattr(m, "num_layers")]
    if not decoder_modules:
        raise RuntimeError("No RT-DETR-family decoder with eval_idx/num_layers was found in the export model.")

    for decoder in decoder_modules:
        num_layers = decoder.num_layers
        if not 0 <= export_eval_idx < num_layers:
            raise ValueError(f"--export-eval-idx must be in [0, {num_layers - 1}], got {export_eval_idx}.")
        decoder.eval_idx = export_eval_idx

    if head is not None and hasattr(head, "eval_idx"):
        head.eval_idx = export_eval_idx

    return export_eval_idx + 1


def apply_export_num_queries_override(deploy_model, export_num_queries):
    """Apply an export-only num_queries override for RT-DETR-family heads."""
    head = deploy_model.model[-1] if hasattr(deploy_model, "model") and len(deploy_model.model) else None
    if head is None or not hasattr(head, "num_queries"):
        raise RuntimeError("No RT-DETR-family head with num_queries was found in the export model.")
    if export_num_queries <= 0:
        raise ValueError(f"--export-num-queries must be positive, got {export_num_queries}.")

    original_num_queries = head.num_queries
    if export_num_queries > original_num_queries:
        raise ValueError(
            f"--export-num-queries may not exceed the checkpoint's num_queries ({original_num_queries}), "
            f"got {export_num_queries}."
        )

    head.num_queries = export_num_queries

    return export_num_queries


def main():
    args = parse_args()
    onnx_path, engine_path = build_output_paths(args)

    # Use a copied RT-DETR-family model so export-only overrides do not mutate the live wrapper model.
    deploy_model = deepcopy(RTDETR(args.weights).model).eval().float()
    deploy_model.pt_path = str(onnx_path.with_suffix(".pt"))
    for p in deploy_model.parameters():
        p.requires_grad = False
    if args.export_num_queries is not None:
        export_num_queries = apply_export_num_queries_override(deploy_model, args.export_num_queries)
        print(f"Using export-only num_queries={export_num_queries}.")
    if args.export_eval_idx is not None:
        export_layers = apply_export_eval_idx_override(deploy_model, args.export_eval_idx)
        print(f"Using export-only decoder eval_idx={args.export_eval_idx} ({export_layers} decoder layers).")
    # Optional deploy conversion for D-FINE/DEIM decoders. Plain RT-DETR decoders simply skip this step.
    for m in deploy_model.modules():
        if hasattr(m, "convert_to_deploy"):
            m.convert_to_deploy()

    if args.ropefix:
        apply_ropefix(deploy_model, args.imgsz)

    export_format = "onnx" if args.format == "engine" else args.format
    print(
        f"Exporting with format={export_format}, imgsz={args.imgsz}, opset={args.opset}, "
        f"device={args.device}, batch={args.batch}, half={args.half}, "
        f"simplify={args.simplify}, debug={args.debug}"
    )
    print(f"Intermediate ONNX: {onnx_path}")
    if args.format == "engine":
        print(f"Target engine: {engine_path}")
    exporter = Exporter(overrides={
        "format": export_format,
        "imgsz": args.imgsz,
        "opset": args.opset,
        "device": args.device,
        "batch": args.batch,
        "half": args.half if args.format != "engine" else False,
        "simplify": args.simplify,
    })
    artifact = Path(str(exporter(model=deploy_model)))

    if args.format == "engine":
        if artifact.suffix != ".onnx":
            raise RuntimeError(f"Expected ONNX export before TensorRT build, got: {artifact}")
        build_engine_fp16(
            onnx_path=artifact,
            engine_path=engine_path,
            workspace=args.workspace,
            half=args.half,
            shape=(args.batch, 3, args.imgsz, args.imgsz),
            fp32_attn=not args.no_fp32_attn,
            debug=args.debug,
            opt_level=args.opt_level,
        )
        artifact = engine_path

    print(artifact)


if __name__ == "__main__":
    main()
