# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML


def torch2litert(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path,
    half: bool,
    int8: bool,
    end2end: bool,
    calibration_dataset: torch.utils.data.DataLoader,
    metadata: dict,
    prefix: str,
) -> Path:
    """Export a PyTorch model to LiteRT format using litert_torch, with optional FP16/INT8 quantization.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model file path used to derive output directory.
        half (bool): Whether to apply FP16 weight-only quantization.
        int8 (bool): Whether to apply static INT8 quantization (takes precedence over half).
        end2end (bool): Whether the model has built-in NMS (end-to-end detection).
        calibration_dataset (DataLoader | None): Calibration dataloader for INT8 quantization, as returned by
            ``get_int8_calibration_dataloader``. Required when ``int8=True``.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (Path): Path to the exported ``_litert_model`` directory.
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements(("litert-torch>=0.9.0", "ai-edge-litert>=2.1.4"))
    import litert_torch

    LOGGER.info(f"\n{prefix} starting export with litert_torch {litert_torch.__version__}...")
    file = Path(file)
    quant_tag = "_int8" if int8 else "_half" if half else ""
    f = Path(str(file).replace(file.suffix, f"{quant_tag}_litert_model"))
    f.mkdir(parents=True, exist_ok=True)

    # For static INT8 the detection head's torch.cat([decoded_boxes, class_scores], dim=1)
    # creates a mixed-range tensor ([0,640] coords + [0,1] probs) that gets a single
    # quantization scale dominated by box coords, collapsing all class scores to zero and
    # breaking NMS.  Fix: patch the head instance so boxes and scores are kept as separate
    # tensors with independent scales throughout, then rejoin in the backend after dequant.
    _head_patched = False
    if int8:
        import types as _types

        head = model.model[-1]

        if end2end and hasattr(head, "one2one"):
            # End2end: override head.forward to skip the cat+split round-trip in _inference /
            # postprocess and instead apply topk directly on the separately-quantized tensors.
            def _int8_e2e_forward(self, feat_maps):
                one2one = self.forward_head([f.detach() for f in feat_maps], **self.one2one)
                if self.training:
                    return {"one2many": self.forward_head(feat_maps, **self.one2many), "one2one": one2one}
                # Separate tensors — each will get its own int8 scale during calibration
                boxes = self._get_decode_boxes(one2one).permute(0, 2, 1)  # (B, N, 4)  [0, 640]
                scores = one2one["scores"].sigmoid().permute(0, 2, 1)  # (B, N, nc) [0, 1]
                # topk on correctly-scaled scores → right detections selected
                scores_top, conf_top, idx = self.get_topk_index(scores, self.max_det)
                boxes_top = boxes.gather(1, idx.expand(-1, -1, 4))
                return boxes_top, torch.cat([scores_top, conf_top], dim=-1)

            head.forward = _types.MethodType(_int8_e2e_forward, head)
            _head_patched = True
            traced = model
        else:
            # Non-end2end: wrap model to split output (boxes, classes) into separate tensors.
            class _SplitDetectionOutput(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.model = m

                def forward(self, x):
                    y = self.model(x)
                    if isinstance(y, torch.Tensor) and y.ndim == 3:
                        return y[:, :4], y[:, 4:]  # (B, 4, N), (B, nc, N)
                    return y

            traced = _SplitDetectionOutput(model)
    else:
        traced = model

    edge_model = litert_torch.convert(traced, (im,))
    suffix = "int8" if int8 else "float16" if half else "float32"
    tflite_file = f / f"{file.stem}_{suffix}.tflite"
    edge_model.export(tflite_file)

    if _head_patched:
        del head.forward  # Remove instance override, restores class-level method

    if int8 or half:
        check_requirements("ai-edge-quantizer>=0.6.0")
        from ai_edge_quantizer import qtyping, quantizer, recipe, recipe_manager
        from ai_edge_quantizer.algorithm_manager import AlgorithmName

        qt = quantizer.Quantizer(str(tflite_file))
        if int8:
            LOGGER.info(f"{prefix} applying INT8 static quantization...")
            qt.load_quantization_recipe(recipe.static_wi8_ai8())
            calib_samples = []
            for batch in calibration_dataset:
                imgs = batch["img"].cpu().float() / 255.0
                for i in range(imgs.shape[0]):
                    calib_samples.append({"args_0": imgs[i : i + 1].numpy()})
            calibration_result = qt.calibrate({"serving_default": calib_samples})
            qt.quantize(calibration_result=calibration_result).export_model(str(tflite_file), overwrite=True)
        else:
            LOGGER.info(f"{prefix} applying FP16 weight-only quantization...")
            rp = recipe_manager.RecipeManager()
            rp.add_weight_only_config(
                regex=".*",
                operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                num_bits=16,
                algorithm_key=AlgorithmName.FLOAT_CASTING,
            )
            qt.load_quantization_recipe(rp.get_quantization_recipe())
            qt.quantize().export_model(str(tflite_file), overwrite=True)

    YAML.save(f / "metadata.yaml", metadata or {})
    return f
