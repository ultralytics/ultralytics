# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""SAM3 multi-file inference backend for ONNX and TensorRT.

Supports 3-module architecture: vision encoder, text encoder, decoder.
I/O matches the export wrappers in ultralytics/utils/export/sam3_onnx.py:
  - Vision encoder: images → fpn_feat_0/1/2, fpn_pos_2
  - Text encoder:   tokens → text_features [B,32,256], text_mask [B,32]
  - Decoder:        fpn features + prompt_features + prompt_mask → predictions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class _BackboneProxy:
    """Proxy so ``predictor.model.backbone.forward_image(im)`` works."""

    def __init__(self, backend: "SAM3Backend"):
        self._backend = backend

    def forward_image(self, im: torch.Tensor) -> dict:
        """Run vision encoder. Returns dict with ``backbone_fpn`` and ``vision_pos_enc``."""
        return self._backend.forward_image(im)


class SAM3Backend:
    """Multi-file inference backend for SAM3 ONNX and TensorRT models.

    Manages three core model files (vision encoder, text encoder, decoder) plus
    two optional point-prompt modules (prompt encoder, mask decoder). Presents
    a unified interface compatible with ``SAM3SemanticPredictor``.

    Attributes:
        names (list[str]): Current class names.
        task (str): Always ``"segment"``.
        stride (int): 14 (ViT patch size).
        text_embeddings (dict): Cached text encoder outputs.
        backbone (_BackboneProxy): Proxy with ``forward_image``.
        has_point_modules (bool): Whether prompt encoder + mask decoder are available.
    """

    _FILE_STEMS = ("sam3_vision_encoder", "sam3_text_encoder", "sam3_decoder")
    _POINT_STEMS = ("sam3_prompt_encoder", "sam3_mask_decoder")

    def __init__(self, model_dir: str | Path, device: torch.device | str = "cpu", fp16: bool = False):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.fp16 = fp16
        self.names: list[str] = []
        self.task = "segment"
        self.stride = 14
        self.text_embeddings: dict = {}
        self.backbone = _BackboneProxy(self)

        self.has_point_modules = False

        self._model_dir = Path(model_dir)
        assert self._model_dir.is_dir(), f"Model directory not found: {self._model_dir}"

        self._format = self._detect_format()
        LOGGER.info(f"SAM3Backend: detected {self._format.upper()} in {self._model_dir}")
        self._load_models()

    # ------------------------------------------------------------------
    # Format detection & loading
    # ------------------------------------------------------------------

    def _detect_format(self) -> str:
        onnx_files = list(self._model_dir.glob("*.onnx"))
        engine_files = list(self._model_dir.glob("*.engine"))
        if len(onnx_files) >= 3:
            return "onnx"
        if len(engine_files) >= 3:
            return "engine"
        raise FileNotFoundError(
            f"Need 3 model files (.onnx or .engine) in {self._model_dir}. "
            f"Found {len(onnx_files)} ONNX, {len(engine_files)} engine."
        )

    def _has_point_files(self, ext: str) -> bool:
        """Check if both point prompt module files exist."""
        return all((self._model_dir / f"{s}.{ext}").exists() for s in self._POINT_STEMS)

    def _load_models(self) -> None:
        if self._format == "onnx":
            self._load_onnx()
        else:
            self._load_tensorrt()

    # ---- ONNX --------------------------------------------------------

    def _load_onnx(self) -> None:
        cuda = self.device.type != "cpu" and torch.cuda.is_available()
        check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime",))
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        providers = self._ort_providers(cuda)
        LOGGER.info(f"SAM3Backend ONNX: using {providers[0] if isinstance(providers[0], str) else providers[0][0]}")

        paths = {s: self._model_dir / f"{s}.onnx" for s in self._FILE_STEMS}
        for s, p in paths.items():
            assert p.exists(), f"Missing: {p}"

        so = ort.SessionOptions()
        so.log_severity_level = 3

        self._vis_session = ort.InferenceSession(str(paths[self._FILE_STEMS[0]]), sess_options=so, providers=providers)
        self._txt_session = ort.InferenceSession(str(paths[self._FILE_STEMS[1]]), sess_options=so, providers=providers)
        self._dec_session = ort.InferenceSession(str(paths[self._FILE_STEMS[2]]), sess_options=so, providers=providers)
        self._sessions = dict(zip(self._FILE_STEMS, (self._vis_session, self._txt_session, self._dec_session)))

        if self._has_point_files("onnx"):
            pe_path = self._model_dir / f"{self._POINT_STEMS[0]}.onnx"
            md_path = self._model_dir / f"{self._POINT_STEMS[1]}.onnx"
            self._pe_session = ort.InferenceSession(str(pe_path), sess_options=so, providers=providers)
            self._md_session = ort.InferenceSession(str(md_path), sess_options=so, providers=providers)
            self._sessions.update(zip(self._POINT_STEMS, (self._pe_session, self._md_session)))
            self.has_point_modules = True
            LOGGER.info("SAM3Backend ONNX: loaded vision encoder, text encoder, decoder, prompt encoder, mask decoder")
        else:
            LOGGER.info("SAM3Backend ONNX: loaded vision encoder, text encoder, decoder")

    @staticmethod
    def _ort_providers(cuda: bool) -> list:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if cuda and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # ---- TensorRT ----------------------------------------------------

    def _load_tensorrt(self) -> None:
        check_requirements(("tensorrt",))
        import tensorrt as trt

        if self.device.type == "cpu":
            LOGGER.warning("SAM3Backend TRT: CPU requested but TRT requires CUDA, using cuda:0")
            self.device = torch.device("cuda:0")

        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")

        paths = {s: self._model_dir / f"{s}.engine" for s in self._FILE_STEMS}
        for s, p in paths.items():
            assert p.exists(), f"Missing: {p}"

        self._trt_contexts: dict = {}
        self._trt_engines: dict = {}
        self._trt_io_dtypes: dict[str, dict[str, torch.dtype]] = {}
        self._cuda_stream = torch.cuda.Stream(device=self.device)

        _np2torch = {
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.bool_: torch.bool,
        }

        all_stems = list(self._FILE_STEMS)
        if self._has_point_files("engine"):
            point_paths = {s: self._model_dir / f"{s}.engine" for s in self._POINT_STEMS}
            paths.update(point_paths)
            all_stems.extend(self._POINT_STEMS)
            self.has_point_modules = True

        for stem in all_stems:
            with open(paths[stem], "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")
                    _ = json.loads(f.read(meta_len).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    f.seek(0)
                engine = runtime.deserialize_cuda_engine(f.read())

            ctx = engine.create_execution_context()
            # Only per-tensor dtypes are needed at load time; _run_trt sets shapes and
            # allocates all output buffers at runtime (outputs can be dynamic).
            io_dt = {
                engine.get_tensor_name(i): _np2torch.get(
                    trt.nptype(engine.get_tensor_dtype(engine.get_tensor_name(i))), torch.float32
                )
                for i in range(engine.num_io_tensors)
            }

            self._trt_engines[stem] = engine
            self._trt_contexts[stem] = ctx
            self._trt_io_dtypes[stem] = io_dt

        modules = "vision encoder, text encoder, decoder"
        if self.has_point_modules:
            modules += ", prompt encoder, mask decoder"
        LOGGER.info(f"SAM3Backend TRT: loaded {modules}")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_onnx(self, session, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run ONNX session. Filters feed to match model inputs, casts dtypes. Returns name→array dict."""
        _type_map = {
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
        }
        filtered = {}
        for inp in session.get_inputs():
            if inp.name in feed:
                v = feed[inp.name]
                dt = _type_map.get(inp.type)
                if dt is not None and hasattr(v, "dtype") and v.dtype != dt:
                    v = v.astype(dt)
                filtered[inp.name] = v

        raw = session.run(None, filtered)
        out_names = [o.name for o in session.get_outputs()]
        return dict(zip(out_names, raw))

    def _run_trt(self, stem: str, feed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run TRT engine. Auto-casts input dtypes, sets dynamic input shapes. Returns name->tensor dict."""
        import tensorrt as trt

        ctx = self._trt_contexts[stem]
        io_dt = self._trt_io_dtypes[stem]
        engine = self._trt_engines[stem]

        tensors = []  # keep references alive during async execution
        for name, tensor in feed.items():
            if name in io_dt:
                if tensor.dtype != io_dt[name]:
                    tensor = tensor.to(io_dt[name])
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                # Set runtime shape for inputs (needed for dynamic axes)
                ctx.set_input_shape(name, tuple(tensor.shape))
                ctx.set_tensor_address(name, tensor.data_ptr())
                tensors.append(tensor)

        # (Re)allocate output buffers based on current shapes (handles dynamic outputs).
        out_bufs = {}
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tname) == trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(tname))
                buf = torch.empty(shape, dtype=io_dt[tname], device=self.device)
                out_bufs[tname] = buf
                ctx.set_tensor_address(tname, buf.data_ptr())

        ctx.execute_async_v3(self._cuda_stream.cuda_stream)
        self._cuda_stream.synchronize()
        return out_bufs

    def _run(self, stem: str, feed: dict) -> dict[str, torch.Tensor]:
        """Run module ``stem`` on either backend. Accepts tensors or arrays (both runners
        re-cast dtypes) and always returns torch tensors on ``self.device``."""
        if self._format == "onnx":
            np_feed = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in feed.items()}
            out = self._run_onnx(self._sessions[stem], np_feed)
            return {k: torch.from_numpy(v).to(self.device) for k, v in out.items()}
        cuda_feed = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v)).to(self.device))
            for k, v in feed.items()
        }
        return self._run_trt(stem, cuda_feed)

    # ------------------------------------------------------------------
    # Vision encoder
    # ------------------------------------------------------------------

    def forward_image(self, im: torch.Tensor) -> dict:
        """Run vision encoder.

        Args:
            im: [1, 3, H, W] normalized image tensor.

        Returns:
            dict with ``backbone_fpn`` and ``vision_pos_enc`` for the DETR decoder,
            plus ``_fpn_feat_*`` for decoder feed and ``_sam2_feat_*`` for point prompts
            (if the dual SAM2 neck was exported).
        """
        out = self._run(self._FILE_STEMS[0], {"images": im})
        fpn0, fpn1, fpn2, pos2 = out["fpn_feat_0"], out["fpn_feat_1"], out["fpn_feat_2"], out["fpn_pos_2"]

        result = {
            "backbone_fpn": [fpn0, fpn1, fpn2],
            "vision_pos_enc": [pos2, pos2, pos2],
            "_fpn_feat_0": fpn0,
            "_fpn_feat_1": fpn1,
            "_fpn_feat_2": fpn2,
            "_fpn_pos_2": pos2,
        }

        # SAM2 neck features for point prompts (separate learned weights, exported together)
        if "sam2_feat_0" in out:
            result["_sam2_feat_0"] = out["sam2_feat_0"]
            result["_sam2_feat_1"] = out["sam2_feat_1"]
            result["_sam2_feat_2"] = out["sam2_feat_2"]

        return result

    # ------------------------------------------------------------------
    # Text encoder
    # ------------------------------------------------------------------

    def forward_text(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run text encoder on pre-tokenized text.

        Args:
            tokens: [N, 32] int64 token array.

        Returns:
            (text_features [N, 32, 256], text_mask [N, 32] bool).
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        out = self._run(self._FILE_STEMS[1], {"tokens": tokens.astype(np.int64)})
        return out["text_features"].cpu().numpy(), out["text_mask"].cpu().numpy()

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _run_decoder(
        self,
        img_out: dict,
        prompt_features: np.ndarray,
        prompt_mask: np.ndarray,
        input_boxes: np.ndarray | None = None,
        input_boxes_labels: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run decoder with FPN features + prompt + optional box prompts.

        Args:
            img_out: Dict from forward_image (with _fpn_feat_* and _fpn_pos_2 cached).
            prompt_features: [seq, B, 256] text/prompt features (sequence-first).
            prompt_mask: [B, seq] bool mask (True=valid token).
            input_boxes: [B, num_boxes, 4] normalized CxCyWH, or None for text-only.
            input_boxes_labels: [B, num_boxes] int32 (1=pos, 0=neg, -10=ignore).

        Returns:
            dict with pred_logits, pred_boxes, pred_masks, presence_logits.
        """
        bs = prompt_mask.shape[0] if hasattr(prompt_mask, "shape") else 1
        if input_boxes is None:
            # Text-only: dummy single-box with label=-10 (ignored by geometry encoder)
            input_boxes = np.zeros((bs, 1, 4), dtype=np.float32)
            input_boxes_labels = np.full((bs, 1), -10, dtype=np.int32)

        return self._run(
            self._FILE_STEMS[2],
            {
                "fpn_feat_0": img_out["_fpn_feat_0"],
                "fpn_feat_1": img_out["_fpn_feat_1"],
                "fpn_feat_2": img_out["_fpn_feat_2"],
                "fpn_pos_2": img_out["_fpn_pos_2"],
                "prompt_features": prompt_features,
                "prompt_mask": prompt_mask,
                "input_boxes": input_boxes,
                "input_boxes_labels": input_boxes_labels,
            },
        )

    # ------------------------------------------------------------------
    # Point prompt inference (prompt encoder + mask decoder)
    # ------------------------------------------------------------------

    def forward_points(
        self,
        img_out: dict,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run point-based segmentation: prompt encoder + mask decoder.

        Uses SAM2 neck features (``_sam2_feat_*``) when available, falling back
        to SAM3 FPN features (``_fpn_feat_*``). The SAM2 neck has separate
        learned weights matched to the mask decoder.

        Args:
            img_out: Dict from forward_image (with _sam2_feat_* or _fpn_feat_* cached).
            point_coords: [B, N, 2] float32 — point coordinates in pixel space.
            point_labels: [B, N] int32 — 1=foreground, 0=background.

        Returns:
            (masks, iou_scores): masks [B, num_masks, H, W], scores [B, num_masks].
        """
        assert self.has_point_modules, "Point prompt modules not found. Re-export with prompt_encoder + mask_decoder."

        # The point-prompt mask decoder REQUIRES the SAM2 neck features (separate
        # learned weights). The SAM3 FPN features are a different feature space
        # (cosine ~0.42) and produce scattered, broken masks. Falling back to them
        # is only a last resort for old exports — warn loudly so it isn't silent.
        # forward_image always sets the three sam2 keys together, so one check suffices.
        has_sam2 = "_sam2_feat_0" in img_out
        if not has_sam2:
            LOGGER.warning(
                "SAM3Backend: point prompt requested but the vision encoder has no sam2_feat_* outputs. "
                "Falling back to SAM3 FPN features — masks will be WRONG. "
                "Re-export the vision encoder with the SAM2 neck (sam2_feat_0/1/2)."
            )
        prefix = "_sam2_feat_" if has_sam2 else "_fpn_feat_"

        pe_out = self._run(self._POINT_STEMS[0], {"point_coords": point_coords, "point_labels": point_labels})
        md_out = self._run(
            self._POINT_STEMS[1],
            {
                "image_embeddings": img_out[f"{prefix}2"],
                "image_pe": pe_out["dense_pe"],
                "sparse_prompt_embeddings": pe_out["sparse_embeddings"],
                "dense_prompt_embeddings": pe_out["dense_embeddings"],
                "high_res_feat_0": img_out[f"{prefix}0"],
                "high_res_feat_1": img_out[f"{prefix}1"],
            },
        )
        return md_out["masks"], md_out["iou_scores"]

    # ------------------------------------------------------------------
    # set_classes
    # ------------------------------------------------------------------

    def set_classes(self, text: list[str]) -> None:
        """Tokenize text, run text encoder per-class, cache results."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        sot = tokenizer.encoder["<|startoftext|>"]
        eot = tokenizer.encoder["<|endoftext|>"]

        tokens_list = []
        for t in text:
            enc = [sot] + tokenizer.encode(t) + [eot]
            enc = enc[:32] if len(enc) > 32 else enc + [0] * (32 - len(enc))
            tokens_list.append(enc)

        tokens = np.array(tokens_list, dtype=np.int64)

        # Run per-class (static batch=1)
        # Text encoder outputs: text_features [32, 1, 256] (seq-first), text_mask [1, 32]
        all_feats, all_masks = [], []
        for i in range(len(tokens)):
            feats, mask = self.forward_text(tokens[[i]])
            # Ensure sequence-first [32, 1, 256] for decoder compatibility
            if feats.ndim == 3 and feats.shape[0] == 1 and feats.shape[1] == 32:
                feats = np.transpose(feats, (1, 0, 2))
            all_feats.append(feats)
            all_masks.append(mask)

        self.text_embeddings = {
            "text_features": np.concatenate(all_feats, axis=1),  # [32, N, 256] (concat along batch dim)
            "text_mask": np.concatenate(all_masks, axis=0),  # [N, 32]
        }
        self.names = text

    # ------------------------------------------------------------------
    # forward_grounding (main entry point)
    # ------------------------------------------------------------------

    def forward_grounding(
        self, backbone_out: dict, text_ids: torch.Tensor, geometric_prompt=None
    ) -> dict[str, torch.Tensor]:
        """Run grounding: select cached text features + optional box prompts -> run decoder.

        Args:
            backbone_out: Dict from forward_image.
            text_ids: [nc] class indices into cached text_embeddings.
            geometric_prompt: Optional Prompt object with box_embeddings and box_labels.

        Returns:
            dict with pred_logits, pred_boxes, pred_masks, presence_logits.
        """
        assert self.text_embeddings, "Call set_classes() first"

        ids = text_ids.cpu().numpy() if isinstance(text_ids, torch.Tensor) else np.asarray(text_ids)
        nc = len(ids)

        feats_all = self.text_embeddings["text_features"]  # [32, N_total, 256] (seq-first)
        masks_all = self.text_embeddings["text_mask"]  # [N_total, 32]

        # Extract box prompts from geometric_prompt if present
        boxes_per_call = None
        labels_per_call = None
        if geometric_prompt is not None and getattr(geometric_prompt, "box_embeddings", None) is not None:
            # box_embeddings: (N, B, 4) sequence-first -> (B, N, 4) batch-first
            # box_labels: (N, B) -> (B, N)
            be = geometric_prompt.box_embeddings
            bl = geometric_prompt.box_labels
            if isinstance(be, torch.Tensor):
                be = be.detach().cpu().numpy()
            if isinstance(bl, torch.Tensor):
                bl = bl.detach().cpu().numpy()
            if be.size > 0:
                boxes_per_call = np.asarray(be, dtype=np.float32).transpose(1, 0, 2)  # (B, N, 4)
                labels_per_call = np.asarray(bl, dtype=np.int32).transpose(1, 0)  # (B, N)

        if nc == 1:
            return self._run_decoder(
                img_out=backbone_out,
                prompt_features=feats_all[:, ids, :],  # [32, 1, 256]
                prompt_mask=masks_all[ids],  # [1, 32]
                input_boxes=boxes_per_call,
                input_boxes_labels=labels_per_call,
            )

        # Multi-class: run decoder per-class, concatenate
        results = []
        for i in range(nc):
            r = self._run_decoder(
                img_out=backbone_out,
                prompt_features=feats_all[:, [ids[i]], :],
                prompt_mask=masks_all[[ids[i]]],
                input_boxes=boxes_per_call,
                input_boxes_labels=labels_per_call,
            )
            results.append(r)

        return {k: torch.cat([r[k] for r in results], dim=0) for k in results[0]}

    # ------------------------------------------------------------------
    # Compatibility stubs
    # ------------------------------------------------------------------

    def set_imgsz(self, imgsz) -> None:
        """No-op. Image size baked into model."""

    def eval(self):
        return self

    def to(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        return self

    def half(self):
        self.fp16 = True
        return self

    def float(self):
        self.fp16 = False
        return self

    def parameters(self):
        return iter([])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Free resources."""
        if self._format == "onnx":
            for attr in ("_vis_session", "_txt_session", "_dec_session", "_pe_session", "_md_session"):
                if hasattr(self, attr):
                    delattr(self, attr)
        else:
            self._trt_contexts.clear()
            self._trt_engines.clear()
        self.text_embeddings.clear()

    def __repr__(self) -> str:
        return f"SAM3Backend(format={self._format!r}, dir={str(self._model_dir)!r}, device={self.device}, names={self.names})"
