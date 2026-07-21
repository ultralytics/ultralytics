# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""SAM3 multi-file inference backend for ONNX and TensorRT.

Supports the multi-module architecture exported by ultralytics/utils/export/sam3_onnx.py:
  - Vision encoder:   images → fpn_feat_0/1/2, fpn_pos_2 (+ sam2_feat_0/1/2 dual neck)
  - Text encoder:     tokens → text_features [B,32,256], text_mask [B,32]
  - Decoder:          fpn features + prompt_features + prompt_mask → predictions
  - Prompt encoder:   point prompts → sparse/dense embeddings (optional)
  - Mask decoder:     embeddings + features → masks, ious, obj_ptrs, scores (optional)
  - Memory encoder:   frame features + mask → memory bank entry (optional, video)
  - Memory attention: current features ⨯ memory bank → conditioned features (optional, video)
  - Mask embed:       mask prompt → dense embeddings (optional, video)

With the video modules present, the backend duck-types the SAM2Model surface used by
SAM2VideoPredictor/SAM3VideoPredictor (``forward_image``, ``_prepare_backbone_features``,
``track_step``, ``_encode_new_memory``): frame/pointer selection and the memory bank live in
Python, mirroring SAM2Model, while all weight-bearing compute runs in the exported graphs. Small
tracker weights the mirror needs outside the graphs (no_mem_embed, temporal encodings, the
mask_downsample conv, no_obj_ptr) ship in the memory-attention module's metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class _BackboneProxy:
    """Proxy so ``predictor.model.backbone.forward_image(im)`` works (semantic image path)."""

    def __init__(self, backend: "SAM3Backend"):
        self._backend = backend

    def forward_image(self, im: torch.Tensor) -> dict:
        """Run vision encoder. Returns dict with ``backbone_fpn`` and ``vision_pos_enc``."""
        return self._backend.forward_image_semantic(im)


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
    _VIDEO_STEMS = ("sam3_memory_encoder", "sam3_memory_attention", "sam3_mask_embed")

    def __init__(self, model_dir: str | Path, device: torch.device | str = "cpu", fp16: bool = False):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.fp16 = fp16
        self.names: list[str] = []
        self.task = "segment"
        self.stride = 14
        self.text_embeddings: dict = {}
        self.backbone = _BackboneProxy(self)

        self.has_point_modules = False
        self.has_video_modules = False
        self._metadata: dict[str, dict] = {}  # stem → metadata dict

        self._model_dir = Path(model_dir)
        assert self._model_dir.is_dir(), f"Model directory not found: {self._model_dir}"

        self._format = self._detect_format()
        LOGGER.info(f"SAM3Backend: detected {self._format.upper()} in {self._model_dir}")
        self._load_models()
        self._init_video_state()

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

        if self.has_point_modules and all((self._model_dir / f"{s}.onnx").exists() for s in self._VIDEO_STEMS):
            for s in self._VIDEO_STEMS:
                self._sessions[s] = ort.InferenceSession(
                    str(self._model_dir / f"{s}.onnx"), sess_options=so, providers=providers
                )
            self.has_video_modules = True

        self._metadata = {s: dict(sess.get_modelmeta().custom_metadata_map) for s, sess in self._sessions.items()}
        LOGGER.info(f"SAM3Backend ONNX: loaded {', '.join(self._sessions)}")

    def _ort_providers(self, cuda: bool) -> list:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if cuda and "CUDAExecutionProvider" in available:
            # kSameAsRequested keeps each session's CUDA arena at its actual peak instead of
            # doubling on growth — with 8 sessions sharing one GPU, the default strategy
            # fragments enough to OOM the memory-attention softmax on 32GB cards.
            device_id = self.device.index or 0
            cuda_options = {"device_id": device_id, "arena_extend_strategy": "kSameAsRequested"}
            return [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # ---- TensorRT ----------------------------------------------------

    def _load_tensorrt(self) -> None:
        check_requirements(("tensorrt",))
        import tensorrt as trt

        if self.device.type == "cpu":
            LOGGER.warning("SAM3Backend TRT: CPU requested but TRT requires CUDA, using cuda:0")
            self.device = torch.device("cuda:0")

        logger = trt.Logger(trt.Logger.ERROR)

        paths = {s: self._model_dir / f"{s}.engine" for s in self._FILE_STEMS}
        for s, p in paths.items():
            assert p.exists(), f"Missing: {p}"

        self._trt_contexts: dict = {}
        self._trt_engines: dict = {}
        self._trt_io_dtypes: dict[str, dict[str, torch.dtype]] = {}
        self._trt_activation: dict[str, torch.Tensor] = {}  # per-engine activation memory, grow-only
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
        if self.has_point_modules and all((self._model_dir / f"{s}.engine").exists() for s in self._VIDEO_STEMS):
            paths.update({s: self._model_dir / f"{s}.engine" for s in self._VIDEO_STEMS})
            all_stems.extend(self._VIDEO_STEMS)
            self.has_video_modules = True

        for stem in all_stems:
            with open(paths[stem], "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")
                    self._metadata[stem] = json.loads(f.read(meta_len).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    f.seek(0)
                engine = runtime.deserialize_cuda_engine(f.read())

            # Contexts are created without device memory: the default strategy preallocates
            # activation memory for the MAX optimization profile, which for the memory
            # attention engine (32 objects x 16 memory frames) exceeds 20GB. _run_trt sizes
            # and attaches activation memory for the actual shapes of each call instead.
            ctx = engine.create_execution_context_without_device_memory()
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

        # Attach activation memory sized for the CURRENT shapes (contexts are created without
        # device memory to avoid max-profile preallocation); grow-only cache per engine.
        size = int(ctx.update_device_memory_size_for_shapes())
        buf = self._trt_activation.get(stem)
        if size > 0 and (buf is None or buf.numel() < size):
            buf = torch.empty(size, dtype=torch.uint8, device=self.device)
            self._trt_activation[stem] = buf
        if buf is not None:
            ctx.set_device_memory(buf.data_ptr(), buf.numel())

        # (Re)allocate output buffers based on current shapes (handles dynamic outputs).
        out_bufs = {}
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tname) == trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(tname))
                out = torch.empty(shape, dtype=io_dt[tname], device=self.device)
                out_bufs[tname] = out
                ctx.set_tensor_address(tname, out.data_ptr())

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

    def forward_image_semantic(self, im: torch.Tensor) -> dict:
        """Run vision encoder for the semantic (DETR grounding) path.

        Args:
            im: [1, 3, H, W] normalized image tensor.

        Returns:
            dict with ``backbone_fpn`` and ``vision_pos_enc`` for the DETR decoder,
            plus ``_fpn_feat_*`` for decoder feed and ``_sam2_feat_*`` for point prompts
            (if the dual SAM2 neck was exported).

        Note:
            ``forward_image`` (no suffix) is the video-tracking entry and returns SAM2-neck
            features instead, matching SAM2Model.forward_image.
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

        image_embeddings = img_out[f"{prefix}2"]
        if self._no_mem_embed_spatial is not None:
            # no_mem_embed is no longer baked into the mask decoder graph (video tracking adds
            # memory-conditioned features instead); apply the initial-frame bias here.
            image_embeddings = image_embeddings + self._no_mem_embed_spatial.to(image_embeddings.device)
        num_prompts = point_coords.shape[0]
        if num_prompts > 1 and image_embeddings.shape[0] == 1:  # decoder expects matching batch
            image_embeddings = image_embeddings.expand(num_prompts, -1, -1, -1).contiguous()

        pe_out = self._run(self._POINT_STEMS[0], {"point_coords": point_coords, "point_labels": point_labels})
        md_out = self._run(
            self._POINT_STEMS[1],
            {
                "image_embeddings": image_embeddings,
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
            out = self._run_decoder(
                img_out=backbone_out,
                prompt_features=feats_all[:, ids, :],  # [32, 1, 256]
                prompt_mask=masks_all[ids],  # [1, 32]
                input_boxes=boxes_per_call,
                input_boxes_labels=labels_per_call,
            )
        else:
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
            out = {k: torch.cat([r[k] for r in results], dim=0) for k in results[0]}

        # xyxy boxes are a weightless transform the torch head emits alongside pred_boxes
        # (consumed by SAM3VideoSemanticPredictor._extract_detection_outputs)
        from ultralytics.utils.ops import xywh2xyxy

        out["pred_boxes_xyxy"] = xywh2xyxy(out["pred_boxes"])

        # Surface the SAM2-neck pyramid for the video tracker's feature cache
        # (SAM3VideoSemanticPredictor._cache_backbone_features). Levels stay unprojected —
        # conv_s0/conv_s1 run inside the sam3_mask_decoder graph — and only the level-2
        # position encoding carries real values (levels 0/1 are shape carriers).
        if "_sam2_feat_0" in backbone_out:
            feats = [backbone_out[f"_sam2_feat_{i}"] for i in range(3)]
            pos2 = backbone_out["_fpn_pos_2"]
            pos = [pos2.new_zeros(1, 1, *f.shape[-2:]).expand(1, f.shape[1], -1, -1) for f in feats[:2]] + [pos2]
            out["backbone_out"] = {
                "sam2_backbone_out": {"backbone_fpn": feats, "vision_pos_enc": pos, "vision_features": feats[-1]}
            }
        return out

    # ------------------------------------------------------------------
    # Video tracking: SAM2Model-compatible surface over the exported graphs
    #
    # SAM2VideoPredictor/SAM3VideoPredictor drive tracking through
    # model.forward_image → model._prepare_backbone_features → model.track_step
    # (plus model._encode_new_memory during multi-object consolidation). The
    # methods below mirror SAM2Model's orchestration in plain Python and call
    # the exported graphs for every weight-bearing step. Frame/pointer
    # selection logic is reused from ultralytics.models.sam.modules where it
    # is weight-free (select_closest_cond_frames, non-overlap constraints).
    # ------------------------------------------------------------------

    def _init_video_state(self) -> None:
        """Parse tracker constants from module metadata and expose the SAM2Model-compatible surface."""
        self.no_mem_embed = None
        self._no_mem_embed_spatial = None
        md_meta = self._metadata.get("sam3_mask_decoder", {})
        if "no_mem_embed" in md_meta:  # current exports ship no_mem_embed unbaked, added at runtime
            self.no_mem_embed = torch.tensor(json.loads(md_meta["no_mem_embed"]), dtype=torch.float32).view(1, 1, -1)
            self._no_mem_embed_spatial = self.no_mem_embed.view(1, -1, 1, 1)
        elif self.has_point_modules:
            LOGGER.warning(
                "SAM3Backend: legacy export detected (no_mem_embed baked into sam3_mask_decoder); "
                "point prompts work but video tracking requires a re-export."
            )
            self.has_video_modules = False

        if not self.has_video_modules:
            return
        vc = json.loads(self._metadata["sam3_memory_attention"]["video_constants"])
        # Tracker weights the Python mirror needs outside the graphs
        self.no_obj_ptr = torch.tensor(vc["no_obj_ptr"], dtype=torch.float32).view(1, -1)
        self.maskmem_tpos_enc = torch.tensor(vc["maskmem_tpos_enc"], dtype=torch.float32).view(
            vc["num_maskmem"], 1, 1, vc["mem_dim"]
        )
        self._mask_downsample_w = torch.tensor(vc["mask_downsample_weight"], dtype=torch.float32).view(1, 1, 4, 4)
        self._mask_downsample_b = torch.tensor(vc["mask_downsample_bias"], dtype=torch.float32)
        # Tracker config mirrored from build_interactive_sam3 (shipped values + fixed SAM3 flags)
        self.num_maskmem = vc["num_maskmem"]
        self.mem_dim = vc["mem_dim"]
        self.hidden_dim = vc["hidden_dim"]
        self.image_size = vc["image_size"]
        self.sigmoid_scale_for_mem_enc = vc["sigmoid_scale_for_mem_enc"]
        self.sigmoid_bias_for_mem_enc = vc["sigmoid_bias_for_mem_enc"]
        self.max_obj_ptrs_in_encoder = vc["max_obj_ptrs_in_encoder"]
        self.memory_temporal_stride_for_eval = vc["memory_temporal_stride_for_eval"]
        self.max_cond_frames_in_attn = vc["max_cond_frames_in_attn"]
        self.num_feature_levels = 3
        self.mask_threshold = 0.0
        self.training = False
        self.add_all_frames_to_correct_as_cond = True
        self.directly_add_no_mem_embed = True
        self.use_mask_input_as_output_without_sam = True
        self.non_overlap_masks_for_mem_enc = False
        self.binarize_mask_from_pts_for_mem_enc = False
        self.use_obj_ptrs_in_encoder = True
        self.only_obj_ptrs_in_the_past_for_eval = True
        self.use_signed_tpos_enc_to_obj_ptrs = True
        self.multimask_output_in_sam = True
        self.multimask_output_for_tracking = True
        self.multimask_min_pt_num = 0
        self.multimask_max_pt_num = 1
        self.pred_obj_scores = True
        # Predictor-touched submodule attributes. conv_s0/conv_s1 are identity because the
        # projection is folded inside the sam3_mask_decoder graph (high-res features flow raw).
        self.memory_encoder = SimpleNamespace(mask_downsampler=SimpleNamespace(interpol_size=list(vc["interpol_size"])))
        self.memory_attention = SimpleNamespace(d_model=self.hidden_dim)
        self.sam_mask_decoder = SimpleNamespace(conv_s0=lambda x: x, conv_s1=lambda x: x)

    def set_binarize(self, binarize: bool = False) -> None:
        """Set mask binarization for the memory encoder on interacted frames (SAM2Model API)."""
        self.binarize_mask_from_pts_for_mem_enc = binarize

    def forward_image(self, im: torch.Tensor) -> dict:
        """Run the vision encoder for video tracking (SAM2Model.forward_image equivalent).

        Returns the SAM2-neck feature pyramid the tracker consumes. Unlike torch SAM3Model,
        levels 0/1 stay unprojected (the sam3_mask_decoder graph applies conv_s0/conv_s1), and
        their position encodings are zero shape-carriers — only the level-2 encoding is consumed
        (as curr_pos in memory attention), and it is identical for both necks.
        """
        out = self._run(self._FILE_STEMS[0], {"images": im})
        assert "sam2_feat_0" in out, (
            "SAM3Backend video tracking requires the dual-neck vision encoder (sam2_feat_* outputs); re-export."
        )
        feats = [out["sam2_feat_0"], out["sam2_feat_1"], out["sam2_feat_2"]]
        pos2 = out["fpn_pos_2"]
        pos = [pos2.new_zeros(1, 1, *f.shape[-2:]).expand(1, f.shape[1], -1, -1) for f in feats[:2]] + [pos2]
        return {"backbone_fpn": feats, "vision_pos_enc": pos, "vision_features": feats[-1]}

    def _prepare_backbone_features(self, backbone_out: dict, batch: int = 1):
        """Flatten multi-level features to (HW)BC lists (SAM2Model._prepare_backbone_features mirror)."""
        if batch > 1:  # expand features if there's more than one prompt/object
            backbone_out = {
                **backbone_out,
                "backbone_fpn": [feat.expand(batch, -1, -1, -1) for feat in backbone_out["backbone_fpn"]],
                "vision_pos_enc": [pos.expand(batch, -1, -1, -1) for pos in backbone_out["vision_pos_enc"]],
            }
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    @staticmethod
    def _apply_non_overlapping_constraints(pred_masks: torch.Tensor) -> torch.Tensor:
        """Keep only the highest-scoring object per location (SAM2Model static method reused)."""
        from ultralytics.models.sam.modules.sam import SAM2Model

        return SAM2Model._apply_non_overlapping_constraints(pred_masks)

    @staticmethod
    def _suppress_object_pw_area_shrinkage(pred_masks: torch.Tensor) -> torch.Tensor:
        """Suppress masks that shrink heavily under pixel-wise non-overlap (SAM3Model statics reused)."""
        from ultralytics.models.sam.modules.sam import SAM2Model, SAM3Model

        return SAM3Model._suppress_shrinked_masks(pred_masks, SAM2Model._apply_non_overlapping_constraints(pred_masks))

    def _use_multimask(self, is_init_cond_frame: bool, point_inputs: dict | None) -> bool:
        """Whether the SAM head should output multiple masks (SAM2Model._use_multimask mirror)."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )

    def _forward_sam_heads(
        self,
        backbone_features: torch.Tensor,
        point_inputs: dict | None = None,
        mask_inputs: torch.Tensor | None = None,
        high_res_features: list[torch.Tensor] | None = None,
        multimask_output: bool = False,
    ):
        """SAM prompt encoder + mask decoder over the exported graphs (SAM2Model mirror).

        The exported mask decoder always produces 3 candidates with object pointers and score
        logits (occlusion suppression and pointer mixing happen inside the graph); the best
        candidate is selected here by IoU. high_res_features are raw 256-ch SAM2-neck features
        at batch 1 — the graph projects and broadcasts them.
        """
        B = backbone_features.shape[0]
        device = backbone_features.device
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
        else:  # pad with an empty point (label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        pe_out = self._run(self._POINT_STEMS[0], {"point_coords": sam_point_coords, "point_labels": sam_point_labels})
        dense_embeddings = pe_out["dense_embeddings"]
        if mask_inputs is not None:  # dense mask prompt replaces the no-mask embedding
            mask_input_size = (self.image_size // 14 * 4,) * 2
            sam_mask_prompt = mask_inputs.float()
            if sam_mask_prompt.shape[-2:] != mask_input_size:
                sam_mask_prompt = F.interpolate(
                    sam_mask_prompt, size=mask_input_size, align_corners=False, mode="bilinear", antialias=True
                )
            dense_embeddings = self._run(self._VIDEO_STEMS[2], {"mask_input": sam_mask_prompt})["dense_embeddings"]

        md_out = self._run(
            self._POINT_STEMS[1],
            {
                "image_embeddings": backbone_features,
                "image_pe": pe_out["dense_pe"],
                "sparse_prompt_embeddings": pe_out["sparse_embeddings"],
                "dense_prompt_embeddings": dense_embeddings,
                "high_res_feat_0": high_res_features[0][:1],
                "high_res_feat_1": high_res_features[1][:1],
            },
        )
        low_res_multimasks = md_out["masks"].float()
        ious = md_out["iou_scores"]
        obj_ptrs = md_out["obj_ptrs"]
        object_score_logits = md_out["object_score_logits"]

        high_res_multimasks = F.interpolate(
            low_res_multimasks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        # Best-candidate selection by IoU. The graph is exported with multimask=3, so the
        # multimask_output=False case (>1-point refinement) also selects the best of 3 —
        # matching the single-image point path already shipped in this backend.
        best_iou_inds = torch.argmax(ious, dim=-1)
        batch_inds = torch.arange(B, device=device)
        low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        obj_ptr = obj_ptrs[batch_inds, best_iou_inds]

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, mask_inputs, backbone_features=None, high_res_features=None):
        """Treat a mask prompt directly as output (SAM2Model._use_mask_as_output mirror)."""
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
        ious = mask_inputs.new_ones(mask_inputs.shape[0], 1).float()
        if backbone_features is None or high_res_features is None:
            obj_ptr = torch.zeros(mask_inputs.shape[0], self.hidden_dim, device=mask_inputs.device)
        else:  # produce an object pointer from the mask input via the SAM decoder
            downsampled = F.conv2d(
                mask_inputs_float,
                self._mask_downsample_w.to(mask_inputs.device),
                self._mask_downsample_b.to(mask_inputs.device),
                stride=4,
            )
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features, mask_inputs=downsampled, high_res_features=high_res_features
            )
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr  # fixed_no_obj_ptr=True
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr.to(obj_ptr.device)
        return low_res_masks, high_res_masks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """Fuse current features with the memory bank (SAM2Model mirror over the exported graph).

        Frame and object-pointer selection is identical to SAM2Model; the composed tensors go to
        the sam3_memory_attention graph batch-first, with spatial memory and pointers as separate
        inputs (RoPE exclusion of pointer tokens is structural in the graph).
        """
        from ultralytics.models.sam.modules.utils import select_closest_cond_frames

        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        if is_init_cond_frame:  # directly_add_no_mem_embed=True: no memory attention on init frames
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed.to(device)
            return pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

        # --- select memory frames (identical to SAM2Model._prepare_memory_conditioned_features)
        cond_outputs = output_dict["cond_frame_outputs"]
        assert len(cond_outputs) > 0
        selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
            frame_idx, cond_outputs, self.max_cond_frames_in_attn
        )
        t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
        r = self.memory_temporal_stride_for_eval
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if t_rel == 1:
                prev_frame_idx = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
            elif not track_in_reverse:
                prev_frame_idx = ((frame_idx - 2) // r) * r - (t_rel - 2) * r
            else:
                prev_frame_idx = -(-(frame_idx + 2) // r) * r + (t_rel - 2) * r
            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
            if out is None:
                out = unselected_cond_outputs.get(prev_frame_idx, None)
            t_pos_and_prevs.append((t_pos, out))

        spatial_mem, spatial_mem_pos = [], []
        tpos_enc = self.maskmem_tpos_enc.to(device)  # [num_maskmem, 1, 1, mem_dim]
        for t_pos, prev in t_pos_and_prevs:
            if prev is None:
                continue
            feats = prev["maskmem_features"].to(device=device, non_blocking=device.type == "cuda").float()
            spatial_mem.append(feats.flatten(2).permute(0, 2, 1))  # [B, HW, mem_dim]
            pos = prev["maskmem_pos_enc"][-1][:1].to(device=device).float()
            pos = pos.flatten(2).permute(0, 2, 1)  # [1, HW, mem_dim]
            spatial_mem_pos.append(pos + tpos_enc[self.num_maskmem - t_pos - 1].view(1, 1, -1))

        # --- select past object pointers (identical to SAM2Model)
        max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
        tpos_sign_mul = -1 if track_in_reverse else 1
        ptr_cond_outputs = {
            t: out
            for t, out in selected_cond_outputs.items()
            if (t >= frame_idx if track_in_reverse else t <= frame_idx)
        }
        pos_and_ptrs = [((frame_idx - t) * tpos_sign_mul, out["obj_ptr"]) for t, out in ptr_cond_outputs.items()]
        for t_diff in range(1, max_obj_ptrs_in_encoder):
            t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
            if t < 0 or (num_frames is not None and t >= num_frames):
                break
            out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
            if out is not None:
                pos_and_ptrs.append((t_diff, out["obj_ptr"]))
        assert pos_and_ptrs, "memory attention requires at least one past object pointer"
        pos_list, ptrs_list = zip(*pos_and_ptrs)
        obj_ptrs = torch.stack([p.to(device).float() for p in ptrs_list], dim=1)  # [B, P, C]
        ptr_tpos = torch.tensor(pos_list, device=device, dtype=torch.float32) / max(max_obj_ptrs_in_encoder - 1, 1)

        out = self._run(
            self._VIDEO_STEMS[1],
            {
                "curr": current_vision_feats[-1][:, :1].permute(1, 0, 2),  # [1, HW, C]
                "curr_pos": current_vision_pos_embeds[-1][:, :1].permute(1, 0, 2),
                "spatial_mem": torch.cat(spatial_mem, dim=1),
                "spatial_mem_pos": torch.cat(spatial_mem_pos, dim=1),
                "obj_ptrs": obj_ptrs,
                "ptr_tpos": ptr_tpos,
            },
        )
        return out["pix_feat_with_mem"].float()

    def _encode_new_memory(
        self, current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts
    ):
        """Encode the current frame + predicted mask into a memory entry (SAM2Model mirror)."""
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1][:, :1].permute(1, 2, 0).view(1, C, H, W)  # graph broadcasts over B
        if self.non_overlap_masks_for_mem_enc and B > 1:
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
        if self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc + self.sigmoid_bias_for_mem_enc
        interpol_size = self.memory_encoder.mask_downsampler.interpol_size
        if list(mask_for_mem.shape[-2:]) != interpol_size:  # weightless antialias resize kept outside the graph
            mask_for_mem = F.interpolate(
                mask_for_mem.float(), size=interpol_size, align_corners=False, mode="bilinear", antialias=True
            )
        out = self._run(
            self._VIDEO_STEMS[0],
            {"pix_feat": pix_feat, "mask_for_mem": mask_for_mem, "object_score_logits": object_score_logits},
        )
        return out["maskmem_features"], [out["maskmem_pos_enc"]]

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
    ):
        """Single tracking step over the exported graphs (SAM2Model.track_step mirror)."""
        assert self.has_video_modules, "Video tracking requires memory modules; re-export this model directory."
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(mask_inputs, pix_feat, high_res_features)
        else:
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs

        current_out = {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem = high_res_masks
            if self.non_overlap_masks_for_mem_enc and high_res_masks.size(0) > 1:
                high_res_masks_for_mem = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats,
                feat_sizes,
                high_res_masks_for_mem,
                object_score_logits,
                is_mask_from_pts=point_inputs is not None,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None
        return current_out

    # ------------------------------------------------------------------
    # Compatibility stubs
    # ------------------------------------------------------------------

    def set_imgsz(self, imgsz) -> None:
        """Validate requested size against the exported graphs (shapes are baked at export time)."""
        exported = self._metadata.get(self._FILE_STEMS[0], {}).get("imgsz")
        if exported:
            try:
                exported = json.loads(exported) if isinstance(exported, str) else exported
            except json.JSONDecodeError:
                return
            if list(imgsz) != list(exported):
                LOGGER.warning(
                    f"SAM3Backend: imgsz={list(imgsz)} requested but the model was exported at {exported}; "
                    f"inference runs at {exported}."
                )

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
            self._sessions.clear()
        else:
            self._trt_contexts.clear()
            self._trt_engines.clear()
            self._trt_activation.clear()
        self.text_embeddings.clear()

    def __repr__(self) -> str:
        return f"SAM3Backend(format={self._format!r}, dir={str(self._model_dir)!r}, device={self.device}, names={self.names})"
