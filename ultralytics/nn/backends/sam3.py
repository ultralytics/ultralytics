# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ultralytics.nn.backends.base import BaseBackend
from ultralytics.nn.backends.onnx import _ORT_DTYPES
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class _BackboneProxy:
    """Proxy so ``predictor.model.backbone.forward_image(im)`` works."""

    def __init__(self, backend: SAM3Backend):
        """Bind the proxy to its owning backend."""
        self._backend = backend

    def forward_image(self, im: torch.Tensor) -> dict:
        """Run vision encoder. Returns dict with ``backbone_fpn`` and ``vision_pos_enc``."""
        return self._backend.forward_image(im)


class SAM3Backend:
    """Multi-file inference backend for SAM3 ONNX and TensorRT models.

    Presents one interface over both formats for ``SAM3SemanticPredictor``.

    Attributes:
        names (list[str]): Current class names.
        device (torch.device): Device the outputs are placed on.
        fp16 (bool): Whether half precision is requested.
        task (str): Always ``"segment"``.
        stride (int): 14 (ViT patch size).
        text_embeddings (dict): Cached text encoder outputs.
        backbone (_BackboneProxy): Proxy with ``forward_image``.
        has_point_modules (bool): Whether prompt encoder + mask decoder are available.
        dynamic (bool): Whether the export accepts a range of image sizes rather than one baked size.
    """

    # Grounding masks are raw logits, so the predictor thresholds them at zero. Mirrors SAM2Model.
    mask_threshold: float = 0.0

    _FILE_STEMS = ("sam3_vision_encoder", "sam3_text_encoder", "sam3_decoder")
    _TEXT_DECODER = "sam3_decoder_text"
    _POINT_STEMS = ("sam3_prompt_encoder", "sam3_mask_decoder")

    def __init__(self, model_dir: str | Path, device: torch.device | str = "cpu", fp16: bool = False):
        """Load every exported SAM3 module found in ``model_dir``.

        Args:
            model_dir (str | Path): Directory holding the exported ``.onnx`` or ``.engine`` files.
            device (torch.device | str): Device to run on and place outputs on.
            fp16 (bool): Whether to request half precision.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.fp16 = fp16
        self.names: list[str] = []
        self.task = "segment"
        self.stride = 14
        self.text_embeddings: dict = {}
        self.backbone = _BackboneProxy(self)

        self._model_dir = Path(model_dir)
        assert self._model_dir.is_dir(), f"Model directory not found: {self._model_dir}"

        self._format = self._detect_format()
        self.has_point_modules = self._present(self._POINT_STEMS)
        self._has_text_decoder = self._present([self._TEXT_DECODER])
        # An ONNX session binds to an execution provider when it is created, and the caller usually
        # picks a device after construction, so defer loading and read the traced size from the file.
        # TensorRT is CUDA only, so there is nothing to wait for and the engine can report its shape.
        self._loaded = False
        if self._format == "engine":
            self._load_models()
        self.imgsz = self._baked_imgsz()
        self.imgsz_range = None if self.imgsz else self._dynamic_imgsz_range()
        # Every other backend inherits this from BaseBackend, and Model.predict reads it when reusing a
        # predictor, so a standalone backend without it breaks on the second prompt of a session.
        self.dynamic = self.imgsz is None
        LOGGER.info(f"SAM3Backend: detected {self._format.upper()} in {self._model_dir} at {self._accepted_imgsz()}")

    def _accepted_imgsz(self) -> str:
        """Describe the image size this export accepts, for logging and for the warning that rejects one."""
        if self.imgsz:
            return f"imgsz {self.imgsz}"
        return f"imgsz {self.imgsz_range[0]} to {self.imgsz_range[1]}" if self.imgsz_range else "any imgsz"

    def _dynamic_imgsz_range(self) -> tuple[int, int] | None:
        """Return the smallest and largest image size a dynamic export accepts, or None if unreadable.

        TensorRT is asked for its optimization profile, which is what actually constrains it. ONNX has
        no such bound in the graph, so the range the export recorded is read back from the metadata.
        """
        stem = self._FILE_STEMS[0]
        try:
            if self._format == "engine":
                lo, _, hi = self._trt_engines[stem].get_tensor_profile_shape("images", 0)
                return int(lo[2]), int(hi[2])
            meta = BaseBackend.read_metadata(self._model_dir / f"{stem}.onnx")
            return json.loads(meta["min_imgsz"])[0], json.loads(meta["imgsz"])[0]
        except Exception:  # an export without a recorded range simply has no bound to check against
            return None

    def _baked_imgsz(self) -> int | None:
        """Return the image size the vision encoder was traced at.

        None when the graph is dynamic in height and width, or when the shape cannot be read, so the
        caller's own image size is left alone instead of being overridden by a symbolic dimension.
        """
        stem = self._FILE_STEMS[0]
        try:
            if self._format == "engine":
                size = int(self._trt_engines[stem].get_tensor_shape("images")[2])
            else:
                # The export records min_imgsz only when it traced a dynamic graph, so its absence is what
                # marks a baked size. Reading that from the metadata avoids parsing a multi gigabyte graph.
                meta = BaseBackend.read_metadata(self._model_dir / f"{stem}.onnx")
                size = 0 if "min_imgsz" in meta else json.loads(meta["imgsz"])[0]
            return size if size > 0 else None  # a dynamic axis reports 0 in ONNX and -1 in TensorRT
        except Exception:  # an unreadable shape must not stop the model from loading
            return None

    def _present(self, stems, ext: str | None = None) -> bool:
        """Whether every stem exists in the model directory with the given extension."""
        return all((self._model_dir / f"{s}.{ext or self._format}").exists() for s in stems)

    def _detect_format(self) -> str:
        """Return ``"onnx"`` or ``"engine"`` based on which required files the directory holds."""
        for ext in ("onnx", "engine"):
            if self._present(self._FILE_STEMS, ext):
                return ext
        raise FileNotFoundError(
            f"Need {', '.join(self._FILE_STEMS)} as .onnx or .engine in {self._model_dir}, found neither set."
        )

    def _stems_to_load(self) -> list[str]:
        """Return every module stem present on disk."""
        stems = list(self._FILE_STEMS)
        if self._has_text_decoder:
            stems.append(self._TEXT_DECODER)
        if self.has_point_modules:
            stems.extend(self._POINT_STEMS)
        return stems

    def _loaded_desc(self) -> str:
        """Return a human readable list of the modules that were loaded."""
        desc = "vision encoder, text encoder, decoder"
        if self._has_text_decoder:
            desc += ", text decoder"
        return desc + (", prompt encoder, mask decoder" if self.has_point_modules else "")

    def _load_models(self) -> None:
        """Load every module once, on first use, for whichever device is current by then."""
        if self._format == "onnx":
            self._load_onnx()
        else:
            self._load_tensorrt()
        self._loaded = True

    def _load_onnx(self) -> None:
        cuda = self.device.type != "cpu" and torch.cuda.is_available()
        check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime",))
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        providers = self._ort_providers(cuda)
        LOGGER.info(f"SAM3Backend ONNX: using {providers[0] if isinstance(providers[0], str) else providers[0][0]}")

        stems = self._stems_to_load()
        paths = {s: self._model_dir / f"{s}.onnx" for s in stems}
        for s in self._FILE_STEMS:
            assert paths[s].exists(), f"Missing: {paths[s]}"

        so = ort.SessionOptions()
        so.log_severity_level = 3

        # Sessions are created on first use. A text prompt never touches the prompt encoder or mask
        # decoder, and holding every module resident cost a smaller card gigabytes it does not have.
        self._onnx_paths = paths
        self._session_opts = (so, providers)
        self._sessions = {}
        LOGGER.info(f"SAM3Backend ONNX: found {self._loaded_desc()}, loading each on first use")

    def _session(self, stem: str):
        """Return the session for ``stem``, creating it the first time it is asked for."""
        if stem not in self._sessions:
            import onnxruntime as ort

            so, providers = self._session_opts
            LOGGER.info(f"SAM3Backend ONNX: loading {stem}")
            self._sessions[stem] = ort.InferenceSession(
                str(self._onnx_paths[stem]), sess_options=so, providers=providers
            )
        return self._sessions[stem]

    def _ort_providers(self, cuda: bool) -> list:
        """Return the ONNX Runtime provider list, preferring CUDA when it is available."""
        import onnxruntime as ort

        if cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            # All modules stay resident, and the decoder attention allocates in hundreds of megabytes.
            # The default arena grows by doubling, which reserves far more than that and can fail the
            # next large allocation, so ask it for exactly what each node needs instead.
            opts = {"device_id": self.device.index or 0, "arena_extend_strategy": "kSameAsRequested"}
            return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_tensorrt(self) -> None:
        check_requirements(("tensorrt",))
        import tensorrt as trt

        if self.device.type == "cpu":
            LOGGER.warning("SAM3Backend TRT: CPU requested but TRT requires CUDA, using cuda:0")
            self.device = torch.device("cuda:0")

        logger = trt.Logger(trt.Logger.ERROR)

        stems = self._stems_to_load()
        paths = {s: self._model_dir / f"{s}.engine" for s in stems}
        for s in self._FILE_STEMS:
            assert paths[s].exists(), f"Missing: {paths[s]}"

        self._trt_contexts: dict = {}
        self._trt_engines: dict = {}
        self._trt_io_dtypes: dict[str, dict[str, torch.dtype]] = {}
        self._cuda_stream = torch.cuda.Stream(device=self.device)

        for stem in stems:
            with open(paths[stem], "rb") as f, trt.Runtime(logger) as runtime:
                f.seek(BaseBackend.engine_header(paths[stem])[0])  # skip the optional metadata header
                engine = runtime.deserialize_cuda_engine(f.read())

            assert engine is not None, f"{paths[stem].name} failed to load, the GPU is likely out of memory"
            ctx = engine.create_execution_context()
            assert ctx is not None, f"{paths[stem].name} got no execution context, the GPU is likely out of memory"
            # Only per-tensor dtypes are needed at load time; _run_trt sets shapes and
            # allocates all output buffers at runtime (outputs can be dynamic).
            io_dt = {
                name: torch.from_numpy(np.empty(0, dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype
                for name in map(engine.get_tensor_name, range(engine.num_io_tensors))
            }

            self._trt_engines[stem] = engine
            self._trt_contexts[stem] = ctx
            self._trt_io_dtypes[stem] = io_dt

        LOGGER.info(f"SAM3Backend TRT: loaded {self._loaded_desc()}")

    @staticmethod
    def _run_onnx(session, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run an ONNX session, casting each input to the dtype the graph declares.

        Args:
            session (onnxruntime.InferenceSession): Session to run.
            feed (dict[str, np.ndarray]): Input name to array, matching the graph inputs.

        Returns:
            (dict[str, np.ndarray]): Output name to array.
        """
        cast = {i.name: feed[i.name].astype(_ORT_DTYPES[i.type][1], copy=False) for i in session.get_inputs()}
        raw = session.run(None, cast)
        return {o.name: v for o, v in zip(session.get_outputs(), raw)}

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
        """Run module ``stem`` on either backend.

        Accepts tensors or arrays, both runners re-cast dtypes, and always returns torch tensors
        on ``self.device``.
        """
        if not self._loaded:
            self._load_models()
        if self._format == "onnx":
            np_feed = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in feed.items()}
            try:
                out = self._run_onnx(self._session(stem), np_feed)
            except Exception as e:
                # The decoder attention allocates in hundreds of megabytes, so a smaller card can
                # still run out part way through even loading one module at a time. Keep serving the
                # request on CPU rather than failing, and stay there so later prompts do not retry.
                if self.device.type == "cpu" or "alloc" not in str(e).lower():
                    raise
                LOGGER.warning(f"SAM3Backend ONNX: {self.device} ran out of memory, falling back to CPU. {e!s:.120}")
                self.to("cpu")
                self._load_models()
                out = self._run_onnx(self._session(stem), np_feed)
            return {k: torch.from_numpy(v).to(self.device) for k, v in out.items()}
        cuda_feed = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v)).to(self.device))
            for k, v in feed.items()
        }
        return self._run_trt(stem, cuda_feed)

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

    def _run_decoder(
        self,
        img_out: dict,
        prompt_features: np.ndarray,
        prompt_mask: np.ndarray,
        input_boxes: np.ndarray | None = None,
        input_boxes_labels: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the decoder on FPN features, a prompt, and optional box prompts.

        Args:
            img_out: Dict from forward_image (with _fpn_feat_* and _fpn_pos_2 cached).
            prompt_features: [seq, B, 256] text/prompt features (sequence-first).
            prompt_mask: [B, seq] bool mask (True=valid token).
            input_boxes: [B, num_boxes, 4] normalized CxCyWH, or None for text-only.
            input_boxes_labels: [B, num_boxes] int32 (1=pos, 0=neg, -10=ignore).

        Returns:
            dict with pred_logits, pred_boxes, pred_masks, presence_logits.
        """
        feed = {
            "fpn_feat_0": img_out["_fpn_feat_0"],
            "fpn_feat_1": img_out["_fpn_feat_1"],
            "fpn_feat_2": img_out["_fpn_feat_2"],
            "fpn_pos_2": img_out["_fpn_pos_2"],
            "prompt_features": prompt_features,
            "prompt_mask": prompt_mask,
        }
        # An ignored box is not neutral, it appends a geometry token that shifts the presence logit,
        # so a text only prompt must use the graph exported without geometry.
        if input_boxes is None:
            assert self._has_text_decoder, (
                f"Text only prompts need {self._TEXT_DECODER} in {self._model_dir}. Re-export this model."
            )
            return self._run(self._TEXT_DECODER, feed)
        feed["input_boxes"] = input_boxes
        feed["input_boxes_labels"] = input_boxes_labels
        return self._run(self._FILE_STEMS[2], feed)

    def forward_points(
        self,
        img_out: dict,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run point prompts through the prompt encoder and mask decoder.

        Requires the SAM2 neck features, whose weights match the mask decoder.

        Args:
            img_out (dict): Dict from forward_image, with ``_sam2_feat_*`` cached.
            point_coords (np.ndarray): [B, N, 2] float32 point coordinates in pixel space.
            point_labels (np.ndarray): [B, N] int32, 1=foreground, 0=background.

        Returns:
            masks (torch.Tensor): [B, num_masks, H, W] predicted masks.
            iou_scores (torch.Tensor): [B, num_masks] quality scores.
        """
        assert self.has_point_modules, "Point prompt modules not found. Re-export with prompt_encoder + mask_decoder."

        # The mask decoder is trained against the SAM2 neck, whose weights are separate from the SAM3
        # FPN. Substituting the SAM3 features gives a different feature space (cosine ~0.42) and
        # scattered masks, so refuse rather than return wrong output. forward_image always sets the
        # three sam2 keys together, so one check suffices.
        assert "_sam2_feat_0" in img_out, (
            "Point prompts need a vision encoder exported with the SAM2 neck (sam2_feat_0/1/2). Re-export this model."
        )

        # A dynamic prompt encoder reads the feature grid off the image embedding, because the dense
        # embedding, its positional encoding and the coordinate scale all follow from that grid. Both
        # runners drop feed keys the graph does not declare, so a static export ignores it.
        pe_out = self._run(
            self._POINT_STEMS[0],
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
                "image_embeddings": img_out["_sam2_feat_2"],
            },
        )
        md_out = self._run(
            self._POINT_STEMS[1],
            {
                "image_embeddings": img_out["_sam2_feat_2"],
                "image_pe": pe_out["dense_pe"],
                "sparse_prompt_embeddings": pe_out["sparse_embeddings"],
                "dense_prompt_embeddings": pe_out["dense_embeddings"],
                "high_res_feat_0": img_out["_sam2_feat_0"],
                "high_res_feat_1": img_out["_sam2_feat_1"],
            },
        )
        return md_out["masks"], md_out["iou_scores"]

    def set_classes(self, text: list[str]) -> None:
        """Tokenize text, run text encoder per-class, cache results."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        # Use the tokenizer callable, as the PyTorch text encoder does. Truncating by hand drops the
        # end of text token on long prompts, and the encoder pools its features from that token.
        tokens = clip.simple_tokenizer.SimpleTokenizer()(text, context_length=32).numpy().astype(np.int64)

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

    def forward_grounding(
        self, backbone_out: dict, text_ids: torch.Tensor, geometric_prompt=None
    ) -> dict[str, torch.Tensor]:
        """Select cached text features, add any box prompts, and run the decoder.

        Args:
            backbone_out (dict): Dict from forward_image.
            text_ids (torch.Tensor): [nc] class indices into cached text_embeddings.
            geometric_prompt (Any): Optional Prompt object with box_embeddings and box_labels.

        Returns:
            (dict[str, torch.Tensor]): pred_logits, pred_boxes, pred_masks, presence_logits.
        """
        assert self.text_embeddings, "Call set_classes() first"

        ids = text_ids.cpu().numpy() if isinstance(text_ids, torch.Tensor) else np.asarray(text_ids)

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

        # One decoder call per class, concatenated. The decoder batch is static at one.
        results = [
            self._run_decoder(
                img_out=backbone_out,
                prompt_features=feats_all[:, [i], :],  # [32, 1, 256]
                prompt_mask=masks_all[[i]],  # [1, 32]
                input_boxes=boxes_per_call,
                input_boxes_labels=labels_per_call,
            )
            for i in ids
        ]
        if len(results) == 1:
            return results[0]
        return {k: torch.cat([r[k] for r in results], dim=0) for k in results[0]}

    def set_imgsz(self, imgsz) -> None:
        """Warn when the requested size is one the graphs cannot serve.

        A fixed export accepts a single size and a dynamic one a range, and asking either for anything else fails far
        from here, as an unreadable TensorRT profile error or a silently wrong result.

        Args:
            imgsz (int | list[int]): Size the predictor intends to letterbox to.
        """
        want = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
        if not want:
            return
        fixed = self.imgsz and want != self.imgsz
        outside = self.imgsz_range and not self.imgsz_range[0] <= want <= self.imgsz_range[1]
        if fixed or outside:
            LOGGER.warning(
                f"SAM3Backend: this export only accepts {self._accepted_imgsz()}, got {want}. "
                f"Pass a size it accepts, or re-export covering {want}."
            )

    def eval(self):
        """Return self, the exported graphs are always in inference mode."""
        return self

    def to(self, device):
        """Move the loaded modules to ``device`` and return self.

        ONNX Runtime picks its execution provider when a session is created, so moving between CPU
        and CUDA has to rebuild the sessions. Without that a caller asking for CUDA keeps running the
        graphs on CPU. TensorRT engines and their CUDA stream are bound to the device they were
        deserialized on, so a device change is ignored for them.

        Args:
            device (torch.device | str): Device to run on and place outputs on.
        """
        device = torch.device(device) if isinstance(device, str) else device
        if self._format == "engine" and device != self.device:
            LOGGER.warning(f"SAM3Backend: TensorRT engines stay on {self.device}, ignoring .to({device})")
            return self
        if self._loaded and self._format == "onnx" and device.type != self.device.type:
            self._loaded = False  # sessions are bound to a provider, so reload for the new one
        self.device = device
        return self

    def half(self):
        """Request half precision and return self."""
        self.fp16 = True
        return self

    def float(self):
        """Request full precision and return self."""
        self.fp16 = False
        return self

    def parameters(self):
        """Return an empty iterator, the exported graphs expose no torch parameters."""
        return iter([])

    def __repr__(self) -> str:
        """Return a summary of the loaded directory, format, device, and class names."""
        return f"SAM3Backend(format={self._format!r}, dir={str(self._model_dir)!r}, device={self.device}, names={self.names})"
