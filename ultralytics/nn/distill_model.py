# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Teacher-student feature distillation for Ultralytics transformer-decoder detectors.

Adapted from the `exp-main-distill-clean` branch (YOLO Detect head) for RT-DETR /
DFINE / DEIM transformer decoders. Forward hooks capture FPN/PAN features at the
indices feeding the decoder; a per-level 1x1-conv projector aligns student channels
to teacher channels, and a score-weighted L2 loss is computed between them. The
per-pixel score is the teacher's own channel-L2 feature magnitude, normalized per
image to [0, 1] — a class-agnostic spatial saliency. This replaces YOLO's
classification-score weighting since the transformer decoder produces only
per-query logits, not a dense per-pixel score map.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.utils.torch_utils import copy_attr

from .tasks import load_checkpoint


class FeatureHook:
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict, idx):
        """Store a reference to the shared dict and the layer index this hook writes to."""
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module, input, output):
        """Write the layer output into the shared dict keyed by layer index."""
        self.feat_dict[self.idx] = output


class InputHook:
    """Picklable forward pre-hook that stores a module's input tensor into a shared dict.

    Used to capture the query-feature tensor entering a decoder prediction head, since that input is exactly the
    per-query hidden state before classification — needed for `query_feat` KD.
    """

    def __init__(self, feat_dict, key):
        """Store a reference to the shared dict and the key this hook writes to."""
        self.feat_dict = feat_dict
        self.key = key

    def __call__(self, module, args):
        """Write the first positional input into the shared dict keyed by the configured key."""
        self.feat_dict[self.key] = args[0]


class DistillationModel(nn.Module):
    """Teacher-student knowledge distillation for RT-DETR / DFINE / DEIM detectors.

    Wraps a frozen teacher and a trainable student. Forward hooks capture features at the FPN/PAN layers feeding the
    transformer decoder (`RTDETRDecoder.f`). A per-level projector aligns student channels to teacher channels, and the
    loss is a score-weighted L2 between projected student and teacher features, summed over levels and scaled by `dis`.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        student_feats_idx (list[int]): Student FPN/PAN layer indices feeding the transformer decoder.
        teacher_feats_idx (list[int]): Teacher FPN/PAN layer indices feeding the transformer decoder (may differ from
            `student_feats_idx` for cross-architecture distillation; paired with student levels by position).
        projector (nn.ModuleList): 1x1-conv-ReLU-1x1-conv projector per feature level.
        dis (float): Distillation loss weight factor.
    """

    def __init__(self, teacher_model: str | nn.Module, student_model: nn.Module):
        """Build the distillation wrapper around a teacher checkpoint and an initialized student.

        Args:
            teacher_model (str | nn.Module): Teacher checkpoint path or already-loaded module.
            student_model (nn.Module): Student module to be trained.
        """
        super().__init__()
        if isinstance(teacher_model, str):
            teacher_model = load_checkpoint(teacher_model)[0]
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.student_feats_idx = self.get_distill_layers(student_model)
        self.teacher_feats_idx = self.get_distill_layers(self.teacher_model)
        if len(self.student_feats_idx) != len(self.teacher_feats_idx):
            raise ValueError(
                f"Student decoder consumes {len(self.student_feats_idx)} FPN levels "
                f"({self.student_feats_idx}) but teacher consumes {len(self.teacher_feats_idx)} "
                f"({self.teacher_feats_idx}); per-level KD needs matching level counts."
            )
        self._parse_distill_config(student_model.args)
        self._init_feature_dicts()
        self._register_neck_hooks()
        self._register_saliency_hooks()
        self._register_decoder_hooks()
        teacher_output, student_output = self._dummy_forward(student_model.args.imgsz, device)
        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis
        self.projector = self._build_neck_projectors(student_output, teacher_output, device)
        self.query_projector = self._build_query_projector(device)
        self.query_matcher = self._build_query_matcher()

    def __getstate__(self):
        """Return a clean copy of state for pickling without hooks and extracted features."""
        state = self.__dict__.copy()
        for k in self._FEAT_DICT_KEYS:
            state[k] = {}
        return state

    def __setstate__(self, state):
        """Clear stale features and hooks, then re-register forward hooks after unpickling."""
        self.__dict__.update(state)
        self._init_feature_dicts()
        self._clear_distill_hooks()
        self._register_neck_hooks()
        self._register_saliency_hooks()
        self._register_decoder_hooks()

    @staticmethod
    def get_distill_layers(model):
        """Return the FPN/PAN layer indices feeding the transformer decoder.

        Raises ValueError if the model does not have a transformer-decoder detection head.
        """
        for m in model.model:
            if isinstance(m, RTDETRDecoder):  # DFineDecoder / DeimDecoder are subclasses
                return list(m.f)
        raise ValueError("No transformer-decoder head (RTDETRDecoder family) found in model")

    def _freeze_teacher(self):
        """Disable gradients and set the teacher to eval mode for distillation."""
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    _FEAT_DICT_KEYS = (
        "_teacher_feats",
        "_student_feats",
        "_enc_logits",
        "_student_enc_logits",
        "_teacher_dec_out",
        "_student_dec_out",
        "_teacher_qf",
        "_student_qf",
    )

    def _parse_distill_config(self, student_args):
        """Read and validate the distillation knobs from `student_model.args`.

        Sets `self.distill_saliency`, `self.distill_decoder`, `self.dis_dec`. Validates allowed
        values up-front so a typo is caught at wrapper construction rather than mid-training.
        """
        self.distill_saliency = getattr(student_args, "distill_saliency", "enc_score")
        if self.distill_saliency not in {"feat_norm", "enc_score"}:
            raise ValueError(
                f"distill_saliency={self.distill_saliency!r} not supported. Choose 'feat_norm' or 'enc_score'."
            )
        self.distill_decoder = getattr(student_args, "distill_decoder", "none")
        if self.distill_decoder not in {"none", "objectness", "query_feat", "all"}:
            raise ValueError(
                f"distill_decoder={self.distill_decoder!r} not supported. "
                f"Choose 'none', 'objectness', 'query_feat', or 'all'."
            )
        self.dis_dec = getattr(student_args, "dis_dec", 0.0)

    def _init_feature_dicts(self):
        """Create empty dicts that hooks write into. Called both at construction and after unpickling."""
        for k in self._FEAT_DICT_KEYS:
            setattr(self, k, {})

    def _register_neck_hooks(self):
        """Register forward hooks on the FPN/PAN layers feeding the transformer decoder (teacher + student).

        Teacher and student may consume their FPN levels at different yaml indices (e.g. teacher `[15, 18, 21]`
        vs student `[16, 19, 22]`), so each side is hooked at its own indices and the per-level pairing is by
        position in the lists (which must have equal length).
        """
        for s_idx, t_idx in zip(self.student_feats_idx, self.teacher_feats_idx):
            self.teacher_model.model[t_idx].register_forward_hook(FeatureHook(self._teacher_feats, t_idx))
            self.student_model.model[s_idx].register_forward_hook(FeatureHook(self._student_feats, s_idx))

    def _uses_objectness(self):
        """True when the configured decoder mode needs the teacher/student `enc_score_head` objectness."""
        return self.distill_decoder in {"objectness", "all"}

    def _uses_query_feat(self):
        """True when the configured decoder mode needs the matched per-query hidden states."""
        return self.distill_decoder in {"query_feat", "all"}

    def _register_saliency_hooks(self):
        """Register the teacher `enc_score_head` hook, shared by `enc_score` saliency and any objectness decoder KD.

        The same captured logits feed two distinct loss paths, so one hook is sufficient regardless of which path
        (or both) is active.
        """
        if self.distill_saliency != "enc_score" and not self._uses_objectness():
            return
        decoder = self.teacher_model.model[-1]
        if not hasattr(decoder, "enc_score_head"):
            raise ValueError(
                "distill_saliency='enc_score' or distill_decoder='objectness'/'all' requires the teacher decoder "
                "to expose enc_score_head; the loaded teacher does not (was it built with learnt_init_query=True?)."
            )
        decoder.enc_score_head.register_forward_hook(FeatureHook(self._enc_logits, "enc"))

    def _register_decoder_hooks(self):
        """Register the student-side hooks required by the selected `distill_decoder` mode.

        Composes per-component: the objectness path taps the student `enc_score_head`; the query_feat path taps
        both decoders' output tuples (for box/score-driven Hungarian matching) and adds a forward pre-hook on
        `dec_score_head[eval_idx]` on each side (whose input is the per-query hidden state). `all` enables both
        sets at once. The teacher `enc_score_head` hook is handled separately by `_register_saliency_hooks`.
        """
        if self.distill_decoder == "none":
            return
        teacher_decoder = self.teacher_model.model[-1]
        student_decoder = self.student_model.model[-1]
        if self._uses_objectness():
            if not hasattr(student_decoder, "enc_score_head"):
                raise ValueError(
                    "distill_decoder='objectness'/'all' requires the student decoder to expose enc_score_head."
                )
            student_decoder.enc_score_head.register_forward_hook(FeatureHook(self._student_enc_logits, "enc"))
        if self._uses_query_feat():
            if not hasattr(teacher_decoder, "dec_score_head") or not hasattr(student_decoder, "dec_score_head"):
                raise ValueError(
                    "distill_decoder='query_feat'/'all' requires both teacher and student decoders to expose "
                    "dec_score_head."
                )
            if not hasattr(teacher_decoder, "eval_idx") or not hasattr(student_decoder, "eval_idx"):
                raise ValueError(
                    "distill_decoder='query_feat'/'all' requires both decoders to expose `eval_idx` (available on "
                    "DFineDecoder / DeimDecoder but not base RTDETRDecoder)."
                )
            teacher_decoder.register_forward_hook(FeatureHook(self._teacher_dec_out, "out"))
            student_decoder.register_forward_hook(FeatureHook(self._student_dec_out, "out"))
            teacher_decoder.dec_score_head[teacher_decoder.eval_idx].register_forward_pre_hook(
                InputHook(self._teacher_qf, "qf")
            )
            student_decoder.dec_score_head[student_decoder.eval_idx].register_forward_pre_hook(
                InputHook(self._student_qf, "qf")
            )

    def _clear_distill_hooks(self):
        """Wipe any forward / forward-pre hooks on layers this wrapper attaches to.

        Called by `__setstate__` so re-registration after unpickling never stacks duplicates. Safe to call when the
        relevant submodules do not exist (the `hasattr` guards skip cleanly).
        """
        for s_idx, t_idx in zip(self.student_feats_idx, self.teacher_feats_idx):
            self.teacher_model.model[t_idx]._forward_hooks.clear()
            self.student_model.model[s_idx]._forward_hooks.clear()
        t_dec = self.teacher_model.model[-1]
        s_dec = self.student_model.model[-1]
        t_dec._forward_hooks.clear()
        s_dec._forward_hooks.clear()
        if hasattr(t_dec, "enc_score_head"):
            t_dec.enc_score_head._forward_hooks.clear()
        if hasattr(s_dec, "enc_score_head"):
            s_dec.enc_score_head._forward_hooks.clear()
        if hasattr(t_dec, "dec_score_head") and hasattr(t_dec, "eval_idx"):
            t_dec.dec_score_head[t_dec.eval_idx]._forward_pre_hooks.clear()
        if hasattr(s_dec, "dec_score_head") and hasattr(s_dec, "eval_idx"):
            s_dec.dec_score_head[s_dec.eval_idx]._forward_pre_hooks.clear()

    def _dummy_forward(self, imgsz, device):
        """Run a 2-sample dummy batch through both models to populate hook-captured shapes.

        Returns the per-level teacher and student FPN feature lists so the projectors can be built with the right
        channel dimensions.

        Returns:
            (tuple[list[torch.Tensor], list[torch.Tensor]]): Per-level (teacher_outputs, student_outputs) features.
        """
        with torch.no_grad():
            self.teacher_model(torch.zeros(2, 3, imgsz, imgsz, device=device))
            self.student_model(torch.zeros(2, 3, imgsz, imgsz, device=device))
        teacher_output = [self._teacher_feats[idx] for idx in self.teacher_feats_idx]
        student_output = [self._student_feats[idx] for idx in self.student_feats_idx]
        assert all(t.ndim == 4 and s.ndim == 4 for t, s in zip(teacher_output, student_output)), (
            "Expected 4D FPN feature maps at every hook index for transformer-decoder distillation."
        )
        return teacher_output, student_output

    @staticmethod
    def _build_neck_projectors(student_output, teacher_output, device):
        """Build a `ModuleList` of per-level 1x1-Conv -> ReLU -> 1x1-Conv projectors aligning student to teacher
        channels.
        """
        projectors = []
        for s, t in zip(student_output, teacher_output):
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(s.shape[1], t.shape[1], kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(t.shape[1], t.shape[1], kernel_size=1, stride=1, padding=0),
                )
            )
        return nn.ModuleList(projectors).to(device)

    def _build_query_projector(self, device):
        """Build the student-to-teacher query feature projector used by `query_feat` / `all` modes, or return None.

        Per-query hidden states have different dimensions on different decoders (e.g. 224 vs 256 for DFINE s vs xl);
        the projector aligns them before the L2 comparison on Hungarian-matched query pairs.
        """
        if not self._uses_query_feat():
            return None
        s_hd = self.student_model.model[-1].hidden_dim
        t_hd = self.teacher_model.model[-1].hidden_dim
        return nn.Sequential(
            nn.Linear(s_hd, t_hd),
            nn.ReLU(inplace=True),
            nn.Linear(t_hd, t_hd),
        ).to(device)

    def _build_query_matcher(self):
        """Build the repo `HungarianMatcher` reused for teacher-student query alignment in `query_feat` / `all` modes.

        Default cost gains (class=2, bbox=5, giou=2) — same as RT-DETR training-time matching. Assumes teacher and
        student share the same class space, so the teacher's argmax class can be passed as `gt_cls` and indexed
        into the student's prediction logits without further mapping.
        """
        if not self._uses_query_feat():
            return None
        from ultralytics.models.utils.ops import HungarianMatcher

        return HungarianMatcher()

    def train(self, mode: bool = True):
        """Set train mode while keeping the teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """Dispatch to loss when given a batch dict, otherwise run student prediction."""
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    @staticmethod
    def _feat_norm_saliency(feat: torch.Tensor) -> torch.Tensor:
        """Channel-wise L2 norm of `feat`, normalized per image to [0, 1].

        Class-agnostic saliency derived purely from teacher feature magnitudes. Each spatial
        location is weighted by how strongly the teacher activates there.

        Args:
            feat (torch.Tensor): Teacher feature of shape (N, C, H, W).

        Returns:
            (torch.Tensor): Score tensor of shape (N, 1, H*W).
        """
        n, _, h, w = feat.shape
        mag = feat.pow(2).mean(dim=1, keepdim=True).sqrt()
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-9)
        return mag.view(n, 1, h * w)

    def _enc_score_saliency(self, teacher_outputs: list) -> tuple:
        """Teacher pre-topk encoder objectness as class-conditioned saliency, per FPN level.

        Reads the teacher's RTDETRDecoder.enc_score_head output (captured via hook), applies
        sigmoid + max over the class dimension to collapse to per-spatial-position objectness,
        and splits the flattened (B, total_HW, 1) tensor back into per-level (B, 1, H*W) chunks
        using the spatial sizes of the captured FPN features.

        Args:
            teacher_outputs (list[torch.Tensor]): Per-level teacher features used to derive runtime spatial split sizes
                (length matches teacher_feats_idx).

        Returns:
            (tuple[torch.Tensor, ...]): Per-level score tensors of shape (N, 1, H*W).
        """
        logits = self._enc_logits["enc"]  # (B, total_HW, nc)
        objectness = logits.sigmoid().max(dim=-1, keepdim=True).values  # (B, total_HW, 1)
        objectness = objectness.transpose(1, 2)  # (B, 1, total_HW)
        split_sizes = [t.shape[-2] * t.shape[-1] for t in teacher_outputs]
        parts = torch.split(objectness, split_sizes, dim=-1)
        return tuple(parts)

    def _teacher_scores(self, teacher_outputs: list) -> tuple:
        """Dispatch to the configured saliency source.

        Returns a per-level tuple of score tensors, each of shape (N, 1, H*W), matching the
        order of `teacher_feats_idx`. The selected mode comes from `args.distill_saliency`.
        """
        if self.distill_saliency == "enc_score":
            return self._enc_score_saliency(teacher_outputs)
        return tuple(self._feat_norm_saliency(t) for t in teacher_outputs)

    def _clear_feat_caches(self):
        """Clear every hook-captured cache. Called at the start of each loss() forward."""
        for k in self._FEAT_DICT_KEYS:
            getattr(self, k).clear()

    def _neck_loss(self):
        """Score-weighted L2 loss summed over the FPN levels feeding the decoder, scaled by `self.dis`.

        Pairs student/teacher levels by position so the two sides can use different yaml indices.
        """
        teacher_outputs = [self._teacher_feats[idx] for idx in self.teacher_feats_idx]
        teacher_scores = self._teacher_scores(teacher_outputs)
        total = 0.0
        for i, s_idx in enumerate(self.student_feats_idx):
            teacher_feat = teacher_outputs[i]
            student_feat = self.projector[i](self._student_feats[s_idx])
            total = total + self.loss_sl2(student_feat, teacher_feat, teacher_scores[i])
        return total * self.dis

    @staticmethod
    def _unpack_dec_predictions(out):
        """Return (last-layer dec_bboxes, last-layer dec_scores) handling both train and eval (non-export) outputs.

        Training mode returns the training tuple directly: `(dec_bboxes, dec_scores, ...)`. Eval (non-export) returns
        `(y, x)` where `x` is the same training tuple. We unwrap the eval shell, then index the final decoder layer.
        """
        if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
            out = out[1]
        return out[0][-1], out[1][-1]

    @staticmethod
    def _class_agnostic_objectness(logits):
        """Sigmoid + max over the class dim to produce a 1-dim objectness map tolerant of label-space mismatches."""
        return logits.sigmoid().amax(dim=-1, keepdim=True)

    def _objectness_decoder_loss(self):
        """Dense pre-topk encoder objectness MSE between teacher and student.

        Both sides apply their respective `enc_score_head` to encoder memory; collapsing the class dim with
        sigmoid+max makes the comparison class-agnostic so it tolerates teacher/student label-space mismatches
        (e.g. Obj365 teacher distilling into a COCO student).
        """
        t_obj = self._class_agnostic_objectness(self._enc_logits["enc"])
        s_obj = self._class_agnostic_objectness(self._student_enc_logits["enc"])
        return F.mse_loss(s_obj, t_obj)

    def _hungarian_match(self, t_boxes, s_boxes, t_scores, s_scores):
        """Match teacher queries (as pseudo-GT) to student queries via the repo's `HungarianMatcher`.

        Adapts to the matcher's `(preds, gts)` signature: teacher boxes flatten into `gt_bboxes`, teacher argmax
        class becomes `gt_cls`, and student boxes / logits feed the prediction side. The matcher returns a
        concatenated `gt_idx` per batch; we strip the cumulative batch offset to recover per-batch teacher indices.

        Args:
            t_boxes (torch.Tensor): Teacher boxes (B, Q_t, 4) in cxcywh.
            s_boxes (torch.Tensor): Student boxes (B, Q_s, 4) in cxcywh.
            t_scores (torch.Tensor): Teacher class logits (B, Q_t, nc).
            s_scores (torch.Tensor): Student class logits (B, Q_s, nc).

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): Per-batch (teacher_idx_local, student_idx) tensors.
        """
        B, Q_t = t_boxes.shape[:2]
        gt_boxes = t_boxes.reshape(-1, 4)
        gt_cls = t_scores.argmax(dim=-1).reshape(-1).long()
        gt_groups = [Q_t] * B
        indices = self.query_matcher(s_boxes, s_scores, gt_boxes, gt_cls, gt_groups)
        return [(gt_idx - b * Q_t, pred_idx) for b, (pred_idx, gt_idx) in enumerate(indices)]

    def _query_feat_decoder_loss(self):
        """L2 loss on Hungarian-matched per-query hidden features.

        Matching cost uses teacher final-layer boxes and class-collapsed objectness, so the assignment is robust to
        differing class counts and to extra one-to-many student queries. Matched student features are projected to
        the teacher hidden dim by `self.query_projector` before MSE.
        """
        t_boxes, t_scores = self._unpack_dec_predictions(self._teacher_dec_out["out"])
        s_boxes, s_scores = self._unpack_dec_predictions(self._student_dec_out["out"])
        with torch.no_grad():
            matches = self._hungarian_match(t_boxes, s_boxes, t_scores, s_scores)
        t_qf = self._teacher_qf["qf"]
        s_qf = self._student_qf["qf"]
        t_matched = torch.stack([t_qf[b][matches[b][0]] for b in range(t_qf.shape[0])])
        s_matched = torch.stack([s_qf[b][matches[b][1]] for b in range(s_qf.shape[0])])
        return F.mse_loss(self.query_projector(s_matched), t_matched)

    def _decoder_loss(self):
        """Compose the decoder-level KD losses enabled by `distill_decoder`, scaled by `self.dis_dec`.

        Returns None for `'none'`. For `'objectness'` / `'query_feat'` only that component runs; for `'all'` both
        components are summed before the single `dis_dec` scaling.
        """
        if self.distill_decoder == "none":
            return None
        total = 0.0
        if self._uses_objectness():
            total = total + self._objectness_decoder_loss()
        if self._uses_query_feat():
            total = total + self._query_feat_decoder_loss()
        return total * self.dis_dec

    def loss(self, batch, preds=None):
        """Compute combined detection loss and feature distillation loss.

        Args:
            batch (dict): Batch dict with the input image and detection targets.
            preds (torch.Tensor | list[torch.Tensor], optional): Cached student predictions.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Concatenated (regular_loss, distill_loss) tensors, both raw and
                detached for logging.
        """
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:  # validation while training: skip distillation, keep loss shape
            preds = self.student_model(batch["img"])
            regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
            distill_loss_detach = torch.zeros(1, device=batch["img"].device)
            return self._concat_losses(regular_loss, loss_distill, regular_loss_detach, distill_loss_detach)

        self._clear_feat_caches()
        with torch.no_grad():
            self.teacher_model(batch["img"])
        preds = self.student_model(batch["img"])
        regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)

        loss_distill = loss_distill + self._neck_loss()
        dec_loss = self._decoder_loss()
        if dec_loss is not None:
            loss_distill = loss_distill + dec_loss

        distill_loss_detach = loss_distill.detach()
        return self._concat_losses(regular_loss, loss_distill, regular_loss_detach, distill_loss_detach)

    @staticmethod
    def _concat_losses(regular_loss, loss_distill, regular_loss_detach, distill_loss_detach):
        """Concatenate regular and distillation losses, promoting 0-D scalars to 1-D first.

        DFINE/DEIM `student_model.loss` returns a 0-D scalar total loss for backward and a
        1-D component breakdown for logging; YOLO returns 1-D for both. Promoting any 0-D
        tensor to 1-D normalizes shapes so `torch.cat` accepts either input convention.
        """
        regular_loss = regular_loss.unsqueeze(0) if regular_loss.ndim == 0 else regular_loss
        regular_loss_detach = regular_loss_detach.unsqueeze(0) if regular_loss_detach.ndim == 0 else regular_loss_detach
        return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, distill_loss_detach])

    @staticmethod
    def loss_sl2(student_feat: torch.Tensor, teacher_feat: torch.Tensor, teacher_score: torch.Tensor) -> torch.Tensor:
        """Score-weighted L2 distillation loss for a single feature level.

        Args:
            student_feat (torch.Tensor): Projected student feature of shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature of shape (N, C, H, W).
            teacher_score (torch.Tensor): Per-pixel score of shape (N, 1, H*W).

        Returns:
            (torch.Tensor): Scalar score-weighted L2 loss.
        """
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        return (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)

    @property
    def model(self):
        """Forward .model access to the student's underlying nn.Sequential of yaml-parsed layers.

        Many validators and exporters in the codebase access `model.model[-1]` to read head
        attributes (e.g. one_to_many_groups, reg_max). Property avoids duplicate submodule
        registration while making those code paths transparent to the distillation wrapper.
        """
        return self.student_model.model

    @property
    def criterion(self):
        """Expose the student model's loss criterion."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Forward criterion updates to the student model."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    def fuse(self, verbose: bool = True):
        """Fuse student model layers for inference speedup."""
        self.student_model.fuse(verbose)
        return self
