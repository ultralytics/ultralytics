# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors, save_one_box
from ultralytics.utils.torch_utils import parse_device, select_device, smart_inference_mode


class TorchVisionVideoClassifier:
    """Video classifier using pretrained TorchVision models for action recognition.

    This class provides an interface for video classification using pretrained models from TorchVision's video model
    collection, supporting S3D, R3D, Swin3D, and MViT architectures trained on Kinetics-400.

    Attributes:
        model (torch.nn.Module): The loaded TorchVision model.
        categories (list): The Kinetics-400 action category names.
        device (torch.device): The device on which the model is loaded.
    """

    _model_specs = {
        "s3d": ("s3d", "S3D_Weights"),
        "r3d_18": ("r3d_18", "R3D_18_Weights"),
        "swin3d_t": ("swin3d_t", "Swin3D_T_Weights"),
        "swin3d_b": ("swin3d_b", "Swin3D_B_Weights"),
        "mvit_v1_b": ("mvit_v1_b", "MViT_V1_B_Weights"),
        "mvit_v2_s": ("mvit_v2_s", "MViT_V2_S_Weights"),
    }

    @classmethod
    def available_models(cls) -> list[str]:
        """Return action-recognition models supported by the installed TorchVision build."""
        try:
            import torchvision.models.video as video_models
        except ImportError:
            return []

        return [
            name
            for name, (fn_name, weights_name) in cls._model_specs.items()
            if getattr(video_models, fn_name, None) is not None
            and getattr(video_models, weights_name, None) is not None
        ]

    def __init__(self, model_name: str = "s3d", device: str = ""):
        """Initialize the TorchVisionVideoClassifier with the specified model."""
        try:
            import torchvision.models.video as video_models
        except ImportError as e:
            raise ImportError("torchvision is required. Install it with: pip install torchvision") from e

        available = self.available_models()
        if not available:
            raise ImportError(
                "ActionRecognition requires a torchvision build with pretrained video model weights. "
                "Install a recent torchvision release and try again."
            )
        if model_name not in available:
            raise ValueError(f"Invalid model '{model_name}'. Choose from: {available}")

        fn_name, weights_name = self._model_specs[model_name]
        model_fn = getattr(video_models, fn_name)
        weights = getattr(video_models, weights_name).DEFAULT
        self.device = select_device(device)
        self.model = model_fn(weights=weights).to(self.device).eval()
        self.categories = weights.meta["categories"]
        self.min_temporal_size = weights.meta.get("min_temporal_size") or 1
        self.transform = weights.transforms()
        crop_size = self.transform.crop_size
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.frame_size = tuple(int(x) for x in crop_size[::-1])  # OpenCV uses (width, height)

    def preprocess(self, crops: list) -> torch.Tensor:
        """Preprocess video crops for classification."""
        # TorchVision video transforms expect a clip in (T, C, H, W) RGB layout.
        clip = torch.stack([torch.from_numpy(c[..., ::-1].copy()).permute(2, 0, 1) for c in crops])
        return self.transform(clip).unsqueeze(0).to(self.device)

    @smart_inference_mode()
    def __call__(self, sequences: torch.Tensor) -> tuple:
        """Run inference and return predicted labels and confidences."""
        output = self.model(sequences)

        labels, confs = [], []
        for out in output:
            prob = out.softmax(0)
            idx = prob.argmax().item()
            labels.append(self.categories[idx])
            confs.append(prob[idx].item())
        return labels, confs


class ActionRecognition(BaseSolution):
    """Real-time action recognition by combining YOLO tracking with video classification.

    This class extends BaseSolution to track persons and classify their actions using TorchVision video models
    pretrained on Kinetics-400.

    Attributes:
        video_classifier (TorchVisionVideoClassifier | None): The video classification model, loaded on first use.
        crop_history (dict): Per-track frame crop history for classification.
        pred_labels (dict): Per-track predicted action labels.
        pred_confs (dict): Per-track predicted confidences.

    Examples:
        >>> action = ActionRecognition(model="yolo26n.pt", video_classifier_model="s3d")
        >>> frame = cv2.imread("frame.jpg")
        >>> results = action(frame)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ActionRecognition with YOLO model and video classifier.

        Args:
            **kwargs: Keyword arguments including:
                - model (str): YOLO model for person detection.
                - video_classifier_model (str): TorchVision model name.
                - crop_margin_percentage (int): Margin for cropping.
                - num_video_sequence_samples (int): Frames per classification sequence.
                - skip_frame (int): Frame skip interval.
        """
        if "classes" not in kwargs:
            kwargs["classes"] = [0]  # Default to person class
        super().__init__(**kwargs)

        self.crop_margin_percentage = int(self.CFG["crop_margin_percentage"])
        self.num_video_sequence_samples = int(self.CFG["num_video_sequence_samples"])
        self.skip_frame = max(1, int(self.CFG["skip_frame"]))
        self.video_cls_overlap_ratio = float(self.CFG["video_cls_overlap_ratio"])

        # Resolve once so an idle-GPU request ("-1") cannot land the tracker and the classifier on different GPUs
        self.track_add_args["device"] = self.device = parse_device(self.CFG["device"])
        self.video_classifier = None

        self.crop_history = defaultdict(list)
        self.frame_counter = 0
        self.pred_labels = {}
        self.pred_confs = {}

    def forget_tracks(self, track_ids: list[int]) -> None:
        """Drop retired IDs from action bookkeeping so it doesn't grow across a 24/7 stream (see BaseSolution)."""
        super().forget_tracks(track_ids)
        for track_id in track_ids:
            self.crop_history.pop(track_id, None)
            self.pred_labels.pop(track_id, None)
            self.pred_confs.pop(track_id, None)

    def process(self, im0: np.ndarray) -> SolutionResults:
        """Process a frame to detect, track, and recognize actions.

        Args:
            im0 (np.ndarray): Input image.

        Returns:
            SolutionResults: Contains plot_im, total_tracks, action_labels (dict mapping track_id to predicted action
                label), and action_confs (dict mapping track_id to confidence).
        """
        # Lazy-load the video classifier on first use
        if self.video_classifier is None:
            self.video_classifier = TorchVisionVideoClassifier(self.CFG["video_classifier_model"], self.device)
            self.num_video_sequence_samples = max(
                self.num_video_sequence_samples, self.video_classifier.min_temporal_size
            )

        self.frame_counter += 1
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        tracks_to_infer = []

        if len(self.boxes):
            for box, track_id in zip(self.boxes, self.track_ids):
                if self.frame_counter % self.skip_frame == 0:
                    gain = 1 + 2 * self.crop_margin_percentage / 100
                    crop = save_one_box(
                        self.get_enclosing_box(box), im0, gain=gain, pad=0, square=True, save=False, BGR=True
                    )
                    if crop.size:
                        self.crop_history[track_id].append(cv2.resize(crop, self.video_classifier.frame_size))

                if len(self.crop_history[track_id]) > self.num_video_sequence_samples:
                    self.crop_history[track_id].pop(0)

                if (
                    len(self.crop_history[track_id]) == self.num_video_sequence_samples
                    and self.frame_counter % self.skip_frame == 0
                ):
                    tracks_to_infer.append(track_id)

            if tracks_to_infer:
                # Crops are collected once every skip_frame frames, so the interval is counted in collected samples
                interval = max(1, int(self.num_video_sequence_samples * (1 - self.video_cls_overlap_ratio)))
                if not self.pred_labels or (self.frame_counter // self.skip_frame) % interval == 0:
                    batch = torch.cat([self.video_classifier.preprocess(self.crop_history[t]) for t in tracks_to_infer])
                    labels, confs = self.video_classifier(batch)
                    for tid, lbl, c in zip(tracks_to_infer, labels, confs):
                        self.pred_labels[tid] = lbl
                        self.pred_confs[tid] = c

            for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
                known = track_id in self.pred_labels
                action = ""
                if self.show_labels:
                    action = self.pred_labels[track_id] if known else "detecting..."
                    if known and self.show_conf:
                        action += f" ({self.pred_confs[track_id]:.2f})"
                label = " | ".join(x for x in (self.adjust_box_label(cls, conf, track_id), action) if x)
                annotator.box_label(box, label, color=colors(track_id, True) if known else (128, 128, 128))

        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            total_tracks=len(self.track_ids) if self.track_ids else 0,
            action_labels=dict(self.pred_labels),
            action_confs=dict(self.pred_confs),
        )
