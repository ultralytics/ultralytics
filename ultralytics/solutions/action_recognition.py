# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.torch_utils import select_device


class TorchVisionVideoClassifier:
    """Video classifier using pretrained TorchVision models for action recognition.

    This class provides an interface for video classification using pretrained models from TorchVision's video model
    collection, supporting S3D, R3D, Swin3D, and MViT architectures trained on Kinetics-400.

    Attributes:
        model (torch.nn.Module): The loaded TorchVision model.
        weights: The pretrained weights used for the model.
        device (torch.device): The device on which the model is loaded.
    """

    _model_names = ["s3d", "r3d_18", "swin3d_t", "swin3d_b", "mvit_v1_b", "mvit_v2_s"]

    def __init__(self, model_name: str = "s3d", device: str = ""):
        """Initialize the TorchVisionVideoClassifier with the specified model."""
        if model_name not in self._model_names:
            raise ValueError(f"Invalid model '{model_name}'. Choose from: {self._model_names}")

        try:
            from torchvision.models.video import (
                MViT_V1_B_Weights,
                MViT_V2_S_Weights,
                R3D_18_Weights,
                S3D_Weights,
                Swin3D_B_Weights,
                Swin3D_T_Weights,
                mvit_v1_b,
                mvit_v2_s,
                r3d_18,
                s3d,
                swin3d_b,
                swin3d_t,
            )
        except ImportError:
            raise ImportError("torchvision is required. Install it with: pip install torchvision")

        model_map = {
            "s3d": (s3d, S3D_Weights.DEFAULT),
            "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
            "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
            "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
            "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
            "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
        }

        model_fn, self.weights = model_map[model_name]
        self.device = select_device(device)
        self.model = model_fn(weights=self.weights).to(self.device).eval()

    def preprocess(self, crops: list) -> torch.Tensor:
        """Preprocess video crops for classification."""
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize([224, 224], antialias=True),
                v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
            ]
        )
        frames = [transform(torch.from_numpy(c[..., ::-1].copy()).permute(2, 0, 1)) for c in crops]  # BGR to RGB
        return torch.stack(frames).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, sequences: torch.Tensor) -> tuple:
        """Run inference and return predicted labels and confidences."""
        with torch.inference_mode():
            output = self.model(sequences)

        labels, confs = [], []
        for out in output:
            prob = out.softmax(0)
            idx = prob.argmax().item()
            labels.append(self.weights.meta["categories"][idx])
            confs.append(prob[idx].item())
        return labels, confs


def crop_and_pad(frame: np.ndarray, box: list, margin_percent: int) -> np.ndarray:
    """Crop a square region around the box with margin."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(frame.shape[1], x2 + mx), min(frame.shape[0], y2 + my)

    size = max(y2 - y1, x2 - x1)
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    half = size // 2

    crop = frame[max(0, cy - half) : min(frame.shape[0], cy + half), max(0, cx - half) : min(frame.shape[1], cx + half)]
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)


class ActionRecognition(BaseSolution):
    """Real-time action recognition by combining YOLO tracking with video classification.

    This class extends BaseSolution to track persons and classify their actions using TorchVision video models
    pretrained on Kinetics-400.

    Attributes:
        video_classifier (TorchVisionVideoClassifier): The video classification model.
        track_history (dict): Per-track frame history for classification.
        pred_labels (dict): Per-track predicted action labels.
        pred_confs (dict): Per-track predicted confidences.

    Examples:
        >>> action = ActionRecognition(model="yolo26n.pt", video_classifier_model="s3d")
        >>> results = action("path/to/video.mp4")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ActionRecognition with YOLO model and video classifier.

        Args:
            **kwargs: Keyword arguments including:
                - model (str): YOLO model for person detection. Defaults to "yolo26n.pt".
                - video_classifier_model (str): TorchVision model name. Defaults to "s3d".
                - crop_margin_percentage (int): Margin for cropping. Defaults to 10.
                - num_video_sequence_samples (int): Frames per sequence. Defaults to 8.
                - skip_frame (int): Frame skip interval. Defaults to 2.
        """
        if "classes" not in kwargs:
            kwargs["classes"] = [0]  # Default to person class
        kwargs["model"] = kwargs.get("model", "yolo26n.pt")
        super().__init__(**kwargs)

        self.crop_margin_percentage = int(self.CFG.get("crop_margin_percentage", 10))
        self.num_video_sequence_samples = int(self.CFG.get("num_video_sequence_samples", 8))
        self.skip_frame = max(1, int(self.CFG.get("skip_frame", 2)))
        self.video_cls_overlap_ratio = float(self.CFG.get("video_cls_overlap_ratio", 0.25))

        video_classifier_model = self.CFG.get("video_classifier_model", "s3d")
        self.video_classifier = TorchVisionVideoClassifier(video_classifier_model, device=self.CFG.get("device", ""))

        self.track_history = defaultdict(list)
        self.frame_counter = 0
        self.pred_labels = {}
        self.pred_confs = {}

    def process(self, im0: np.ndarray) -> SolutionResults:
        """Process a frame to detect, track, and recognize actions.

        Args:
            im0 (np.ndarray): Input image.

        Returns:
            SolutionResults: Contains plot_im, total_tracks, action_labels (dict mapping track_id to
                predicted action label), and action_confs (dict mapping track_id to confidence).
        """
        self.frame_counter += 1
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        tracks_to_infer, crops_to_infer = [], []

        if len(self.boxes):
            for box, track_id in zip(self.boxes, self.track_ids):
                if self.frame_counter % self.skip_frame == 0:
                    self.track_history[track_id].append(crop_and_pad(im0, box, self.crop_margin_percentage))

                if len(self.track_history[track_id]) > self.num_video_sequence_samples:
                    self.track_history[track_id].pop(0)

                if (
                    len(self.track_history[track_id]) == self.num_video_sequence_samples
                    and self.frame_counter % self.skip_frame == 0
                ):
                    crops_to_infer.append(self.video_classifier.preprocess(self.track_history[track_id]))
                    tracks_to_infer.append(track_id)

            if crops_to_infer:
                interval = int(self.num_video_sequence_samples * self.skip_frame * (1 - self.video_cls_overlap_ratio))
                if not self.pred_labels or self.frame_counter % max(1, interval) == 0:
                    batch = torch.cat(crops_to_infer, dim=0)
                    labels, confs = self.video_classifier(batch)
                    for tid, lbl, conf in zip(tracks_to_infer, labels, confs):
                        self.pred_labels[tid] = lbl
                        self.pred_confs[tid] = conf

            for box, track_id in zip(self.boxes, self.track_ids):
                if track_id in self.pred_labels:
                    label = f"{self.pred_labels[track_id]} ({self.pred_confs[track_id]:.2f})"
                    annotator.box_label(box, label, color=(0, 255, 0))
                else:
                    annotator.box_label(box, "detecting...", color=(128, 128, 128))

        plot_im = annotator.result()
        self.display_output(plot_im)

        # Clean up lost tracks
        current = set(self.track_ids) if self.track_ids is not None else set()
        for tid in set(self.track_history.keys()) - current:
            self.track_history.pop(tid, None)
            self.pred_labels.pop(tid, None)
            self.pred_confs.pop(tid, None)

        return SolutionResults(
            plot_im=plot_im,
            total_tracks=len(self.track_ids) if self.track_ids else 0,
            action_labels=dict(self.pred_labels),
            action_confs=dict(self.pred_confs),
        )
