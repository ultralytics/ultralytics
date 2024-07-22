# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np
import torch

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.torch_utils import select_device


class ActionRecognition:
    """A class to recognize actions in a real-time video stream based on object tracks."""

    def __init__(
        self,
        video_classifier_model="microsoft/xclip-base-patch32",
        labels=None,
        fp16=False,
        crop_margin_percentage=10,
        num_video_sequence_samples=8,
        skip_frame=2,
        video_cls_overlap_ratio=0.25,
        device="",
    ):
        """
        Initializes the ActionRecognition with the given parameters.

        Args:
            video_classifier_model (str): Name or path of the video classifier model. Defaults to "microsoft/xclip-base-patch32".
            labels (List[str], optional): List of labels for zero-shot classification. Defaults to predefined list.
            fp16 (bool, optional): Whether to use half-precision floating point. Defaults to False.
            crop_margin_percentage (int, optional): Percentage of margin to add around detected objects. Defaults to 10.
            num_video_sequence_samples (int, optional): Number of video frames to use for classification. Defaults to 8.
            skip_frame (int, optional): Number of frames to skip between detections. Defaults to 2.
            video_cls_overlap_ratio (float, optional): Overlap ratio between video sequences. Defaults to 0.25.
            device (str or torch.device, optional): The device to run the model on. Defaults to "".
        """
        self.labels = (
            labels
            if labels is not None
            else ["walking", "running", "brushing teeth", "looking into phone", "weight lifting", "cooking", "sitting"]
        )
        self.fp16 = fp16
        self.device = select_device(device)

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)
        self.window_name = "Ultralytics YOLOv8 Action Recognition"

        if video_classifier_model in TorchVisionVideoClassifier.available_model_names():
            print("'fp16' is not supported for TorchVisionVideoClassifier. Setting fp16 to False.")
            print(
                "'labels' is not used for TorchVisionVideoClassifier. Ignoring the provided labels and using Kinetics-400 labels."
            )
            self.video_classifier = TorchVisionVideoClassifier(video_classifier_model, device=self.device)
        else:
            self.video_classifier = HuggingFaceVideoClassifier(
                self.labels, model_name=video_classifier_model, device=self.device, fp16=fp16
            )

        self.track_history = defaultdict(list)
        self.annotator = None
        self.frame_counter = 0

        # Properties with default values
        self.crop_margin_percentage = crop_margin_percentage
        self.num_video_sequence_samples = num_video_sequence_samples
        self.skip_frame = skip_frame
        self.video_cls_overlap_ratio = video_cls_overlap_ratio

    def process_tracks(self, tracks):
        """
        Extracts results from the provided tracking data and stores track information.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu().numpy()
        self.track_ids = tracks[0].boxes.id.cpu().numpy()

        for box, track_id in zip(self.boxes, self.track_ids):
            crop = self.crop_and_pad(self.im0, box, self.crop_margin_percentage)
            self.track_history[track_id].append(crop)

            if len(self.track_history[track_id]) > self.num_video_sequence_samples:
                self.track_history[track_id].pop(0)

    def plot_box_and_action(self, box, pred_label, pred_conf):
        """
        Plots track and bounding box with action label.

        Args:
            box (list): Object bounding box data.
            pred_label (str): Predicted action label.
            pred_conf (float): Confidence of the predicted action.
        """
        label_text = f"{pred_label} ({pred_conf:.2f})"
        self.annotator.box_label(box, label_text, color=(0, 0, 255))

    def display_frames(self):
        """Displays the current frame."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def predict_action(self, sequences):
        """
        Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model. Batched video frames with shape (B, T, H, W, C).

        Returns:
            torch.Tensor: The model's output.
        """
        return self.video_classifier(sequences)

    def postprocess(self, outputs):
        """
        Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output.

        Returns:
            List[List[str]]: The predicted top3 labels.
            List[List[float]]: The predicted top3 confidences.
        """
        pred_labels = []
        pred_confs = []

        with torch.no_grad():
            logits_per_video = outputs
            probs = logits_per_video.softmax(dim=-1)

        for prob in probs:
            top2_indices = prob.topk(2).indices.tolist()
            top2_labels = [self.labels[idx] for idx in top2_indices]
            top2_confs = prob[top2_indices].tolist()
            pred_labels.append(top2_labels)
            pred_confs.append(top2_confs)

        return pred_labels, pred_confs

    def recognize_actions(self, im0, tracks):
        """
        Recognizes actions based on tracking data.

        Args:
            im0 (ndarray): Image.
            tracks (list): List of tracks obtained from the object tracking process.

        Returns:
            (ndarray): The image with annotated boxes and tracks.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            return im0

        self.annotator = Annotator(im0, line_width=3, font_size=10, pil=False)
        self.frame_counter += 1
        track_ids_to_infer = []
        crops_to_infer = []
        pred_labels = []
        pred_confs = []

        self.process_tracks(tracks)

        if self.frame_counter % self.skip_frame == 0:
            crops_to_infer = []
            track_ids_to_infer = []

        for box, track_id in zip(self.boxes, self.track_ids):
            if (
                len(self.track_history[track_id]) == self.num_video_sequence_samples
                and self.frame_counter % self.skip_frame == 0
            ):
                crops = self.video_classifier.preprocess_crops_for_video_cls(self.track_history[track_id])
                crops_to_infer.append(crops)
                track_ids_to_infer.append(track_id)

        if crops_to_infer and (
            not pred_labels
            or self.frame_counter
            % int(self.num_video_sequence_samples * self.skip_frame * (1 - self.video_cls_overlap_ratio))
            == 0
        ):
            crops_batch = torch.cat(crops_to_infer, dim=0)
            output_batch = self.predict_action(crops_batch)
            pred_labels, pred_confs = self.postprocess(output_batch)

        if track_ids_to_infer and crops_to_infer:
            for box, track_id, pred_label, pred_conf in zip(self.boxes, track_ids_to_infer, pred_labels, pred_confs):
                top2_preds = sorted(zip(pred_label, pred_conf), key=lambda x: x[1], reverse=True)
                label_text = " | ".join([f"{label} ({conf:.2f})" for label, conf in top2_preds])
                self.plot_box_and_action(box, track_id, label_text, pred_conf)

        return im0

    def crop_and_pad(self, frame, box, margin_percent):
        """Crop box with margin and take square crop from frame."""
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1

        margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
        x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

        size = max(y2 - y1, x2 - x1)
        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
        half_size = size // 2
        square_crop = frame[
            max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
            max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
        ]

        return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


class TorchVisionVideoClassifier:
    """Classifies videos using pretrained TorchVision models; see https://pytorch.org/vision/stable/."""

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

    model_name_to_model_and_weights = {
        "s3d": (s3d, S3D_Weights.DEFAULT),
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
        "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
    }

    def __init__(self, model_name: str, device: str or torch.device = ""):
        """
        Initialize the VideoClassifier with the specified model name and device.

        Args:
            model_name (str): The name of the model to use.
            device (str or torch.device, optional): The device to run the model on. Defaults to "".

        Raises:
            ValueError: If an invalid model name is provided.
        """
        if model_name not in self.model_name_to_model_and_weights:
            raise ValueError(f"Invalid model name '{model_name}'. Available models: {self.available_model_names()}")
        model, self.weights = self.model_name_to_model_and_weights[model_name]
        self.device = select_device(device)
        self.model = model(weights=self.weights).to(self.device).eval()

    @staticmethod
    def available_model_names() -> List[str]:
        """
        Get the list of available model names.

        Returns:
            list: List of available model names.
        """
        return list(TorchVisionVideoClassifier.model_name_to_model_and_weights.keys())

    def preprocess_crops_for_video_cls(self, crops: List[np.ndarray], input_size: list = None) -> torch.Tensor:
        """
        Preprocess a list of crops for video classification.

        Args:
            crops (List[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C)
            input_size (tuple, optional): The target input size for the model. Defaults to (224, 224).

        Returns:
            torch.Tensor: Preprocessed crops as a tensor with dimensions (1, T, C, H, W).
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(input_size, antialias=True),
                v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
        return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, sequences: torch.Tensor):
        """
        Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model. The expected input dimensions are
                                      (B, T, C, H, W) for batched video frames or (T, C, H, W) for single video frames.

        Returns:
            torch.Tensor: The model's output.
        """
        with torch.inference_mode():
            return self.model(sequences)

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[str], List[float]]:
        """
        Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output.

        Returns:
            List[str]: The predicted labels.
            List[float]: The predicted confidences.
        """
        pred_labels = []
        pred_confs = []
        for output in outputs:
            pred_class = output.argmax(0).item()
            pred_label = self.weights.meta["categories"][pred_class]
            pred_labels.append(pred_label)
            pred_conf = output.softmax(0)[pred_class].item()
            pred_confs.append(pred_conf)

        return pred_labels, pred_confs


class HuggingFaceVideoClassifier:
    """Zero-shot video classifier using Hugging Face models for various devices."""

    def __init__(
        self,
        labels: List[str],
        model_name: str = "microsoft/xclip-base-patch16-zero-shot",
        device: str or torch.device = "",
        fp16: bool = False,
    ):
        """
        Initialize the HuggingFaceVideoClassifier with the specified model name.

        Args:
            labels (List[str]): List of labels for zero-shot classification.
            model_name (str): The name of the model to use. Defaults to "microsoft/xclip-base-patch16-zero-shot".
            device (str or torch.device, optional): The device to run the model on. Defaults to "".
            fp16 (bool, optional): Whether to use FP16 for inference. Defaults to False.
        """
        from transformers import AutoModel, AutoProcessor

        self.fp16 = fp16
        self.labels = labels
        self.device = select_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        if fp16:
            model = model.half()
        self.model = model.eval()

    def preprocess_crops_for_video_cls(self, crops: List[np.ndarray], input_size: list = None) -> torch.Tensor:
        """
        Preprocess a list of crops for video classification.

        Args:
            crops (List[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C)
            input_size (tuple, optional): The target input size for the model. Defaults to (224, 224).

        Returns:
            torch.Tensor: Preprocessed crops as a tensor (1, T, C, H, W).
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize(input_size),
                transforms.Normalize(
                    mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std
                ),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]  # (T, C, H, W)
        output = torch.stack(processed_crops).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        if self.fp16:
            output = output.half()
        return output

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model. Batched video frames with shape (B, T, H, W, C).

        Returns:
            torch.Tensor: The model's output.
        """

        input_ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)

        inputs = {"pixel_values": sequences, "input_ids": input_ids}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        return outputs.logits_per_video

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output.

        Returns:
            List[List[str]]: The predicted top3 labels.
            List[List[float]]: The predicted top3 confidences.
        """
        pred_labels = []
        pred_confs = []

        with torch.no_grad():
            logits_per_video = outputs  # Assuming outputs is already the logits tensor
            probs = logits_per_video.softmax(dim=-1)  # Use softmax to convert logits to probabilities

        for prob in probs:
            top2_indices = prob.topk(2).indices.tolist()
            top2_labels = [self.labels[idx] for idx in top2_indices]
            top2_confs = prob[top2_indices].tolist()
            pred_labels.append(top2_labels)
            pred_confs.append(top2_confs)

        return pred_labels, pred_confs


def crop_and_pad(frame, box, margin_percent):
    """Crop box with margin and take square crop from frame."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # Add margin
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # Take square crop from frame
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    square_crop = frame[
        max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
        max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    ]

    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


if __name__ == "__main__":
    device = ""
    video_classifier_model = "microsoft/xclip-base-patch32"
    labels = ["walking", "running", "brushing teeth", "looking into phone", "weight lifting", "cooking", "sitting"]
    action_recognition = ActionRecognition(video_classifier_model=video_classifier_model, labels=labels, device=device)
