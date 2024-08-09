# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.utils import crop_and_pad
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.torch_utils import select_device


class ActionRecognition:
    """A class to recognize actions in a real-time video stream based on object tracks."""

    def __init__(
        self,
        video_classifier_model="microsoft/xclip-base-patch32",
        labels: Optional[List[str]] = None,
        fp16: bool = False,
        crop_margin_percentage: int = 10,
        num_video_sequence_samples: int = 8,
        vid_stride: int = 2,
        video_cls_overlap_ratio: float = 0.25,
        device: str or torch.device = "",
    ):
        """
        Initializes the ActionRecognition with the given parameters.

        Args:
            video_classifier_model (str): Name or path of the video classifier model. Defaults to "microsoft/xclip-base-patch32".
            labels (List[str], optional): List of labels for zero-shot classification. Defaults to predefined list.
            fp16 (bool, optional): Whether to use half-precision floating point. Defaults to False.
            crop_margin_percentage (int, optional): Percentage of margin to add around detected objects. Defaults to 10.
            num_video_sequence_samples (int, optional): Number of video frames to use for classification. Defaults to 8.
            vid_stride (int, optional): Number of frames to skip between detections. Defaults to 2.
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
        self.vid_stride = vid_stride
        self.video_cls_overlap_ratio = video_cls_overlap_ratio

    def process_tracks(self, tracks: List[Results]):
        """
        Extracts results from the provided tracking data and stores track information.

        Args:
            tracks (List[Results]): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu().numpy()
        self.track_ids = tracks[0].boxes.id.cpu().numpy()

        for box, track_id in zip(self.boxes, self.track_ids):
            crop = crop_and_pad(self.im0, box, self.crop_margin_percentage)
            self.track_history[track_id].append(crop)

            if len(self.track_history[track_id]) > self.num_video_sequence_samples:
                self.track_history[track_id].pop(0)

    def display_frames(self):
        """Displays the current frame."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output. In the shape (batch_size, num_classes).

        Returns:
            (List[List[str]]): The predicted top3 labels.
            (List[List[float]]): The predicted top3 confidences.
        """
        pred_labels = []
        pred_confs = []

        with torch.no_grad():
            logits_per_video = outputs
            probs = logits_per_video.softmax(dim=-1)

        for prob in probs:
            top3_indices = prob.topk(3).indices.tolist()
            top3_labels = [self.labels[idx] for idx in top3_indices]
            top3_confs = prob[top3_indices].tolist()
            pred_labels.append(top3_labels)
            pred_confs.append(top3_confs)

        return pred_labels, pred_confs

    def recognize_actions(self, im0: np.ndarray, tracks: List[Results]) -> np.ndarray:
        """
        Recognizes actions based on tracking data.

        Args:
            im0 (ndarray): Image.
            tracks (List[Results]): List of tracks obtained from the object tracking process.

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

        if self.frame_counter % self.vid_stride == 0:
            crops_to_infer = []
            track_ids_to_infer = []

        for box, track_id in zip(self.boxes, self.track_ids):
            if (
                len(self.track_history[track_id]) == self.num_video_sequence_samples
                and self.frame_counter % self.vid_stride == 0
            ):
                crops = self.video_classifier.preprocess_crops_for_video_cls(self.track_history[track_id])
                crops_to_infer.append(crops)
                track_ids_to_infer.append(track_id)

        if crops_to_infer and (
            not pred_labels
            or self.frame_counter
            % int(self.num_video_sequence_samples * self.vid_stride * (1 - self.video_cls_overlap_ratio))
            == 0
        ):
            crops_batch = torch.cat(
                crops_to_infer, dim=0
            )  # crops_batch shape: (batch_size, timestep, height, width, channel)
            output_batch = self.video_classifier(crops_batch)
            pred_labels, pred_confs = self.postprocess(output_batch)

        if track_ids_to_infer and crops_to_infer:
            for box, track_id, pred_label, pred_conf in zip(self.boxes, track_ids_to_infer, pred_labels, pred_confs):
                top2_preds = sorted(zip(pred_label, pred_conf), key=lambda x: x[1], reverse=True)
                label_text = " | ".join([f"{label} ({conf:.2f})" for label, conf in top2_preds])
                self.annotator.box_label(box, label_text, color=(0, 0, 255))

        return im0


class TorchVisionVideoClassifier:
    """Classifies videos using pretrained TorchVision models; see https://pytorch.org/vision/stable/models.html#video-classification."""

    supports_r3d = check_requirements("torchvision>=0.13.0", install=False)
    supports_transforms_v2 = check_requirements("torchvision>=0.16.0", install=False)
    supports_mvitv1b = supports_s3d = check_requirements("torchvision>=0.14.0", install=False)
    supports_mvitv2s = supports_swin3dt = supports_swin3db = check_requirements("torchvision>=0.15.0", install=False)

    model_name_to_model_and_weights = {}
    if supports_r3d:
        from torchvision.models.video import R3D_18_Weights, r3d_18

        model_name_to_model_and_weights["r3d_18"] = (r3d_18, R3D_18_Weights.DEFAULT)
    if supports_s3d:
        from torchvision.models.video import S3D_Weights, s3d

        model_name_to_model_and_weights["s3d"] = (s3d, S3D_Weights.DEFAULT)
    if supports_swin3db:
        from torchvision.models.video import Swin3D_B_Weights, swin3d_b

        model_name_to_model_and_weights["swin3d_b"] = (swin3d_b, Swin3D_B_Weights.DEFAULT)
    if supports_swin3dt:
        from torchvision.models.video import Swin3D_T_Weights, swin3d_t

        model_name_to_model_and_weights["swin3d_t"] = (swin3d_t, Swin3D_T_Weights.DEFAULT)
    if supports_mvitv1b:
        from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b

        model_name_to_model_and_weights["mvit_v1_b"] = (mvit_v1_b, MViT_V1_B_Weights.DEFAULT)
    if supports_mvitv2s:
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        model_name_to_model_and_weights["mvit_v2_s"] = (mvit_v2_s, MViT_V2_S_Weights.DEFAULT)

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

        if self.supports_transforms_v2:
            from torchvision.transforms import v2

            transform = v2.Compose(
                [
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Resize(input_size, antialias=True),
                    v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
                ]
            )
        else:
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

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
        return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, sequences: torch.Tensor):
        """
        Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model. The expected input dimensions are
                                      (B, T, C, H, W) for batched video frames or (T, C, H, W) for single video frames.

        Returns:
            (torch.Tensor): The model's output.
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
    """ideo classifier using Hugging Face models; see https://huggingface.co/models?pipeline_tag=video-classification."""

    supports_transforms_v2 = check_requirements("torchvision>=0.16.0", install=False)

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
        check_requirements("transformers")
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

        if self.supports_transforms_v2:
            from torchvision.transforms import v2

            transform = v2.Compose(
                [
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Resize(input_size, antialias=True),
                    v2.Normalize(
                        mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std
                    ),
                ]
            )
        else:
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


if __name__ == "__main__":
    from ultralytics import YOLO
    # from ultralytics.solutions.action_recognition import ActionRecognition

    # Initialize the YOLO model
    model = YOLO("yolov8n.pt")

    device = ""
    video_classifier_model = "microsoft/xclip-base-patch32"
    labels = ["walking", "running", "brushing teeth", "looking into phone", "weight lifting", "cooking", "sitting"]
    action_recognition = ActionRecognition(video_classifier_model=video_classifier_model, labels=labels, device=device)

    cap = cv2.VideoCapture("your/video/path.mp4")
    assert cap.isOpened(), "Error reading video file"
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Perform object tracking
        tracks: List[Results] = model.track(frame, persist=True, classes=[0])
        # Perform action recognition
        annotated_frame = action_recognition.recognize_actions(frame, tracks)
        # Display the frame
        cv2.imshow("Action Recognition", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
