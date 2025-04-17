# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import List, Optional, Union

from ultralytics import SAM, YOLO

import time
import torch

from tqdm import tqdm
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xyxy2xywhn


def auto_annotate(
    data: Union[str, Path],
    det_model: str = "yolo11x.pt",
    sam_model: str = "sam_b.pt",
    device: str = "",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Automatically annotate images using a YOLO object detection model and a SAM segmentation model.

    This function processes images in a specified directory, detects objects using a YOLO model, and then generates
    segmentation masks using a SAM model. The resulting annotations are saved as text files.

    Args:
        data (str | Path): Path to a folder containing images to be annotated.
        det_model (str): Path or name of the pre-trained YOLO detection model.
        sam_model (str): Path or name of the pre-trained SAM segmentation model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0').
        conf (float): Confidence threshold for detection model.
        iou (float): IoU threshold for filtering overlapping boxes in detection results.
        imgsz (int): Input image resize dimension.
        max_det (int): Maximum number of detections per image.
        classes (List[int] | None): Filter predictions to specified class IDs, returning only relevant detections.
        output_dir (str | Path | None): Directory to save the annotated results. If None, a default directory is created.

    Examples:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolo11n.pt", sam_model="mobile_sam.pt")
    """
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(
        data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes
    )

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if class_ids:
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w", encoding="utf-8") as f:
                for i, s in enumerate(segments):
                    if s.any():
                        segment = map(str, s.reshape(-1).tolist())
                        f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")


class AutoAnnotator:
    """
    A class for automated annotation of images and videos using caption-to-phrase grounding.

    This class loads a Microsoft Florence-2 by default to automatically generate object annotations
    from text prompts. It supports saving results in YOLO format and optionally generating visual outputs
    with labeled bounding boxes.

    Attributes:
        device (str): Torch device used for inference ('cuda' or 'cpu').
        task_prompt (str): Prompt used for caption-to-phrase grounding.
        default_task_prompt (str): Fallback prompt for object detection.
        min_box_w (float): Minimum width threshold for bounding boxes (normalized).
        min_box_h (float): Minimum height threshold for bounding boxes (normalized).
        m_id (str): Model ID used to load the Florence-2 model from Hugging Face.
        model (torch.nn.Module): Loaded grounding model.
        processor (transformers.Processor): Corresponding processor to handle inputs/outputs.
        torch_dtype (torch.dtype): Data type used for model inputs (fp16 or fp32).

    Methods:
        __init__: Initializes the annotator with optional model selection and size filters.
        _load_model: Loads a model and processor based on the specified model name.
        convert_to_yolo: Static method to convert (x1, y1, x2, y2) to normalized YOLO format.
        annotate: Performs annotation over a dataset with optional class filtering and saving.
    """

    def __init__(self, model=None, min_box_w=0.02, min_box_h=0.02):
        """Initialize the AutoAnnotator with a visual grounding model and box filtering parameters."""
        self.default_tp = "<OD>"  # Fallback prompt for generic object detection
        self.min_box_w = min_box_w
        self.min_box_h = min_box_h
        self.m_id = "microsoft/Florence-2-base-ft"
        self.model, self.processor = None, None
        self.device = select_device()  # Auto-select CUDA or CPU
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32  # Use fp16 on CUDA
        self._load_model("florence-2" if model is None else model)  # Load model

    def _load_model(self, model_name):
        """Load the Florence-2 grounding model and corresponding processor."""
        if model_name.lower() == "florence-2":
            from ultralytics.utils.checks import check_requirements
            check_requirements(["transformers==4.49.0", "einops"])  # Ensure required libraries

            from transformers import AutoModelForCausalLM, AutoProcessor, logging
            LOGGER.info(f"💡Loading model: {self.m_id}")
            logging.set_verbosity_error()  # Suppress excessive logs from transformers library: https://huggingface.co/docs/transformers/en/main_classes/logging

            # Load the model and processor from Hugging Face
            self.model = AutoModelForCausalLM.from_pretrained(
                self.m_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(self.m_id, trust_remote_code=True)
            LOGGER.info(f"✅ Model loaded on {self.device}")
        else:
            LOGGER.warning(f"⚠️ Unknown model '{model_name}', defaulting to 'florence-2'")
            self._load_model("florence-2")

    @staticmethod
    def convert_to_yolo(x1, y1, x2, y2, w, h):
        """Convert bounding box coordinates from (x1, y1, x2, y2) to normalized YOLO format."""
        cx, cy, bw, bh = xyxy2xywhn(torch.tensor([[x1, y1, x2, y2]]), w=w, h=h, clip=True)[0].tolist()
        return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def annotate(self, data, classes=None, save=True, output_dir="labels", save_visuals=True, visuals_output_dir=None):
        """Annotate images or video frames using a caption-grounding model."""
        start = time.time()
        dataset = LoadImagesAndVideos(path=str(data))  # Load data (images/videos)

        label_map = {}
        output_dir = Path(output_dir)
        visuals_output_dir = Path(visuals_output_dir or output_dir.parent / f"{output_dir.name}_visuals")
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists
        if save_visuals:
            import cv2
            visuals_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure visual output dir exists

        processed = 0  # Counter for successfully annotated images

        for path, im0, _ in tqdm(dataset, desc="Annotating Images"):
            try:
                img_name = Path(path[0] if isinstance(path, (list, tuple)) else path).stem

                if isinstance(im0, (list, tuple)):
                    im0 = im0[0] if im0 else None
                if im0 is None:
                    LOGGER.warning(f"⚠️ Could not read image: {path}")
                    continue

                h, w = im0.shape[:2]  # Get image dimensions

                # Encode image and text prompt
                inputs = self.processor(text=self.default_tp,
                    images=im0,
                    return_tensors="pt"
                ).to(self.device, self.torch_dtype)

                # Perform inference
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    early_stopping=False,
                    num_beams=3
                )

                # Decode and post-process output
                out_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
                result = self.processor.post_process_generation(out_text, task=self.default_tp, image_size=(w, h))
                results = result.get(self.default_tp, {})
                bboxes = results.get("bboxes") or []
                labels = results.get("labels") or []

                if not bboxes or not labels:
                    LOGGER.warning(f"⚠️ No output for image: {img_name}")
                    continue

                lines = []
                if save_visuals:
                    annotator = Annotator(im0)  # For drawing bounding boxes

                # Process each box-label pair
                for box, label in zip(bboxes, labels):
                    if label in classes:
                        x1, y1, x2, y2 = box
                        bw, bh = (x2 - x1) / w, (y2 - y1) / h  # Normalize box size

                        if bw < self.min_box_w or bh < self.min_box_h:
                            continue  # Skip small boxes

                        # Add new label to label_map if not already present
                        if label not in label_map:
                            label_map[label] = int(len(label_map))

                        if save_visuals:
                            annotator.box_label(box, label=label, color=colors(label_map[label], True))

                        lines.append(f"{label_map[label]} {self.convert_to_yolo(x1, y1, x2, y2, w, h)}")

                if not lines:
                    if label not in classes:
                        LOGGER.warning(f"⚠️ Skipping image file {img_name}, no {classes} found")
                    else:
                        LOGGER.warning(f"⚠️ All boxes skipped (too small or invalid) for: {img_name}")
                    continue

                # Save annotation file
                if save:
                    with open(output_dir / f"{img_name}.txt", "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))

                # Save visual output
                if save_visuals:
                    cv2.imwrite(str(visuals_output_dir / f"{img_name}.png"), annotator.result())

                processed += 1  # Count processed images

            except Exception as e:
                LOGGER.error(f"❌ Error processing {path}: {e}", exc_info=True)

        # Save class name mapping
        if save and label_map:
            try:
                with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
                    for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
                        f.write(f"{label}\n")
                LOGGER.info(f"✅ Saved class names to {output_dir / 'classes.txt'}")
            except Exception as e:
                LOGGER.error(f"❌ Failed to save classes.txt: {e}")

        LOGGER.info(f"✅ Annotated {processed} image(s) in {time.time() - start:.2f} seconds.")
