# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
from pathlib import Path
from typing import List, Optional, Union

from ultralytics import SAM, YOLO
from ultralytics.utils.checks import check_requirements


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


import time
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors
import torch
from tqdm import tqdm
from ultralytics.utils.torch_utils import select_device

class AutoAnnotator:
    def __init__(self, model=None):

        self.device = select_device()
        self.default_task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        self.od_task_prompt = "<OD>"
        self.model, self.processor = None, None
        self.m_id = "microsoft/Florence-2-base-ft"
        self.torch_dtype = torch.float16 if self.device=="cuda" else torch.float32
        self._load_model("florence-2" if model is None else model)

    def _load_model(self, model_name):
        if model_name.lower() == "florence-2":
            from ultralytics.utils.checks import check_requirements
            check_requirements(["transformers==4.49.0", "einops"])
            from transformers import AutoModelForCausalLM, AutoProcessor
            LOGGER.info(f"💡 Loading ({self.m_id}) model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.m_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.m_id, trust_remote_code=True)
            LOGGER.info(f"✅ Model {self.m_id} loaded successfully on {self.device}.")
        else:
            LOGGER.warning(f"⚠️ Model '{model_name}' not supported. Defaulting to 'florence-2'.")
            self._load_model("florence-2")

    @staticmethod
    def convert_to_yolo(x1, y1, x2, y2, w, h):
        cx = max(0.0, min(1.0, (x1 + x2) / (2 * w)))
        cy = max(0.0, min(1.0, (y1 + y2) / (2 * h)))
        bw = max(0.0, min(1.0, (x2 - x1) / w))
        bh = max(0.0, min(1.0, (y2 - y1) / h))
        return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def annotate(self, data, classes=None, save=True, output_dir="labels", save_visuals=True, visuals_output_dir=None):
        start = time.time()
        dataset = LoadImagesAndVideos(path=str(data))

        task_prompt = self.default_task_prompt if classes else self.od_task_prompt
        text_input = ",".join(classes) if isinstance(classes, list) else classes
        label_map = {}

        full_prompt = task_prompt + (text_input or "")

        output_dir = Path(output_dir)
        visuals_output_dir = Path(visuals_output_dir or output_dir.parent / f"{output_dir.name}_visuals")
        output_dir.mkdir(parents=True, exist_ok=True) if save else None
        visuals_output_dir.mkdir(parents=True, exist_ok=True) if save_visuals else None

        for path, im0, _ in tqdm(dataset, desc="Annotating Images"):
            try:
                img_name = Path(path[0] if isinstance(path, (list, tuple)) and path else path).stem

                if isinstance(im0, (list, tuple)):
                    im0 = im0[0] if im0 else None
                if im0 is None:
                    LOGGER.warning(f"⚠️ Could not read image: {path}")
                    continue

                h, w = im0.shape[:2]
                inputs = self.processor(
                    text=full_prompt,
                    images=im0,
                    return_tensors="pt"
                ).to(self.device, self.torch_dtype)

                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    early_stopping=False,
                    num_beams=3
                )

                out_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
                result = self.processor.post_process_generation(out_text, task=task_prompt, image_size=(w, h))

                results = result.get(task_prompt, {})
                bboxes = results.get("bboxes") or []
                labels = results.get("labels") or []

                if not bboxes or not labels:
                    continue

                annotator = Annotator(im0)
                unique_labels = sorted(set(labels))
                label_map.update({l: idx for idx, l in enumerate(unique_labels) if l not in label_map})
                lines = []

                for box, label in zip(bboxes, labels):
                    annotator.box_label(box, label=label, color=colors(label_map[label], True))
                    lines.append(f"{label_map[label]} {self.convert_to_yolo(*box, w, h)}")

                # Save annotations and visuals
                if save:
                    with open(output_dir / f"{img_name}.txt", "w") as f:
                        f.write("\n".join(lines))

                if save_visuals:
                    import cv2
                    cv2.imwrite(str(visuals_output_dir / f"{img_name}.png"), annotator.result())

            except Exception as e:
                LOGGER.error(f"❌ Error processing {path}: {e}", exc_info=True)

        LOGGER.info(f"✅ Annotated {len(path)} images in {time.time() - start:.2f}s")