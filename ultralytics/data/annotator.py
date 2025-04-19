# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import List, Optional, Union

import torch

from ultralytics import SAM, YOLO
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.utils import ASSETS, LOGGER
from ultralytics.utils.ops import xyxy2xywhn
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.torch_utils import select_device


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
    A class for automatic annotation of images using YOLO or Florence2 models.

    This class handles model selection, object detection, and label generation in YOLO format.
    It supports both YOLO-based and Florence2-based annotation pipelines, including visual output.

    Attributes:
        Y0LO_MODEL (bool): Flag indicating if the annotation uses a YOLO model.
        model (YOLO | Florence2): Loaded model instance for annotation.
        processor (Florence2Processor): Used only for Florence2-based annotation.
        names (dict): Class names for YOLO-based models.

    Methods:
        annotate: Performs annotation on a dataset of images or videos.
        convert_to_yolo: Converts bounding box coordinates to YOLO normalized format.
        create_classes_txt: Saves class names into a `classes.txt` file.
    """

    def __init__(self, model=None, device="", variant="base"):
        """
        Initializes the AutoAnnotator with either a YOLO or Florence2 model.

        Args:
            model (str, optional): Model name or path. Defaults to "yolo11n.pt".
            device (str): Device to use for inference (e.g., 'cpu', 'cuda:0').
            variant (str): Florence2 model variant if using Florence2 (e.g., "base", "large-ft").
        """
        self.YOLO_MODEL = True  # Assume YOLO by default

        if model is None:
            LOGGER.warning("⚠️ No model provided. Using default: yolo11n.pt")
            self.model = "yolo11n.pt"
        else:
            model_str = str(model).casefold()

            if model_str.endswith(".pt") and model_str not in ("florence2.pt", "florence2"):
                self.model = YOLO(model)
            elif model_str in ("florence2.pt", "florence2"):
                self.model = Florence2(device=device, variant=variant)
                self.YOLO_MODEL = False
            else:
                LOGGER.warning(
                    f"⚠️ Unsupported model '{model}'. Using default: yolo11n.pt.\n"
                    f"✅ Supported models:\n"
                    f"   - YOLO: yolov8n.pt, yolov11n.pt, yolov10.pt, yolo11n.pt, yolo12n.pt\n"
                    f"   - Florence2: florence2"
                )
                self.model = YOLO("yolo11n.pt")

        if self.YOLO_MODEL:
            self.names = self.model.names

    @staticmethod
    def convert_to_yolo(x1, y1, x2, y2, w, h):
        """
        Converts bounding box coordinates to YOLO format.

        Args:
            x1 (float): X-coordinate of the top-left corner of the bounding box.
            y1 (float): Y-coordinate of the top-left corner of the bounding box.
            x2 (float): X-coordinate of the bottom-right corner of the bounding box.
            y2 (float): Y-coordinate of the bottom-right corner of the bounding box.
            w (int): Image width.
            h (int): Image height.

        Returns:
            str: Bounding box in YOLO format (cx, cy, width, height).
        """
        cx, cy, bw, bh = xyxy2xywhn(torch.tensor([[x1, y1, x2, y2]]), w=w, h=h, clip=True)[0].tolist()
        return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    @staticmethod
    def create_classes_txt(output_dir, label_map):
        """
        Creates a `classes.txt` file listing all class names.

        Args:
            output_dir (Path): Directory where to save the file.
            label_map (dict): Dictionary of {label_name: class_id}.
        """
        try:
            with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
                for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
                    f.write(f"{label}\n")
            LOGGER.info(f"✅ Saved class names to {output_dir / 'classes.txt'}")
        except Exception as e:
            LOGGER.error(f"❌ Failed to save classes.txt: {e}")

    def annotate(self, source=None, conf=0.25, iou=0.45, classes=None,
                 save=True, output_dir="labels", save_visuals=True, visuals_output_dir=None):
        """
        Annotates images or videos using the specified model and saves YOLO-format labels.

        Args:
            source (str | Path): Input image or video directory.
            conf (float): Confidence threshold (YOLO only).
            iou (float): IoU threshold for NMS (YOLO only).
            classes (List[str] | None): Specific classes to annotate.
            save (bool): Whether to save label files.
            output_dir (str): Path to save YOLO annotation .txt files.
            save_visuals (bool): Whether to save annotated visual images.
            visuals_output_dir (str): Directory to save annotated visual output.

        Returns:
            None
        """
        import time
        from pathlib import Path

        from tqdm import tqdm

        if save_visuals:
            import cv2

        start = time.time()

        if source is None:
            LOGGER.warning("⚠️ 'source' argument is missing. Using default source.")
            source = ASSETS

        dataset = LoadImagesAndVideos(path=str(source))
        if not dataset:
            LOGGER.warning("⚠️ No images or videos found in the source.")
            return

        output_dir = Path(output_dir)
        visuals_output_dir = Path(visuals_output_dir or output_dir.parent / f"{output_dir.name}_visuals")

        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
        if save_visuals:
            visuals_output_dir.mkdir(parents=True, exist_ok=True)

        label_map = {}
        processed = 0
        is_yolo = self.YOLO_MODEL

        if is_yolo:
            classes = [i for i, name in self.names.items() if name in classes]

        for path, im0, _ in tqdm(dataset, desc="Annotating Images"):
            try:
                img_path = path[0] if isinstance(path, (list, tuple)) else path
                img_name = Path(img_path).stem
                im0 = im0[0] if isinstance(im0, (list, tuple)) else im0

                if im0 is None:
                    LOGGER.warning(f"⚠️ Could not read image: {img_path}")
                    continue

                h, w = im0.shape[:2]
                if is_yolo:
                    result = self.model.predict(im0, classes=classes, verbose=False, conf=conf, iou=iou)[0]
                    bboxes = result.boxes.xyxy.tolist()
                    labels = result.boxes.cls.tolist()
                else:
                    result = self.model.process(im0, w, h)
                    bboxes = result.get("bboxes") or []
                    labels = result.get("labels") or []

                if not bboxes or not labels:
                    LOGGER.warning(f"⚠️ No output for image: {img_name}")
                    continue

                lines = []
                if save_visuals:
                    annotator = Annotator(im0)

                found_labels = set()

                for box, label in zip(bboxes, labels):
                    if classes is None or label in classes:
                        found_labels.add(label)
                        label_name = self.names[label] if is_yolo else label
                        if label_name not in label_map:
                            label_map[label_name] = len(label_map)
                        mapped_id = label_map[label_name]

                        if save_visuals:
                            annotator.box_label(box, label=label_name, color=colors(mapped_id, True))

                        x1, y1, x2, y2 = box
                        lines.append(f"{mapped_id} {self.convert_to_yolo(x1, y1, x2, y2, w, h)}")

                if not lines:
                    if classes:
                        missing = set(classes) - found_labels
                        msg = f"Missing classes: {sorted(missing)}" if missing else "All boxes skipped."
                        LOGGER.warning(f"⚠️ Skipping image {img_name}. {msg}")
                    else:
                        LOGGER.warning(f"⚠️ All boxes skipped for: {img_name}")
                    continue

                if save:
                    (output_dir / f"{img_name}.txt").write_text("\n".join(lines), encoding="utf-8")

                if save_visuals:
                    cv2.imwrite(str(visuals_output_dir / f"{img_name}.png"), annotator.result())

                processed += 1

            except Exception as e:
                LOGGER.error(f"❌ Error processing {path}: {e}", exc_info=True)

        if save and label_map:
            self.create_classes_txt(output_dir, label_map)

        if processed:
            LOGGER.info(f"✅ Annotated {processed} image(s) in {time.time() - start:.2f} seconds.")
        else:
            LOGGER.warning("⚠️ No images were successfully annotated.")


class Florence2:
    """
    A Florence-2 model wrapper for grounding-style object detection and auto-annotation.

    This class handles loading Florence2 variants and running image-text inference to produce object bounding boxes
    and labels. It integrates with Ultralytics' AutoAnnotator for automatic dataset labeling.

    Attributes:
        device (torch.device): Inference device (e.g., 'cuda', 'cpu').
        variant (str): Florence2 variant used (e.g., "base", "large-ft").
        mid (str): Model identifier string used with Hugging Face.
        model (torch.nn.Module): Loaded Florence2 model.
        processor (Florence2Processor): Image and text processor used for inference.

    Methods:
        _load_model: Internal method to load Florence2 model and processor.
        process: Performs inference and returns bounding boxes and labels in YOLO-compatible format.
    """

    def __init__(self, device="", variant="base"):
        """
        Initializes the Florence2 model with the specified variant and device.

        Args:
            device (str): Target device for inference, e.g., 'cuda', 'cpu'.
            variant (str): Model variant to use: "base", "large", "base-ft", "large-ft".
        """
        self.device = select_device(device)

        supported_variants = {"base", "large", "large-ft", "base-ft"}
        self.variant = variant if variant in supported_variants else "base"

        if variant not in supported_variants:
            LOGGER.warning(
                f"⚠️ Invalid variant '{variant}' provided. Falling back to 'base'. "
                f"Supported variants: {supported_variants}"
            )

        self.mid = f"microsoft/florence-2-{self.variant}"
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """
        Loads the Florence2 model and its corresponding processor from Hugging Face Hub.

        This method requires `transformers` and `einops`. It sets up the model and processor
        using `trust_remote_code=True`, which allows Florence2 custom architecture loading.

        Returns:
            None
        """
        from ultralytics.utils.checks import check_requirements
        check_requirements(["transformers==4.49.0", "einops"])

        from transformers import AutoModelForCausalLM, AutoProcessor, logging
        logging.set_verbosity_error()  # Minimize Hugging Face logs

        LOGGER.info(f"💡 Initializing Florence2-{self.variant} on {str(self.device).upper()}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.mid,
            trust_remote_code=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.mid,
            trust_remote_code=True,
        )

        LOGGER.info(f"🚀 Loaded Florence2-{self.variant} model successfully.")

    def process(self, im0, w, h):
        """
        Performs object detection using Florence2 on an input image.

        This method sends a grounding-style prompt "<OD>" to the model to detect objects
        and returns predictions formatted for YOLO-compatible annotation.

        Args:
            im0 (np.ndarray): Input image (HWC format).
            w (int): Image width.
            h (int): Image height.

        Returns:
            dict: Dictionary with keys "bboxes" and "labels", formatted for YOLO annotation.
                  Structure:
                  {
                      "bboxes": List of [x1, y1, x2, y2] floats,
                      "labels": List of str or int label names
                  }

        Example:
            >>> model = Florence2(device="cuda", variant="base")
            >>> output = model.process(im, w=640, h=480)
            >>> print(output["bboxes"], output["labels"])
        """
        # Encode image and grounding prompt
        inputs = self.processor(text="<OD>", images=im0, return_tensors="pt").to(self.device)
        ids, pval = inputs["input_ids"], inputs["pixel_values"]

        # Perform text-vision inference
        outids = self.model.generate(
            ids,
            pixel_values=pval,
            max_new_tokens=1024,
            early_stopping=False,
            num_beams=3
        )

        # Decode and convert Florence2 output to detection format
        outputs = self.processor.batch_decode(outids, skip_special_tokens=False)[0]
        processed = self.processor.post_process_generation(outputs, task="<OD>", image_size=(w, h))

        return processed.get("<OD>", {})  # Return dictionary with 'bboxes' and 'labels'
