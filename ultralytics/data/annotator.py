# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import List, Optional, Union

from ultralytics import SAM, YOLO
import cv2
import time
import torch
import glob
from tqdm import tqdm

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import LOGGER


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

    def __init__(self, model=None):
        """
        Initializes the AutoAnnotator class with a specified model.

        Args:
            model (str, optional): Model identifier. Defaults to "florence-2",
                                   which loads "microsoft/Florence-2-large".
                                   Currently, only "florence-2" is supported.
        """
        self.model = None
        self.processor = None
        self.torch_dtype = None
        self.m_id = None # Store model ID
        self.device = select_device()  # Select device using ultralytics util

        self.default_task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"  # Default task prompt for grounding/detection
        self.od_task_prompt = "<OD>"  # Default task prompt when no classes are provided (general object detection)

        self.img_width = 0  # input image width (will be updated per image)
        self.img_height = 0 # input image height (will be updated per image)

        self.torch_dtype = torch.float16 if self.device=="cuda:0" else torch.float32  # required for processing inputs
        self._load_model("florence-2" if model is None else model)  # Model initialization

    def _load_model(self, model_name):
        """Loads the model and processor."""
        if model_name.lower() == "florence-2":

            check_requirements("transformers==4.49.0") # Check transformer version
            from transformers import AutoModelForCausalLM, AutoProcessor

            self.m_id = "microsoft/Florence-2-base-ft"  # Use large model by default for florence-2 alias
            LOGGER.info(f"💡 Loading ({self.m_id}) model...")

            # Model configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.m_id,
                # use_flash_attention_2=False,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)

            # Generate the input data for model processing from the given prompt and image
            self.processor = AutoProcessor.from_pretrained(
                self.m_id,
                trust_remote_code=True
            )
            LOGGER.info(f"✅ Model {self.m_id} loaded successfully on {self.device}.")
        else:
            LOGGER.warning(f"⚠️ Model '{model_name}' is not supported. "
                           f"Defaulting to 'florence-2' ({self.m_id}).")
            if self.m_id is None:  # Direct call
                self._load_model("florence-2")

    @staticmethod
    def _get_image_paths(data):
        """
        Resolves the data source into a list of image file paths.

        Args:
            data (str | Path | list | np.ndarray): A directory path, a file path,
                                                   a glob pattern, a list of paths,
                                                   or a single loaded image (np.ndarray).

        Returns:
            list: A list of validated image file paths (Path objects).
                  Returns empty list if no valid images found or on error.
        """
        image_paths = []
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            if data_path.is_dir():
                for ext in IMG_FORMATS:
                    image_paths.extend(data_path.glob(f'*.{ext}'))  # Use dot here
            elif "*" in str(data_path) or "?" in str(data_path):
                image_paths.extend([Path(p) for p in glob.glob(str(data_path), recursive=True)])
            elif data_path.is_file():
                image_paths.append(data_path)
            else:
                LOGGER.warning(f"⚠️ Input path '{data}' is not a directory, file, or recognized glob pattern.")
        elif isinstance(data, list):
            image_paths = [Path(p) for p in data if isinstance(p, (str, Path))]

        valid_paths = []
        for p in image_paths:
            if p.is_file():
                suffix = p.suffix.lower().lstrip(".")  # Remove the dot before checking
                if suffix in IMG_FORMATS:
                    valid_paths.append(p)
                else:
                    LOGGER.warning(f"⚠️ Skipping file with unsupported suffix: {p}")

        if not valid_paths:
            LOGGER.warning(f"No valid image files found for source: '{data}'. Searched for suffixes: {IMG_FORMATS}")

        return sorted(valid_paths)

    @staticmethod
    def _prepare_output_dirs(output_dir, visuals_output_dir, save, save_visuals):
        """Creates output directories if they don't exist."""
        path_output_dir = None
        path_visuals_dir = None

        if save:
            path_output_dir = Path(output_dir)
            path_output_dir.mkdir(parents=True, exist_ok=True)

        if save_visuals:
            if not visuals_output_dir:
                 if path_output_dir:
                      visuals_output_dir = path_output_dir.parent / f"{path_output_dir.name}_visuals"
                 else:
                      visuals_output_dir = Path("./labels_visuals")
            path_visuals_dir = Path(visuals_output_dir)
            path_visuals_dir.mkdir(parents=True, exist_ok=True)

        return path_output_dir, path_visuals_dir, save # Return updated save status

    def _process(self, im0, task_prompt, text_input=None):
        """
        Runs the Florence-2 model inference on a single PIL image.

        Args:
            im0 (PIL.Image): Input image in PIL format.
            task_prompt (str): The base task prompt (e.g., <OD>, <CAPTION_TO_PHRASE_GROUNDING>).
            text_input (str, optional): Additional text input, like class names.

        Returns:
            dict: The parsed output dictionary from the processor's post-processing.
        """
        prompt = task_prompt + text_input if text_input else task_prompt  # Construct the full prompt

        # Generate the input data for model processing from the given prompt and image
        inputs = self.processor(text=prompt, images=im0, return_tensors="pt").to(self.device, self.torch_dtype)

        # Generate model predictions (token IDs)
        # with torch.no_grad(): # Ensure inference mode
        output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,  # Set maximum number of tokens to generate
                early_stopping=False,
                num_beams=3,
            )

        # Decode generated token IDs into text, results is a list, keep special tokens for post-processing
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]

        try:
            # Pass the base task_prompt used for generation
            parsed_results = self.processor.post_process_generation(
                output_text,
                task=task_prompt, # Use the base task prompt
                image_size=(self.img_width, self.img_height), # Use W, H format
            )
            return parsed_results

        except Exception as e:
            LOGGER.error(f"❌ Error during post-processing: {e}")
            LOGGER.error(f"⚠️ Generated text was: {output_text}")
            return {"bboxes": [], "labels": []} # Return empty structure on error

    def annotate(self,
                 data,  # Can be image path, dir, glob, list, or numpy array
                 classes=None,  # List of classes for detection (optional)
                 save=True,  # Save annotation files
                 output_dir="labels",  # Directory to save annotation files
                 save_visuals=True,  # Save annotated images
                 visuals_output_dir=None, # Directory for visuals (defaults relative to output_dir)
                 ):
        """
        Performs auto-annotation on images using the loaded Florence-2 model.

        Args:
            data (str | Path | list | np.ndarray): Source image(s). Can be a single image file path,
                a directory path, a glob pattern, a list of image paths, or a preloaded
                OpenCV image (np.ndarray).
            classes (list[str], optional): A list of class names to detect. If provided,
                uses the grounding prompt. If None, uses the OD prompt. Defaults to None.
            save (bool): If True, saves YOLO format annotation files. Defaults to True.
            output_dir (str): Directory to save the annotation files (.txt). Defaults to "labels_auto".
            save_visuals (bool): If True, saves images with annotations drawn. Defaults to True.
            visuals_output_dir (str, optional): Directory to save visualized images (.png).
                If None, defaults to a directory named like '{output_dir}_visuals'.
        """
        total_start_time = time.time()

        image_paths = self._get_image_paths(data)
        num_images = len(image_paths)

        if num_images == 0:
            LOGGER.error("❌ No images found or provided to annotate. Use data='path/to/img/directory'")
            return

        # Determine task prompt and text input based on classes
        if classes:
            if not all(isinstance(c, str) for c in classes):
                 LOGGER.error("❌ 'classes' must be a comma-separated string. i.e 'person, car, bus, truck'")
                 return
            task_prompt = self.default_task_prompt
            text_input = f"{classes}"
            LOGGER.info(f"🤩 Using grounding prompt for classes: {classes}")
        else:
            task_prompt = self.od_task_prompt # Use Object Detection prompt
            text_input = None
            label_map = {} # Will be built dynamically based on detected labels if no classes are given
            LOGGER.info("No classes provided, using general object detection prompt.")

        # Prepare output directories
        path_output_dir, path_visuals_dir, save = self._prepare_output_dirs(
            output_dir, visuals_output_dir, save, save_visuals
        )

        processed_count = 0
        for i, img_source in enumerate(tqdm(image_paths, desc="Annotating Images")):
            try:
                img_path = Path(img_source)  # Load Image
                img_name = img_path.stem # Name without extension
                im0 = cv2.imread(str(img_path))
                if im0 is None:
                    LOGGER.warning(f"⚠️ Could not read image: {img_path}, skipping.")
                    continue

                self.img_height, self.img_width = im0.shape[:2]  # Store image width and height for post-processing

                results = self._process(im0, task_prompt=task_prompt, text_input=text_input)[task_prompt]  # Process im0

                labels = results.get("labels") or []

                # Ensure results format is consistent (dict with 'bboxes', 'labels')
                if not isinstance(results, dict) or 'bboxes' not in results or not labels:
                    LOGGER.warning(f"⚠️ Unexpected result format for {img_name}: {results}. Skipping image save.")
                    continue

                annotator = Annotator(im0)

                if not classes:  # build label_map dynamically
                     unique_labels = sorted(list(set(labels)))
                     if not label_map: # First time or if previously empty
                          label_map = {name: idx for idx, name in enumerate(unique_labels)}
                     else: # Update map only with new labels found in this image
                          current_max_idx = max(label_map.values()) if label_map else -1
                          for label in unique_labels:
                              if label not in label_map:
                                  current_max_idx += 1
                                  label_map[label] = current_max_idx

                # Process detections
                yolo_lines = []
                label_map = {name: idx for idx, name in enumerate(sorted(set(labels)))}
                for idx, (box, label) in enumerate(zip(results.get("bboxes", []), labels)):
                    if label in label_map:  # Get class index from map
                        class_index = label_map[label]
                    elif not classes: # OD task and label not seen before (should have been added above)
                        LOGGER.warning(f"⚠️ Label '{label}' detected but missing from dynamic map, Rebuilding map.")
                        current_max_idx = max(label_map.values()) if label_map else -1
                        label_map[label] = current_max_idx + 1
                        class_index = label_map[label]
                        LOGGER.info(f"✅ Updated dynamic map: {label_map}")

                    annotator.box_label(box, label=f"{label}", color=colors(class_index, True))  # draw bounding box

                    # Convert to YOLO format (cx, cy, w, h) normalized and clamped to [0.0, 1.0]
                    x1, y1, x2, y2 = box
                    iw, ih = self.img_width, self.img_height
                    cx = min(1.0, max(0.0, (x1 + x2) / (2 * iw)))
                    cy = min(1.0, max(0.0, (y1 + y2) / (2 * ih)))
                    w = min(1.0, max(0.0, (x2 - x1) / iw))
                    h = min(1.0, max(0.0, (y2 - y1) / ih))
                    yolo_lines.append(f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                # Save Output
                if save and path_output_dir:  # Save YOLO annotation file
                    output_ann_path = path_output_dir / f"{img_name}.txt"
                    if yolo_lines:
                        try:
                            with open(output_ann_path, "w") as f:
                                f.write("\n".join(yolo_lines))
                        except Exception as e:
                             LOGGER.error(f"❌ Failed to write annotation file {output_ann_path}: {e}")

                # Save result image
                if save_visuals and path_visuals_dir:
                    output_vis_path = path_visuals_dir / f"{img_name}.png" # Save as png
                    annotated_image = annotator.result()
                    try:
                        cv2.imwrite(str(output_vis_path), annotated_image)
                    except Exception as e:
                        LOGGER.error(f"❌ Failed to save visual annotation to {output_vis_path}: {e}")

                processed_count += 1

            except Exception as e:
                LOGGER.error(f"❌ Failed to process image {img_source or 'input array'} due to error: {e}", exc_info=True) # Add traceback

        total_end_time = time.time()
        LOGGER.info(f"Annotation process finished for {processed_count}/{num_images} images.")
        LOGGER.info(f"🚀 Total time: {total_end_time - total_start_time:.2f} seconds.")
