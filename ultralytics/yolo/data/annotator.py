from pathlib import Path

from ultralytics import YOLO
from ultralytics.vit.sam import PromptPredictor, build_sam
from ultralytics.yolo.utils.torch_utils import select_device


def auto_annotate(data, det_model='yolov8x.pt', sam_model='sam_b.pt', device='', output_dir=None):
    """
    Automatically annotates images using a YOLO object detection model and a SAM segmentation model.
    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str, optional): Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.
        sam_model (str, optional): Pre-trained SAM segmentation model. Defaults to 'sam_b.pt'.
        device (str, optional): Device to run the models on. Defaults to an empty string (CPU or GPU, if available).
        output_dir (str | None | optional): Directory to save the annotated results.
            Defaults to a 'labels' folder in the same directory as 'data'.
    """
    device = select_device(device)
    det_model = YOLO(det_model)
    sam_model = build_sam(sam_model)
    det_model.to(device)
    sam_model.to(device)

    if not output_dir:
        output_dir = Path(str(data)).parent / 'labels'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    prompt_predictor = PromptPredictor(sam_model)
    det_results = det_model(data, stream=True)

    for result in det_results:
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if len(class_ids):
            prompt_predictor.set_image(result.orig_img)
            masks, _, _ = prompt_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=prompt_predictor.transform.apply_boxes_torch(boxes, result.orig_shape[:2]),
                multimask_output=False,
            )

            result.update(masks=masks.squeeze(1))
            segments = result.masks.xyn  # noqa

            with open(str(Path(output_dir) / Path(result.path).stem) + '.txt', 'w') as f:
                for i in range(len(segments)):
                    s = segments[i]
                    if len(s) == 0:
                        continue
                    segment = map(str, segments[i].reshape(-1).tolist())
                    f.write(f'{class_ids[i]} ' + ' '.join(segment) + '\n')
