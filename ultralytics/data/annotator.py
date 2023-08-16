# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.models import SAM, YOLO


def auto_annotate_segment(data, det_model='yolov8x.pt', sam_model='sam_b.pt', device='', output_dir=None):
    """
    Automatically annotates images using a YOLO object detection model and a SAM segmentation model.
    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str, optional): Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.
        sam_model (str, optional): Pre-trained SAM segmentation model. Defaults to 'sam_b.pt'.
        device (str | int | optional): Device to run the models on. Defaults to an empty string (CPU or GPU, if available).
        output_dir (str | None | optional): Directory to save the annotated results.
            Defaults to a 'labels' folder in the same directory as 'data'.
    """
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    if not output_dir:
        output_dir = Path(str(data)).parent / 'labels'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(data, stream=True, device=device)

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if len(class_ids):
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn  # noqa

            with open(f'{str(Path(output_dir) / Path(result.path).stem)}.txt', 'w') as f:
                for segment, class_id in zip(segments, class_ids):
                    if len(segment) == 0:
                        continue
                    segment = map(str, segment.reshape(-1).tolist())
                    f.write(f'{class_id} ' + ' '.join(segment) + '\n')


def auto_annotate_detect(data, det_model='yolov8x.pt', device='', output_dir=None):
    """
    Automatically annotates images using a YOLO object detection model.
    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str, optional): Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.
        device (str | int | optional): Device to run the models on. Defaults to an empty string (CPU or GPU, if available).
        output_dir (str | None | optional): Directory to save the annotated results.
            Defaults to a 'labels' folder in the same directory as 'data'.
    """
    det_model = YOLO(det_model)

    if not output_dir:
        output_dir = Path(str(data)).parent / 'labels'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(data, stream=True, device=device)

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if len(class_ids):
            boxes = result.boxes.xywhn  # Boxes object for bbox outputs
            with open(f'{str(Path(output_dir) / Path(result.path).stem)}.txt', 'w') as f:
                for box, class_id in zip(boxes, class_ids):
                    if len(box) == 0:
                        continue
                    box = map(str, box.reshape(-1).tolist())
                    f.write(f'{class_id} ' + ' '.join(box) + '\n')
