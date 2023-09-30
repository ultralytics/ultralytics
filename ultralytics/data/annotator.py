# Ultralytics YOLO 🚀, AGPL-3.0 license

import shutil
from pathlib import Path

from ultralytics import YOLO


def auto_annotate(data, det_model='yolov8x.pt', device='', output_dir=None):
    """
    Automatically annotates images using a YOLO object detection model and a SAM segmentation model.

    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str, optional): Pre-trained YOLO detection / segmentation / pose estimation model. Defaults to 'yolov8x.pt'.
        device (str, optional): Device to run the models on. Defaults to an empty string (CPU or GPU, if available).
        output_dir (str | None | optional): Directory to save the annotated results.
            Defaults to a 'labels' folder in the same directory as 'data'.

    Example:
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data='ultralytics/assets', det_model='yolov8n.pt')
        ```
    """
    yolo_model = YOLO(det_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f'{data.stem}_auto_annotate_labels'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    results = yolo_model(data, stream=True, device=device)

    for result in results:
        if result.probs:
            classify_dir = output_dir / 'classify'
            Path(classify_dir).mkdir(exist_ok=True, parents=True)
            Path(classify_dir / result.names.get(result.probs.top1)).mkdir(exist_ok=True, parents=True)
            shutil.copy(result.path, classify_dir / result.names.get(result.probs.top1))
        else:
            if len(classes := result.boxes.cls.int().tolist()) == 0:
                continue

            if result.boxes:
                detect_dir = output_dir / 'detect'
                Path(detect_dir).mkdir(exist_ok=True, parents=True)
                with open(f'{str(Path(detect_dir) / Path(result.path).stem)}.txt', 'w') as f:
                    for cls_, box in zip(classes, result.boxes.xywhn.cpu().numpy()):
                        f.write(f'{cls_} ' + ' '.join(list(map(str, box))) + '\n')
            
            if result.masks:
                segment_dir = output_dir / 'segment'
                Path(segment_dir).mkdir(exist_ok=True, parents=True)
                with open(f'{str(Path(segment_dir) / Path(result.path).stem)}.txt', 'w') as f:
                    for cls_, mask in zip(classes, result.masks.xyn):
                        f.write(f'{cls_} ' + ' '.join(list(map(str, mask.flatten()))) + '\n')

            if result.keypoints:
                pose_dir = output_dir / 'pose'
                Path(pose_dir).mkdir(exist_ok=True, parents=True)
                with open(f'{str(Path(pose_dir) / Path(result.path).stem)}.txt', 'w') as f:
                    for cls_, box, keypoint in zip(classes, result.boxes.xywhn.cpu().numpy(), result.keypoints.xyn.cpu().numpy()):
                        f.write(f'{cls_} ' + ' '.join(list(map(str, box))) + ' ' + ' '.join(list(map(str, keypoint.flatten()))) + '\n')