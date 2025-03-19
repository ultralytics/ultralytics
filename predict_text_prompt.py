import argparse
import os

import supervision as sv
from PIL import Image

from ultralytics import YOLOE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, default="yoloe-v8l-seg.pt", help="Path or ID of the model checkpoint")
    parser.add_argument("--names", nargs="+", default=["person"], help="List of class names to set for the model")
    parser.add_argument("--output", type=str, help="Path to save the annotated image")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.source)
        args.output = f"{base}-output{ext}"

    image = Image.open(args.source).convert("RGB")

    model = YOLOE(args.checkpoint)
    model.to(args.device)

    model.set_classes(args.names, model.get_text_pe(args.names))
    results = model.predict(image, verbose=False)

    detections = sv.Detections.from_ultralytics(results[0])

    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections["class_name"], detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(
        scene=annotated_image, detections=detections
    )
    annotated_image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(
        scene=annotated_image, detections=detections
    )
    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True
    ).annotate(scene=annotated_image, detections=detections, labels=labels)

    annotated_image.save(args.output)
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
