import argparse
import os

import yaml

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, default="yolov10s.pt")
    parser.add_argument("--task", type=str, default="detect")
    parser.add_argument("--dataset", type=str, default="coco_wb.yaml")  # yaml or directory path with images
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=list, default=["0"])
    parser.add_argument("--end2end", action="store_true", default=False)
    parser.add_argument("--no_en2end", action="store_false", dest="end2end")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", type=str, default="ultralytics-runs")
    parser.add_argument("--name", type=str, default="pred")
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--no_stream", action="store_false", dest="stream")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    print("ARGS:", args)

    model = YOLO(args.pt, task=args.task)

    del model.model.model[-1].cv2
    del model.model.model[-1].cv3
    del model.model.model[-1].cv4

    if args.dataset.split(".")[-1] == "yaml":  # yaml file
        with open(args.dataset) as f:
            data = yaml.safe_load(f)

            directories = os.path.join(data["path"][1:], data["val"])

            with open(directories) as f:
                images = f.readlines()

                len_images = int(len(images) * args.fraction)

                sources = [os.path.join(data["path"][1:], images[i][2:-1]) for i in range(len_images)]
    else:  # directory path
        sources = [
            os.path.join(args.dataset, img)
            for img in os.listdir(args.dataset)
            if os.path.isfile(os.path.join(args.dataset, img))
        ]

    for source in sources:
        try:
            model.predict(
                task=args.task,
                source=source,
                stream=args.stream,
                device=args.device,
                end2end=args.end2end,
                project=args.project,
                name=args.name,
                conf=args.conf,
                save=True,
            )
        except Exception as e:
            print(f"Skipping {source} because of error: {e}.")
            continue
