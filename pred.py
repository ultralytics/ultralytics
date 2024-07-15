from ultralytics import YOLO
import argparse
import yaml
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, default="yolov10s.pt")
    parser.add_argument("--task", type=str, default="detect")
    parser.add_argument("--dataset", type=str, default="coco_wb.yaml")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=list, default=['0'])
    parser.add_argument("--end2end", action="store_true", default=True)
    parser.add_argument("--no_en2end", action="store_false", dest="end2end", default=False)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", type=str, default="ultralytics-runs")
    parser.add_argument("--name", type=str, default="pred")
    args = parser.parse_args()

    print("ARGS:", args)

    model = YOLO(args.pt, task=args.task)

    del model.model.model[-1].cv2
    del model.model.model[-1].cv3
    del model.model.model[-1].cv4

    with open(args.dataset, "r") as f:
        data = yaml.safe_load(f)

        directories = os.path.join(data["path"][1:], data["val"])

        with open(directories, "r") as f:
            images = f.readlines()

            len_images = int(len(images) * args.fraction)

            for i in range(len_images):

                model.predict(
                    task=args.task,
                    source=os.path.join(data["path"][1:], images[i][2:-1]),
                    device=args.device,
                    end2end=args.end2end,
                    project=args.project,
                    name=args.name,
                    save=True
                )