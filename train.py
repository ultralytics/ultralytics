from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, default="yolov10s.pt")
    parser.add_argument("--task", type=str, default="detect")
    parser.add_argument("--dataset", type=str, default="coco_wb.yaml")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--device", type=list, default=['0'])
    parser.add_argument("--end2end", action="store_true", default=True)
    parser.add_argument("--no_en2end", action="store_false", dest="end2end", default=False)
    parser.add_argument("--project", type=str, default="ultralytics-runs")
    parser.add_argument("--name", type=str, default="train")
    args = parser.parse_args()

    print("ARGS:", args)

    model = YOLO(args.pt, task=args.task)

    model.train(
        task=args.task,
        data=args.dataset, 
        fraction=args.fraction,
        epochs=args.epochs, 
        batch=args.batch,
        pretrained=args.pretrained,
        device=args.device,
        end2end=args.end2end,
        project=args.project,
        name=args.name
    )