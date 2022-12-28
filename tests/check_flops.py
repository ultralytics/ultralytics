import argparse
from ultralytics import YOLO
# import ultralytics.yolo.v8
from ultralytics.yolo.utils.checks import check_yaml, print_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov8n-seg.yaml', help='model.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))

    model = YOLO()
    model.new(opt.cfg)
