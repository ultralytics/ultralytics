
from ultralytics import YOLO

import argparse

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--annotations', type=str, default=None, help='ground truth annotation json file path')
    parser.add_argument('--device', type=str, default='cpu', help='device to use')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--no-separate-outputs', action='store_true', help='exported file without separate outputs')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    # Create a dictionary of kwargs
    kwargs = vars(args)

    print(kwargs)

    model = YOLO(args.weights)

    separate_outputs = False if (args.weights.endswith('.pt') or args.no_separate_outputs) else True
    save_json = True if args.annotations else False
    success = model.val(data=args.data, imgsz=args.imgsz, rect=False, device=args.device, separate_outputs=separate_outputs, save_json=save_json, anno_json=args.annotations)

    
