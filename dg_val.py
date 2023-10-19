
from ultralytics import YOLO

import argparse

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path')
    parser.add_argument('--data', type=str, default='coco.yaml', help='dataset.yaml path')
    parser.add_argument('--anno', type=str, help='dataset.yaml path')
    parser.add_argument('--device', type=str, default='cuda:7', help='device to use')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    # Create a dictionary of kwargs
    kwargs = vars(args)

    print(kwargs)

    model = YOLO(args.weights)

    separate_outputs = False if args.weights.endswith('.pt') else True
    success = model.val(data=args.data, imgsz=args.imgsz, rect=False, device=args.device, separate_outputs=separate_outputs, save_json=True)

    
