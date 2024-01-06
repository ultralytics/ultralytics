
from ultralytics import YOLO

import argparse

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path')
    parser.add_argument('--source', type=str, help='dataset.yaml path')
    parser.add_argument('--device', type=str, default='0', help='device to use')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    # Create a dictionary of kwargs
    kwargs = vars(args)

    print(kwargs)

    model = YOLO(args.weights)

    separate_outputs = False if args.weights.endswith('.pt') else True
    success = model.predict(args.source, imgsz=args.imgsz, rect=False, device=args.device, save=True, separate_outputs=separate_outputs)

    
