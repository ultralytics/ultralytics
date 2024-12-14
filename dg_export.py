
from ultralytics import YOLO

import argparse

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--format', type=str, default='tflite', help='export format')
    parser.add_argument('--quantize', action='store_true', help='int8 export')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--max_ncalib_imgs', type=int, default=100, help='calibration image set size')	
    parser.add_argument('--no-separate-outputs', action='store_true', help='do not separate outputs')
    parser.add_argument('--separate-pose', action='store_true', help='separate pose outputs')
    parser.add_argument('--separate-masks', action='store_true', help='separate masks outputs')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    # Create a dictionary of kwargs
    kwargs = vars(args)
    print(kwargs)

    model = YOLO(args.weights)

    success = model.export(format=args.format, simplify=True, imgsz=args.imgsz, data=args.data, int8=args.quantize, separate_outputs=not args.no_separate_outputs, export_hw_optimized=True, uint8_io_dtype=True, max_ncalib_imgs=args.max_ncalib_imgs, separate_pose=args.separate_pose, separate_masks=args.separate_masks)


