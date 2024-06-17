
from ultralytics import YOLO

import argparse

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=None, help='default.yaml path')
    parser.add_argument('--model-cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', nargs='+', default=[0], help='cuda device, i.e. [0 1]')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--patience', type=int, default=0, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--cache', action='store_true', help='')
    parser.add_argument('--close-mosaic', type=int, default=3, help='(int) disable mosaic augmentation for final epochs')
    parser.add_argument('--resume', action='store_true', help='resume most recent training')
    
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate (i.e. SGD=1E-2, Adam=1E-3)')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate (lr0 * lrf)')
    parser.add_argument('--cos-lr', action='store_true', help='cosine learning rate schedule')    
    # Augmentation
    parser.add_argument('--scale', type=float, default=0.5, help='image scale (+/- gain)')
    parser.add_argument('--mixup', type=float, default=0.00, help='image mixup (probability)')
    parser.add_argument('--copy-paste', type=float, default=0.00, help='copy-paste (probability)')
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    # Create a dictionary of kwargs
    kwargs = vars(args)

    if args.model_cfg != '':
        # Create a new YOLO model from scratch
        model = YOLO(args.model_cfg)
        if args.weights != '':
            # transfer weights
            model.load(args.weights) 
    elif args.weights != '':
        # Load a pretrained YOLO model (recommended for training)
        model = YOLO(args.weights)
    else:
        raise SystemError('--cfg or --weights must be provided')

    del kwargs['model_cfg']
    del kwargs['weights']

    # Train the model
    print(kwargs)
    results = model.train(**kwargs)

    # # Evaluate the model's performance on the validation set
    # results = model.val(**kwargs)
