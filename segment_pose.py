import argparse

from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Process SegmentPoseModel.')
parser.add_argument('-w','--weights', default='', type=str, help='path to weights model')
parser.add_argument('--mode', default='train', type=str, help='train | val | export | predict expects')
parser.add_argument('-m','--model', default='m', type=str, help='n | s | m | l | x expects')
parser.add_argument('-i','--images', nargs='*', help='list of paths to images to predict')

'''
    Commands checked:

    python3 segment_pose.py --mode=val --weights=yolov8n-segpose.pt --model=n
    python3 segment_pose.py --mode=predict --weights=yolov8n-segpose.pt --model=n --images ultralytics/assets/bus.jpg ultralytics/assets/zidane.jpg
    python3 segment_pose.py --mode=train --weights=yolov8n-segpose.pt --model=n
    python3 segment_pose.py --mode=export --weights=yolov8n-segpose.pt --model=n
'''

if __name__ == '__main__':
    args = parser.parse_args()
    weights = args.weights if args.weights else f'yolov8{args.model}-seg.pt'
    model = YOLO(model=f'yolov8{args.model}-segpose.yaml', task='segment_pose').load(weights)
    if args.mode == 'train':
        model.train(data='coco-segment_pose.yaml', name=f'train_segmentposecoco_{args.model}', lr0=1e-4, epochs=3000)
        # model = YOLO(model=args.weights, task='segment_pose')
        # model.train(resume=True)
    elif args.mode == 'val':
        model.val(data='coco-segment_pose.yaml', name=f'val_segmentposecoco_{args.model}')
    elif args.mode == 'export':
        model.export(format='onnx', device=0, simplify=True, dynamic=True)
    elif args.mode == 'predict':
        if len(args.images):
            model.predict(source=args.images, name=f'predict_segmentposecoco_{args.model}', imgsz=640, save=True, task='segment_pose')
    else:
        print('train | val | export expects')
