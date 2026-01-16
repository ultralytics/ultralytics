




from ultralytics import YOLO


def print_model(model_weight):

    model=YOLO(model_weight)
    model.args['clip_weight_name']="mobileclip2:b"
    model_head=model.model.model[-1]
    print(model_head.cv3)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--model_weight', type=str, default="weights/yoloe-26n-seg-pf.pt", help='path to the model weight file')
args = parser.parse_args()
print_model(args.model_weight)