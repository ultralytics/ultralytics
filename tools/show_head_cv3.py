
from ultralytics import YOLOE


def print_model(model_weight):

    model=YOLOE(model_weight)
    model.args['clip_weight_name']="mobileclip2:b"
    model_head=model.model.model[-1]
    print(model_head.cv3)



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, default='weights/yoloe-26s-seg-pf.pt', help='path to the model weight')
    args = parser.parse_args()

    print_model(args.model_weight)
