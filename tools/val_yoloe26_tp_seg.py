import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)





from ultralytics import YOLOE





import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='cuda device(s) to use')
parser.add_argument('--model_weight', type=str, default="./weights/yoloe-26s.pt", help='path to model weight')
parser.add_argument('--batch', type=int, default=1, help='batch size for validation')
parser.add_argument('--version', type=str, default='26s', help='model version')
args = parser.parse_args()



# model=YOLOE("yoloe-{args.version}-seg.yaml").load(args.model_weight).to("cuda:"+args.device)

model=YOLOE(args.model_weight).to("cuda:"+args.device)




metrics = model.val(data="../datasets/lvis.yaml", split="val", max_det=1000, batch=args.batch, save_json=True,task="segment")       