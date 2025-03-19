from ultralytics import YOLOE

unfused_model = YOLOE("yoloe-v8l.yaml")
unfused_model.load("pretrain/yoloe-v8l-seg.pt")
unfused_model.eval()
unfused_model.cuda()

with open("tools/ram_tag_list.txt") as f:
    names = [x.strip() for x in f.readlines()]
vocab = unfused_model.get_vocab(names)

model = YOLOE("pretrain/yoloe-v8l-seg-pf.pt").cuda()
model.set_vocab(vocab, names=names)
model.model.model[-1].is_fused = True
model.model.model[-1].conf = 0.001
model.model.model[-1].max_det = 1000

filename = "ultralytics/cfg/datasets/lvis.yaml"

model.predict("ultralytics/assets/bus.jpg", save=True)
