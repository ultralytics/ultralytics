

from ultralytics import YOLOE

scale="26x"
end2end=True
model=f"weights/yoloe-{scale}-seg-pf.pt"

model=YOLOE(model)
model.args['clip_weight_name']="mobileclip2:b"


if not end2end:
    model.model.end2end=False
    model.model.model[-1].end2end=False
else:
    model.model.end2end=True
    model.model.model[-1].end2end=True


# infer test
img_path="./ultralytics/assets/bus.jpg"
results=model.predict(img_path,conf=0.25)
func_name="infer_yoloe26s_pf"
results[0].save("runs/res-{scale}.jpg".format(scale=scale))

