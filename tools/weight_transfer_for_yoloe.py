
import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)

from ultralytics import YOLO

import os, sys


def init_yoloe_from_ptw(model,weight_path):


    model= YOLO(model).load(weight_path)

    ptw=YOLO(weight_path)

    # copy weights from ptw to model
    model_head=model.model.model[-1]
    ptw_head=ptw.model.model[-1]
    # copy ptw.cv2 to model.cv2
    model_head.cv2.load_state_dict(ptw_head.cv2.state_dict())
    model_head.one2one_cv2.load_state_dict(ptw_head.one2one_cv2.state_dict())


    # copy ptw.cv3[0] to model.cv3[0] #cls head
    for i in range(3):
        model_head.cv3[i][0].load_state_dict(ptw_head.cv3[i][0].state_dict())
        model_head.one2one_cv3[i][0].load_state_dict(ptw_head.one2one_cv3[i][0].state_dict())
        # copy ptw.cv3[1] to model.cv3[1]
        model_head.cv3[i][1].load_state_dict(ptw_head.cv3[i][1].state_dict())
        model_head.one2one_cv3[i][1].load_state_dict(ptw_head.one2one_cv3[i][1].state_dict())


    # copy ptw.cv4 to model.cv5    (different segmentation head)
    model_head.cv5.load_state_dict(ptw_head.cv4.state_dict())

    # copy ptw.proto to model.proto except the proto.semseg 
    for key in ptw_head.proto.state_dict().keys():
        if "semseg" not in key:
            model_head.proto.state_dict()[key].data.copy_(ptw_head.proto.state_dict()[key].data.clone())

    return model


for scale in ["26s","26m","26l"]:
    model=init_yoloe_from_ptw("yoloe-{}-seg.yaml".format(scale),"weights/yolo{}-objv1-seg.pt".format(scale))
    model.save("weights/yolo{}-objv1-seg[foryoloe].pt".format(scale))
    print("Converted yoloe-{}-seg weights saved.".format(scale))
