

from ultralytics import YOLO


def get_yoloe_info(model ,end2end=None):
    """
    get yoloe model info
    Args:
        model: str, model path
        end2end: bool or None, whether to use end2end model, default None. It will achieve better performance when end2end is False.
    """
    
    assert end2end in [True,False,None], "end2end must be True, False, or None"
    if end2end is None:
        model.fuse()
        model.info()
        return
    
    if end2end==True:
        model_head=model.model.model[-1]
        if hasattr(model_head, 'cv2'):
            print("has cv2, delete cv2")
            del model_head.cv2
        if hasattr(model_head, 'cv3'):
            del model_head.cv3
            print("has cv3, delete cv3")
        if hasattr(model_head, 'cv4'):
            del model_head.cv4
            print("has cv4, delete cv4")
        if hasattr(model_head, 'cv5'):
            del model_head.cv5
            print("has cv5, delete cv5")
        model.model.end2end = True
        model.fuse()
        model.info()
    else:

        if hasattr(model.model.model[-1], 'one2one_cv2'):
            del model.model.model[-1].one2one_cv2
            print("has one2one_cv2, delete one2one_cv2")
        if hasattr(model.model.model[-1], 'one2one_cv3'):
            del model.model.model[-1].one2one_cv3
            print("has one2one_cv3, delete one2one_cv3")
        if hasattr(model.model.model[-1], 'one2one_cv4'):
            del model.model.model[-1].one2one_cv4
            print("has one2one_cv4, delete one2one_cv4")
        if hasattr(model.model.model[-1], 'one2one_cv5'):
            del model.model.model[-1].one2one_cv5
            print("has one2one_cv5, delete one2one_cv5")
        model.model.end2end = False
        model.fuse()
        model.info()


print("="*100)
for scale in ["n","s","m","l","x"]:

    print(f"----- get_info for yoloe-26{scale}-seg-pf.pt -----")
    
    # weight_dir="/Users/louis/workspace/ultra_louis_work/yoloe26_weight/yoloe26_seg_pf/"

    # model=YOLO(f"{weight_dir}/yoloe-26{scale}-seg-pf.pt")
    # get_yoloe_info(model ,end2end=True)



    weight_dir="/Users/louis/workspace/ultra_louis_work/yoloe26_weight/yoloe26_seg_pf/"
    model=YOLO(f"{weight_dir}/yoloe-26{scale}-seg-pf.pt")
    
    del model.model.model[-1].cv2
    del model.model.model[-1].cv3
    del model.model.model[-1].cv4
    del model.model.model[-1].cv5

    model.fuse()
    model.info()