import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)







# check open-clip-torch package installation, if not 
def check_open_clip_installed():
    """
    Check if the 'open-clip-torch' package is installed.
    If not, raise an ImportError with installation instructions. mobileclip2 is added in open-clip-torch (https://github.com/ShuaiLYU/open_clip)
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("Please download and install the 'open-clip-torch' package from https://github.com/ShuaiLYU/open_clip.")

def check_mobileclip2_weight():
    # check file "mobileclip2_b.pt" in the workspace
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
    if not os.path.exists(os.path.join(workspace, "mobileclip2_b.pt")):
        raise FileNotFoundError("Please download the file 'mobileclip2_b.pt' and place it in the workspace directory.")


def check_lvis_train_vps():
    """
    check lvis_train_vps.yaml file exists in the datasets folder. if not, genereate it with "https://github.com/ShuaiLYU/yoloe/blob/main/tools/generate_lvis_visual_prompt_data.py"

    """
    refer_data_path = "../datasets/lvis_train_vps.yaml"
    if not os.path.exists(refer_data_path):
        raise FileNotFoundError(f"File '{refer_data_path}' not found. Please generate it using 'https://github.com/ShuaiLYU/yoloe/blob/main/tools/generate_lvis_visual_prompt_data.py'")
    
    else:
        print(f"File '{refer_data_path}' found.")



from ultralytics import YOLOE



def val_yoloe26(model_path,mode,end2end=True,device="0"):
    """
    validate yoloe26 with text prompt or visual prompt.
    Args:
        model_path: str, path to the model weight
        mode: str, "text_prompt" or "visual_prompt"
        end2end: bool, whether to use end2end model, default True. It will achieve better performance when end2end is False.
    Returns:
        model: YOLOE model with validation results. Access model.metrics for DetMetrics, model.val_stats for full COCO eval results
    """

    assert mode in ["text_prompt","visual_prompt"]

    model = YOLOE(model_path)
    model.args["clip_weight_name"]="mobileclip2:b"
    if not end2end:
        del model.model.model[-1].one2one_cv2
        del model.model.model[-1].one2one_cv3
        del model.model.model[-1].one2one_cv4
        model.model.end2end = False


    if mode=="text_prompt":

        data="../datasets/lvis.yaml"

        metrics = model.val(data=data, split="minival", max_det=1000,  save_json=True,device=device)

    if mode=="visual_prompt":

        check_lvis_train_vps()
        data="../datasets/lvis.yaml"
        refer_data="../datasets/lvis_train_vps.yaml"

        metrics = model.val(data=data, split="minival", max_det=1000,load_vp=True,refer_data=refer_data,  save_json=True,device=device)
    return model



if __name__=="__main__":


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='cuda device(s) to use')
    parser.add_argument('-model_weight', type=str, default="./weights/yoloe-26s.pt", help='path to model weight')
    args = parser.parse_args()

    check_open_clip_installed()
    check_mobileclip2_weight()
    model_weight=args.model_weight
    device=args.device

    if not os.path.exists(model_weight):
        raise FileNotFoundError(f"Please download the file '{model_weight}' and place it in the 'weights' directory.")


    results={}

    print("Validating YOLOE26 with Text Prompt...")
    model=val_yoloe26(model_weight,mode="text_prompt",end2end=True,device=device)
    results["tp_end2end"] = model.val_stats if hasattr(model, 'val_stats') else model.metrics.results_dict
    tp_metrics=val_yoloe26(model_weight,mode="text_prompt",end2end=False,device=device)
    results["tp_not_end2end"]=tp_metrics.val_stats if hasattr(tp_metrics, 'val_stats') else tp_metrics.metrics.results_dict


    print("Validating YOLOE26 with Visual Prompt...")
    vp_metrics=val_yoloe26(model_weight,mode="visual_prompt",end2end=True,device=device)
    results["vp_end2end"]=vp_metrics.val_stats if hasattr(vp_metrics, 'val_stats') else vp_metrics.metrics.results_dict
    vp_metrics=val_yoloe26(model_weight,mode="visual_prompt",end2end=False,device=device)
    results["vp_not_end2end"]=vp_metrics.val_stats if hasattr(vp_metrics, 'val_stats') else vp_metrics.metrics.results_dict
    
    print("\n" + "="*80)
    print("Validation Results (IoU=0.50:0.95):")
    print("="*80)
    for k,v in results.items():
        print(f"{k:20s}: mAP50-95={v['metrics/mAP50-95(B)']:.3f}, mAP50={v['metrics/mAP50(B)']:.3f}, P={v['metrics/precision(B)']:.3f}, R={v['metrics/recall(B)']:.3f}")







    
