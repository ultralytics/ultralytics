import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


# 9801e33af3c11010f9ce52bbe95da507c4bab584
#  * (HEAD detached at 9801e33a) 9801e33a add show_head_cv3.py

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator


from pathlib import Path
import re,yaml

def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data




def fuse_tp_vocab_into_pf_model(pf_model_weight,yaml_name,tp_model_weight,clip_weight_name="mobileclip2:b",open_ended_classes="./tools/open_ended_classes.txt"):
    """
        This function fuses the vocabulary from a TP model into a PF model.
        First tp_model.get_vocab() is used to merge the tpe, bn_head, cls_head[-1] into one conv layer with vocab size.
        Then pf_model.set_vocab() is used to set the fused vocab into the pf model.

    Args:
        pf_model_weight (str): Path to the PF model weights.
        yaml_name (str): YAML configuration file name.
        tp_model_weight (str): Path to the TP model weights.
        clip_weight_name (str): Name of the CLIP weight to use. Default is "mobileclip2:b".
        open_ended_classes (str): Path to the file containing open-ended class names.
    Returns:
        YOLOE: The fused PF model with the TP vocabulary.
    """

    # set vocab from the unfused model
    tp_model=YOLOE(tp_model_weight)
    tp_model.model.args['clip_weight_name']=clip_weight_name
    tp_model.eval()
    tp_model.cuda()
    tp_model.args['clip_model_weight']=clip_weight_name
    with open('../buffer/ram_tag_list.txt', 'r') as f:
        names = [x.strip() for x in f.readlines()]
    tp_model.model.end2end = True
    # tp_model.model.model[-1].end2end = True
    vocab = tp_model.get_vocab(names)


    if scale == "26n":
        # pf_model = YOLOE(yaml_name).load(pf_model_weight)
        pf_model=YOLOE(pf_model_weight)
    else:
        pf_model=YOLOE(pf_model_weight)
    pf_model.model.args['clip_weight_name']=clip_weight_name
    pf_model.eval()
    pf_model.cuda()

    pf_model.model.end2end = True
    # pf_model.model.model[-1].end2end = True
    pf_model.set_vocab(vocab, names=names)

    pf_model.model.model[-1].is_fused = True
    pf_model.model.model[-1].conf = 0.001
    pf_model.model.model[-1].max_det = 1000
    
    return pf_model





def merge_pf_to_seg(model,seg_model):
    """
        merge the pf model to seg model
    """
    from copy import deepcopy
    import torch.nn as nn
    seg_model_head=seg_model.model.model[-1]

    # cv2=seg_model_head.cv2
    # cv3=seg_model_head.cv3    
    # cv4=seg_model_head.cv4 

    # for loc_head, cls_head in zip(cv2, cv3):
    #     assert isinstance(loc_head, nn.Sequential)
    #     assert isinstance(cls_head, nn.Sequential)
    #     del loc_head[-1]
    #     del cls_head[-1]
    # cv4[0].fuse()
    # cv4[1].fuse()
    # cv4[2].fuse()
    seg_model_head.lrpc = deepcopy(model.model.model[-1].lrpc)

    cv2=seg_model_head.one2one_cv2
    cv3=seg_model_head.one2one_cv3    
    cv4=seg_model_head.one2one_cv4 

    for loc_head, cls_head in zip(cv2, cv3):
        assert isinstance(loc_head, nn.Sequential)
        assert isinstance(cls_head, nn.Sequential)
        del loc_head[-1]
        del cls_head[-1]
    cv4[0].fuse()
    cv4[1].fuse()
    cv4[2].fuse()
    # seg_model_head.one2one_lrpc = deepcopy(model.model.model[-1].one2one_lrpc)




    names=model.model.names
    assert len(names) ==4585

    print("set seg model classes num:", len(names))

    seg_model_head.nc = len(names)

    from ultralytics.nn.autobackend import check_class_names    
    seg_model.model.names = check_class_names(names)
    seg_model.model.model.names = check_class_names(names)
    seg_model.model.nc=len(names)
    seg_model.model.model.nc=len(names)

    seg_model.model.model[-1].is_fused = True
    seg_model.model.model[-1].conf = 0.001
    seg_model.model.model[-1].max_det = 1000


    return seg_model





# yoloe26n_pf="runs/yoloe26_pf/26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
yoloe26s_pf="runs/yoloe26_pf/26s_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
yoloe26m_pf="runs/yoloe26_pf/26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
yoloe26l_pf="runs/yoloe26_pf/26l_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6]/weights/best.pt"
yoloe26x_pf="runs/yoloe26_pf/26x_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6]/weights/best.pt"

yoloe26n_tp="runs/yoloe26_tp/26n_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt"
yoloe26s_tp="runs/yoloe26_tp/26s_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt"
yoloe26m_tp="runs/yoloe26_tp/26m_ptwobjv1_bs256_epo25_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt"
yoloe26l_tp="runs/yoloe26_tp/26l_ptwobjv1_bs256_epo20_close2_engine_old_engine_data_tp[ultra6]/weights/best.pt"
yoloe26x_tp="runs/yoloe26_tp/26x_ptwobjv1_bs256_epo15_close2_engine_old_engine_data_tp[ultra6]/weights/best.pt"

yoloe26n_pf="runs/yoloe26_pf/26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf2[ultra8]/weights/best.pt"


for scale, pf_model, tp_model in zip(["26n"],
                             [yoloe26n_pf,yoloe26s_pf,yoloe26m_pf,yoloe26l_pf,yoloe26x_pf],
                             [yoloe26n_tp,yoloe26s_tp,yoloe26m_tp,yoloe26l_tp,yoloe26x_tp]):
    # seg_model=f"weights/yoloe-{scale}-seg.pt" 

    dst_path=f"weights/yoloe-{scale}-seg-pf-new.pt"

 
    model=fuse_tp_vocab_into_pf_model(pf_model,f"yoloe-{scale}.yaml",tp_model)


    seg_model_pt=f"weights/yoloe-{scale}-seg.pt"
    seg_model=YOLOE(seg_model_pt)


    seg_model=merge_pf_to_seg(model,seg_model)

    seg_model.save(dst_path)


