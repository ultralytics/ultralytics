import os 
os.chdir("/home/louis/ultra_louis_work/ultralytics")

from ultralytics import YOLOE


unfused_model = YOLOE("/home/louis/ultra_louis_work/ultralytics/yoloe-v8s-seg.pt")
# unfused_model.load("pretrain/yoloe-v8l-seg.pt")
unfused_model.eval()
unfused_model.cuda()

with open('/home/louis/ultra_louis_work/yoloe/tools/ram_tag_list.txt', 'r') as f:
    names = [x.strip() for x in f.readlines()]
vocab = unfused_model.get_vocab(names)

# model = YOLOE("pretrain/yoloe-v8l-seg-pf.pt").cuda()
# model.set_vocab(vocab, names=names)
# model.model.model[-1].is_fused = True
# model.model.model[-1].conf = 0.001
# model.model.model[-1].max_det = 1000

# filename = "ultralytics/cfg/datasets/lvis.yaml"

# model.predict('ultralytics/assets/bus.jpg', save=True)




def run_val(**kwargs):

    assert "single_cls"  in kwargs.keys(), "need single_cls in kwargs"

    model = YOLOE(kwargs["model_weight"])
    model.set_vocab(vocab, names=names)
    model.model.model[-1].is_fused = True
    model.model.model[-1].conf = 0.1 #
    model.model.model[-1].max_det = 1000


    data=kwargs["data"]

    batch=kwargs["batch"]
    split=kwargs["split"]
    plots=kwargs["plots"]
    save_json=kwargs["save_json"]
    max_det=kwargs["max_det"]
    single_cls=kwargs["single_cls"]
    assert single_cls is True, "only support single_cls=True for prompt free val"
    

    metrics = model.val(data=data, split=split, save_json=save_json, batch=batch,
                        max_det=max_det, plots=plots,single_cls=single_cls)

    states=dict(**kwargs)

    if isinstance(metrics, dict):
        for k, v in metrics.items():
            states[k] = v
    else:
        states["metrics"]=metrics
    return states


model_weight="/home/louis/ultra_louis_work/ultralytics/runs/detect/train14/weights/best.pt"


# model_weight=os.path.abspath("./yoloe-v8s-seg-pf.pt")
print(f"model_weight: {model_weight}")

kwargs= dict(model_weight=model_weight,
              max_det=1000,
              batch=64,
              data="./train_vp/lvis.yaml",
              split='minival',
              plots=False,
              save_json=True,
              single_cls=True
)

states=run_val(**kwargs)



######################################################################
# # test the val mode

# from ultralytics import YOLOE
# model_weight=os.path.abspath("./yoloe-v8s-seg-pf.pt")
# # Create a YOLOE model
# model = YOLOE(model_weight)  # or select yoloe-11s/m-seg-pf.pt for different sizes

# # Conduct model validation on the COCO128-seg example dataset
# metrics = model.val(data="coco128-seg.yaml", single_cls=True)


######################################################################
# from ultralytics import YOLOE
# model_weight=os.path.abspath("./yoloe-v8s-seg-pf.pt")

# def print_model_structure(model_path):
#     """
#     read the model with torch and print the model structure

#     """
#     import torch
#     model = torch.load(model_path, map_location='cpu',weights_only=False)['model'].float()  # load to CPU
#     print(model)
    

# print_model_structure(model_weight)
