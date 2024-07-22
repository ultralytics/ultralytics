from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
import argparse
from copy import deepcopy


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_flops_from_model(model, imgsz, should_forward_one2many_head=False, verbose = False,):
    layers_flops_incremental = []

    for name in layers_names[::-1]:
        flops = get_flops(model.model, imgsz=imgsz, should_forward_one2many_head=should_forward_one2many_head)
        if verbose:
            print(f"Incremental Layer: {name}, FLOPS: {flops}")
        layers_flops_incremental.append(flops)
        del model.model.model[-1]

    layers_flops = [layers_flops_incremental[-1]]
    for i in range(len(layers_flops_incremental) - 1, 0, -1):
        layers_flops.append(layers_flops_incremental[i-1] - layers_flops_incremental[i])
        
    for name, flops in zip(layers_names, layers_flops):
        print(f"{name[0]}_{name[1]}: {round(flops, 3) if flops != 0 else 0} GFLOPs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, default="yolov10s.pt")
    parser.add_argument("--task", type=str, default="detect")
    parser.add_argument("--dataset", type=str, default="coco_wb.yaml")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=list, default=['0'])
    parser.add_argument("--end2end", action="store_true", default=True)
    parser.add_argument("--no_en2end", action="store_false", dest="end2end", default=False)
    parser.add_argument("--project", type=str, default="ultralytics-runs")
    parser.add_argument("--name", type=str, default="val")
    args = parser.parse_args()

    print("ARGS:", args)

    model = YOLO(args.pt, task=args.task)

    model.overrides["task"] = args.task
    model.overrides["data"] = args.dataset
    model.overrides["batch"] = args.batch
    model.overrides["device"] = args.device
    model.overrides["end2end"] = args.end2end
    model.overrides["project"] = args.project
    model.overrides["name"] = args.name
    model.overrides["save"] = False
    model.overrides["mode"] = "val"

    model_640_one2many = deepcopy(model)
    model_640 = deepcopy(model)
    model_320_one2many = deepcopy(model)
    model_320 = deepcopy(model)

    layers_names = [
        (name, str(type(module)).split(".")[-1][:-2]) 
        for name, module in 
        [m for m in model.model.children()][0].named_children()
    ] # last to first


    # ---------------------------- 640x640 ---------------------------- #
    imgsz = 640
    print(
        bcolors.HEADER, 
        "-"*10, f"Igmsz: {imgsz}x{imgsz}", "-"*10, 
        bcolors.ENDC
    )


    # --------- WITH one2many HEAD --------- #
    should_forward_one2many_head = True
    print(
        bcolors.WARNING, 
        "-"*10, f"WITH one2many HEAD", "-"*10, 
        bcolors.ENDC
    )

    print(
        bcolors.OKBLUE,
        f"All GFLOPS:",
        get_flops(model_640_one2many.model, imgsz=imgsz, should_forward_one2many_head=should_forward_one2many_head),
        bcolors.ENDC
    )

    get_flops_from_model(model_640_one2many, imgsz=imgsz, should_forward_one2many_head=should_forward_one2many_head, verbose=False)

    # --------- WITHOUT one2many HEAD --------- #
    should_forward_one2many_head = False
    print(
        bcolors.WARNING, 
        "-"*10, f"WITHOUT one2many HEAD", "-"*10, 
        bcolors.ENDC
    )

    print(
        bcolors.OKBLUE,
        f"All GFLOPS:",
        get_flops(model_640.model, imgsz=imgsz, should_forward_one2many_head=should_forward_one2many_head),
        bcolors.ENDC
    )

    get_flops_from_model(model_640, imgsz=imgsz, should_forward_one2many_head=should_forward_one2many_head, verbose=False)


    # ---------------------------- 320x320 ---------------------------- #
    imgsz = 320
    print(
        bcolors.HEADER, 
        "-"*10, f"Igmsz: {imgsz}x{imgsz}", "-"*10, 
        bcolors.ENDC
    )

    # --------- WITH one2many HEAD --------- #
    should_forward_one2many_head = True
    print(
        bcolors.WARNING, 
        "-"*10, f"WITH one2many HEAD", "-"*10, 
        bcolors.ENDC
    )

    print(
        bcolors.OKBLUE,
        f"All GFLOPS:",
        get_flops(model_320_one2many.model, imgsz=imgsz, should_forward_one2many_head=should_forward_one2many_head),
        bcolors.ENDC
    )

    get_flops_from_model(model_320_one2many, imgsz=imgsz, should_forward_one2many_head=should_forward_one2many_head, verbose=False)

    # --------- WITHOUT one2many HEAD --------- #
    should_forward_one2many_head = False
    print(
        bcolors.WARNING, 
        "-"*10, f"WITHOUT one2many HEAD", "-"*10, 
        bcolors.ENDC
    )

    print(
        bcolors.OKBLUE,
        f"All GFLOPS:",
        get_flops(model_320.model, imgsz=imgsz, should_forward_one2many_head=should_forward_one2many_head),
        bcolors.ENDC
    )

    get_flops_from_model(model_320, imgsz=imgsz, should_forward_one2many_head=should_forward_one2many_head, verbose=False)

