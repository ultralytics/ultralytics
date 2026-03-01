# from ultralytics import RTDETRDual as RTDETR
# from ultralytics import RTDETR as RTDETR
from ultralytics import YOLO as YOLO

device = "0"
# import os
# os.environ['WANDB_MODE'] = 'disabled'
"""
'augment', 'verbose', 'project', 'name', 'exist_ok', 'resume', 
'batch', 'epochs', 'cache', 'save_json', 'half', 'v5loader', 
'device', 'cfg', 'save', 'rect', 'plots':
"""

"""
https://docs.ultralytics.com/cfg/

Train:
Key	                Value	        Description
model	            null	        path to model file, i.e. yolov8n.pt, yolov8n.yaml
data	            null	        path to data file, i.e. i.e. coco128.yaml
epochs	            100	            number of epochs to train for
patience	        50	            epochs to wait for no observable improvement for early stopping of training
batch	            16	            number of images per batch (-1 for AutoBatch)
imgsz	            640	            size of input images as integer or w,h
save	            True	        save train checkpoints and predict results
save_period	        -1	            Save checkpoint every x epochs (disabled if < 1)
cache	            False	        True/ram, disk or False. Use cache for data loading
device	            null	        device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
workers	            8	            number of worker threads for data loading (per RANK if DDP)
project	            null	        project name
name	            null	        experiment name
exist_ok	        False	        whether to overwrite existing experiment
pretrained	        False	        whether to use a pretrained model
optimizer	        'SGD'	        optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
verbose	            False	        whether to print verbose output
seed	            0	            random seed for reproducibility
deterministic	    True	        whether to enable deterministic mode
single_cls	        False	        train multi-class data as single-class
image_weights	    False	        use weighted image selection for training
rect	            False	        support rectangular training
cos_lr	            False	        use cosine learning rate scheduler
close_mosaic	    10	            disable mosaic augmentation for final 10 epochs
resume	            False	        resume training from last checkpoint
lr0	0.01	        initial         learning rate (i.e. SGD=1E-2, Adam=1E-3)
lrf	0.01	        final           learning rate (lr0 * lrf)
momentum	        0.937	        SGD momentum/Adam beta1
weight_decay	    0.0005	        optimizer weight decay 5e-4
warmup_epochs	    3.0	            warmup epochs (fractions ok)
warmup_momentum	    0.8	            warmup initial momentum
warmup_bias_lr	    0.1	            warmup initial bias lr
box	                7.5	            box loss gain
cls	                0.5	            cls loss gain (scale with pixels)
dfl	                1.5	            dfl loss gain
fl_gamma	        0.0	            focal loss gamma (efficientDet default gamma=1.5)
label_smoothing	    0.0	            label smoothing (fraction)
nbs	                64	            nominal batch size
overlap_mask	    True	        masks should overlap during training (segment train only)
mask_ratio	        4	            mask downsample ratio (segment train only)
dropout	            0.0	            use dropout regularization (classify train only)
val	                True	        validate/test during training
min_memory	        False	        minimize memory footprint loss function, choices=[False, True, ]
"""


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def train(
    path_yaml: str,
    data_path: str,
    init_weight="",
    device="0",
    project="",
    name="",
    epochs=300,
    batch=16,
    imgsz=640,
    workers=8,
    patience=50,
    save_period=-1,
    seed=0,
) -> None:
    model = YOLO(path_yaml)  # build a new model from scratch
    if init_weight != "":
        import torch

        ckpt = torch.load(init_weight, map_location="cpu")
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.model.state_dict())  # intersect
        model.model.load_state_dict(csd, strict=False)  # load
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        project=project,
        name=name,
        workers=workers,
        patience=patience,
        save_period=save_period,
        seed=seed,
    )


def train_resume(init_weight: str, data_path: str, device="0") -> None:
    model = YOLO(init_weight)
    print(init_weight)
    model.train(data=data_path, epochs=50, imgsz=640, device=device, batch=4, workers=4, resume=True, exist_ok=True)


"""
Predict:
Key	                Value	        Description
source	            'assets'	    source directory for images or videos
conf	            0.25	        object confidence threshold for detection
iou	                0.7	            intersection over union (IoU) threshold for NMS
half	            False	        use half precision (FP16)
device	            null	        device to run on, i.e. cuda device=0/1/2/3 or device=cpu
show	            False	        show results if possible
save	            False	        save images with results
save_txt	        False	        save results as .txt file
save_conf	        False	        save results with confidence scores
save_crop	        False	        save cropped images with results
hide_labels	        False	        hide labels
hide_conf	        False	        hide confidence scores
max_det	            300	            maximum number of detections per image
vid_stride	        False	        video frame-rate stride
line_thickness	    3	            bounding box thickness (pixels)
visualize	        False	        visualize model features
augment	            False	        apply image augmentation to prediction sources
agnostic_nms	    False	        class-agnostic NMS
retina_masks	    False	        use high-resolution segmentation masks
classes	            null	        filter results by class, i.e. class=0, or class=[0,2,3]
box	                True	        Show boxes in segmentation predictions
"""


def predict(path_yaml: str, img_path: str, device="cpu", name="") -> None:
    model = YOLO(path_yaml)
    name = "predict/" + name
    model.predict(
        img_path,
        conf=0.3,
        iou=0.45,
        device=device,
        save=True,
        save_conf=False,
        save_txt=False,
        name=name,
        show_conf=False,
        show_labels=False,
        line_width=2,
        max_det=500,
    )


"""
val:
Key	                Value	        Description
save_json	        False	        save results to JSON file
save_hybrid	        False	        save hybrid version of labels (labels + additional predictions)
conf	            0.001	        object confidence threshold for detection
iou	                0.6	            intersection over union (IoU) threshold for NMS
max_det	            300	            maximum number of detections per image
half	            True	        use half precision (FP16)
device	            null	        device to run on, i.e. cuda device=0/1/2/3 or device=cpu
dnn	                False	        use OpenCV DNN for ONNX inference
plots	            False	        show plots during training
rect	            False	        support rectangular evaluation
split	            val	            dataset split to use for validation, i.e. 'val', 'test' or 'train'
"""


def val(weights: str, device="0", name="", dataset="", project="") -> None:
    model = YOLO(weights)
    if dataset != "":
        model.overrides["data"] = dataset
    name = "val-" + name
    # print(model)
    model.val(
        imgsz=640,
        device=device,
        conf=0.001,
        iou=0.5,
        save_json=True,
        save_hybrid=False,
        project=project,
        name=name,
        split="test",
    )
    # print(metrics)


"""
print model
"""


def printModel(path_yaml: str, model_type: str = "v8"):
    YOLO(path_yaml)
    # print(model.model)


"""
format	    'torchscript'	    format to export to
imgsz	    640	                image size as scalar or (h, w) list, i.e. (640, 480)
keras	    False	            use Keras for TF SavedModel export
optimize	False	            TorchScript: optimize for mobile
half	    False	            FP16 quantization
int8	    False	            INT8 quantization
dynamic	    False	            ONNX/TF/TensorRT: dynamic axes
simplify	False	            ONNX: simplify model
opset	    None	            ONNX: opset version (optional, defaults to latest)
workspace	4	                TensorRT: workspace size (GB)
nms	        False	            CoreML: add NMS
"""


def export(model_path: str, format: str = "onnx"):
    model = YOLO(model_path)
    model.export(format=format, half=False, opset=11, device="0", imgsz=(1088, 1920))


def load_model(model_path: str):
    model = YOLO(model_path)
    return model


import argparse

# python tools.py --mode val \
#     --init_weight runs/ATF/AITOD/yolov11-rtdetr-aitod-1500q/weights/best.pt \
#     --data_path ../custom_data/AITOD.yaml --device 0 --name yolov11s-rtdetr-aitod-1500q


def main():
    parser = argparse.ArgumentParser(description="Small Object Detection Training and Evaluation")
    parser.add_argument("--device", type=str, default="0", help="Device to use, e.g., '0', 'cpu'")
    parser.add_argument(
        "--data_path", type=str, default="../custom_data/3rdUAV/dataset.yaml", help="Path to the dataset YAML file"
    )
    parser.add_argument(
        "--path_yaml", type=str, default="rtdetr-r18.yaml", help="Path to the model YAML configuration file"
    )
    parser.add_argument(
        "--init_weight",
        type=str,
        default="3rdUAV-modify/3rdUAV_rtdetr-r18-ImprovedMFConv-GAP3/weights/best.pt",
        help="Path to the initial weights file",
    )
    parser.add_argument("--project", type=str, default="runs", help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="experiment", help="Experiment name")
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["train", "train_resume", "val", "predict", "export", "track"],
        help="Mode of operation",
    )
    parser.add_argument(
        "--testvideo_path",
        type=str,
        default="cases/success_case/testimages",
        help="Path to the test video or images (for predict/track)",
    )
    parser.add_argument("--pred_json", type=str, default="", help="Path to predictions JSON file (for eval)")
    parser.add_argument("--anno_json", type=str, default="", help="Path to annotations JSON file (for eval)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")
    parser.add_argument("--save_period", type=int, default=-1, help="Save checkpoint every N epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            args.path_yaml,
            args.data_path,
            args.init_weight,
            args.device,
            project=args.project,
            name=args.name,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            patience=args.patience,
            save_period=args.save_period,
            seed=args.seed,
        )
    elif args.mode == "train_resume":
        train_resume(args.init_weight, args.data_path, args.device)
    elif args.mode == "val":
        val(args.init_weight, device=args.device, name=args.name, dataset=args.data_path, project=args.project)
    elif args.mode == "predict":
        if not args.testvideo_path:
            raise ValueError("testvideo_path is required for predict mode")
        predict(args.init_weight, args.testvideo_path, device=args.device, name=args.name)
    elif args.mode == "export":
        export(args.init_weight)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    # device = "0"
    # data_path = "../custom_data/VisDrone.yaml"

    # # 模型和初始权�?
    # # path_yaml = "RT-DERT-cfg/yolov8.yaml"
    # path_yaml = "/home/benol/Programming/small-object-detection-2024/RTDETR/RT-DERT-cfg/ATF/yolov11l-ATF-MP2-AP2.yaml"

    # init_weight = ""
    # # init_weight = "yolo11l.pt"
    # # init_weight = "runs/moe-yolov5-rtdetr-balance/weights/best.pt"
    # # exp name
    # name = "yolo11l-ATF-MP2-AP2-1280"

    # # load_model(path_yaml)
    # train(path_yaml, data_path, init_weight, device, project = "runs", name=name)
    # # train_resume(init_weight, data_path, device)
    # # val(init_weight, device=device, name=name, dataset=data_path)
    # # export("runs/detect/yolov8n/yolov8n-visdrone.pt")

    # # test video
    # # predict(init_weight, testvideo_path, device="0", name=name)

    # # eval json
    # # pred_json = "vals/val-yolov5s-rtdetr-P28/predictions.json"
    # # anno_josn = "/media/benol/131710FB9319A910/DataSet/VisDrone/annotations/instances_test.json"
    # # eval_cocoMetric(anno_josn, pred_json)

    # # test video
    # # testvideo_path = "/mnt/d/DataSet/miniVisDrone/val/images"
    # # predict(init_weight, testvideo_path, device="0", name=name)
    # # track(init_weight, testvideo_path, device="0", name=name)


if __name__ == "__main__":
    main()
