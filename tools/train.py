# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train any supported YOLO task with command-line arguments."""

import argparse

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, TASK2DATA, TASK2MODEL, TASKS, check_cfg, smart_value

ARGS = {"epochs": 100, "imgsz": 640, "workers": 0, "device": None, "project": None, "name": "train"}
HYPERPARAMETERS = {
    "Optimizer and learning rate": (
        "optimizer lr0 lrf momentum weight_decay warmup_epochs warmup_momentum warmup_bias_lr cos_lr nbs"
    ),
    "Training controls": "patience time seed deterministic amp cache freeze resume pretrained",
    "Data and performance": "fraction rect single_cls classes multi_scale compile channels_last cls_remap",
    "Augmentation": (
        "hsv_h hsv_s hsv_v degrees translate scale shear perspective flipud fliplr bgr mosaic mixup cutmix "
        "copy_paste copy_paste_mode close_mosaic"
    ),
    "Segmentation and classification": "auto_augment erasing dropout overlap_mask mask_ratio",
    "Loss weights and distillation": "box cls cls_pw dfl pose kobj rle angle dlog dgrad dlam distill_model dis",
    "Saving and logging": "verbose save save_period plots val exist_ok",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="detect", help="Default model task (default: detect)")
    parser.add_argument("--model", help="Model checkpoint (.pt) or architecture (.yaml); defaults to the task model")
    parser.add_argument("--data", help="Dataset YAML or classification directory (default: model task demo data)")
    parser.add_argument(
        "--batch", type=smart_value, default=8, help="Batch size, -1, or AutoBatch fraction (default: 8)"
    )
    for key, default in ARGS.items():
        parser.add_argument(
            f"--{key}", type=type(default) if default is not None else str, default=default, help="Default: %(default)s"
        )
    for title, keys in HYPERPARAMETERS.items():
        group = parser.add_argument_group(title)
        for key in keys.split():
            group.add_argument(
                f"--{key}",
                type=smart_value,
                default=argparse.SUPPRESS,
                help=f"Framework default: {DEFAULT_CFG_DICT[key]!r}; accepts Python literals, True/False, or strings",
            )
    args = vars(parser.parse_args())
    check_cfg(args)
    task = args.pop("task")
    model = YOLO(args.pop("model") or TASK2MODEL[task])
    model.train(data=args.pop("data") or TASK2DATA[model.task], **args)
    print(f"Training output: {model.trainer.save_dir}")
    print(f"Best weights: {model.trainer.best}")
