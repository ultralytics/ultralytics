import argparse
import csv
import random
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASETS = ("NUAA-SIRST", "NUDT-SIRST")
SPLIT_DIR = ROOT / "infrared_dataset"
DEFAULT_PROJECT = ROOT / "runs" / "infrared"


def parse_args() -> argparse.Namespace:
    # 这里定义命令行参数。你在 PowerShell 里写的 --epochs、--imgsz、--batch 等，
    # 都会先进入这个函数，再传给后面的 Ultralytics model.train()。
    parser = argparse.ArgumentParser(description="Train a YOLO baseline on the infrared SIRST datasets.")
    parser.add_argument("--model", default=str(ROOT / "yolov8n.pt"), help="Model yaml or weights path.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=320, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers. Use 0 on Windows for stability.")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or 0.")
    parser.add_argument("--seed", type=int, default=42, help="Train/val split seed.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio per dataset.")
    parser.add_argument("--name", default="baseline_yolov8n", help="Ultralytics run name.")
    parser.add_argument(
        "--ra-fa-conf",
        type=float,
        default=0.001,
        help="Confidence threshold for Ra/Fa. Ultralytics validation uses low confidence for PR/mAP curves.",
    )
    parser.add_argument("--ra-fa-iou-index", type=int, default=0, help="IoU index in Ultralytics stats, 0 means IoU=0.50.")
    return parser.parse_args()


def ensure_labels(dataset_dir: Path) -> None:
    # YOLO 检测训练要求 images 和 labels 一一对应：
    # images/xxx.png 必须能找到 labels/xxx.txt。
    # 这里先做数量检查，避免训练跑到一半才发现标签缺失。
    image_count = len(list((dataset_dir / "images").glob("*.png")))
    label_count = len(list((dataset_dir / "labels").glob("*.txt")))
    if image_count == 0:
        raise FileNotFoundError(f"No images found in {dataset_dir / 'images'}")
    if label_count != image_count:
        raise FileNotFoundError(
            f"{dataset_dir.name} labels are missing or incomplete: images={image_count}, labels={label_count}. "
            "Run mask2yolo.py first."
        )


def build_split(seed: int, val_ratio: float) -> tuple[Path, Path, Path, int]:
    # 这个函数负责把 NUAA-SIRST 和 NUDT-SIRST 合并成一个训练任务。
    # 它不会复制图片，只会生成 train.txt / val.txt。
    # 每一行是图片绝对路径，Ultralytics 会根据图片路径自动把 images 替换成 labels 来找标注。
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")

    # 固定随机种子，让每次切分出的训练集/验证集一致，便于复现实验。
    rng = random.Random(seed)
    train_images = []
    val_images = []

    for dataset in DATASETS:
        dataset_dir = ROOT / dataset
        ensure_labels(dataset_dir)
        images = sorted((dataset_dir / "images").glob("*.png"))
        rng.shuffle(images)
        # 每个数据集内部按 val_ratio 划分验证集。
        # max(1, ...) 保证即使数据集很小，也至少留 1 张图做验证。
        val_count = max(1, round(len(images) * val_ratio))
        val_images.extend(images[:val_count])
        train_images.extend(images[val_count:])

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_txt = SPLIT_DIR / "train.txt"
    val_txt = SPLIT_DIR / "val.txt"
    data_yaml = SPLIT_DIR / "infrared.yaml"

    train_txt.write_text("\n".join(str(p) for p in train_images) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(str(p) for p in val_images) + "\n", encoding="utf-8")
    # infrared.yaml 是 Ultralytics 的数据集配置文件。
    # nc=1 表示只有一个类别，names=["target"] 是这个类别的名字。
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(ROOT),
                "train": str(train_txt),
                "val": str(val_txt),
                "nc": 1,
                "names": ["target"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    return train_txt, val_txt, data_yaml, len(val_images)


def concat_stats(stats: dict) -> dict[str, np.ndarray]:
    # Ultralytics 验证时会按 batch 保存统计量，例如每个 batch 的 TP、conf、pred_cls。
    # 计算 Ra/Fa 时需要先把多个 batch 的数组拼成一个完整验证集数组。
    output = {}
    for key, values in stats.items():
        output[key] = np.concatenate(values, axis=0) if values else np.array([])
    return output


def compute_ra_fa(
    raw_stats: dict[str, list[np.ndarray]], val_image_count: int, conf_threshold: float, iou_index: int
) -> tuple[float, float, int, int]:
    # Ra/Fa 是红外小目标任务里常用的评价指标。
    # 这里的定义：
    #   Ra = TP / GT
    #   Fa = FP / 验证集图片数
    # conf_threshold 控制保留哪些预测框；阈值低时召回高，但误警也会高。
    # iou_index=0 对应 Ultralytics 的 IoU=0.50，1 对应 0.55，依次到 0.95。
    stats = concat_stats(raw_stats)
    target_count = int(len(stats.get("target_cls", [])))
    conf = stats.get("conf", np.array([]))
    tp = stats.get("tp", np.zeros((0, 1), dtype=bool))

    if conf.size == 0:
        return 0.0, 0.0, 0, target_count

    iou_index = min(max(iou_index, 0), tp.shape[1] - 1)
    # keep 表示哪些预测框的置信度达到了 Ra/Fa 统计阈值。
    keep = conf >= conf_threshold
    true_positives = int(tp[keep, iou_index].sum())
    false_positives = int(keep.sum() - true_positives)

    ra = true_positives / target_count if target_count else 0.0
    fa = false_positives / val_image_count if val_image_count else 0.0
    return ra, fa, true_positives, target_count


def format_float(value) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    args = parse_args()
    # 先生成 Ultralytics 可读的数据集配置，再开始训练。
    _, _, data_yaml, val_image_count = build_split(args.seed, args.val_ratio)

    # YOLO 是 Ultralytics 对模型的统一封装。
    # 这里传入 yolov8n.pt，表示加载预训练 YOLOv8n 权重做迁移训练。
    model = YOLO(args.model)
    history = []

    # validator.metrics.stats 在 Ultralytics 内部计算 mAP 后会被清空。
    # 所以我们在每个验证 batch 结束时，把 TP/FP 所需的原始统计缓存到 val_stats。
    val_stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
    val_state = {"last_count": 0}

    def reset_val_stats(validator) -> None:
        # 每个 epoch 的验证开始时清空上一轮缓存。
        # validator.training=True 表示这是训练过程中的验证，不是单独调用 model.val()。
        if not validator.training:
            return
        for values in val_stats.values():
            values.clear()
        val_state["last_count"] = 0

    def collect_val_stats(validator) -> None:
        # 每个验证 batch 结束后，把新增的统计量复制出来。
        # 这些统计量包括：
        #   tp: 每个预测框在不同 IoU 阈值下是否匹配成功
        #   conf: 每个预测框的置信度
        #   pred_cls: 预测类别
        #   target_cls: 真实类别
        #   target_img: 每张图是否包含该类别
        if not validator.training:
            return
        stats = validator.metrics.stats
        current_count = len(stats["tp"])
        start = val_state["last_count"]
        for key, values in stats.items():
            val_stats[key].extend(np.array(v).copy() for v in values[start:current_count])
        val_state["last_count"] = current_count

    def log_epoch(trainer) -> None:
        # on_fit_epoch_end 回调会在每个 epoch 的训练和验证都结束后触发。
        # 这时 trainer.metrics 里已经有 mAP、precision、recall、val loss 等指标。
        epoch = int(trainer.epoch) + 1
        if epoch > int(args.epochs):
            return

        metrics = trainer.metrics or {}
        # trainer.tloss 是当前 epoch 的训练平均 loss。
        # label_loss_items 会把它转换成 train/box_loss、train/cls_loss、train/dfl_loss。
        losses = trainer.label_loss_items(trainer.tloss) if trainer.tloss is not None else {}
        ra, fa, tp, gt = compute_ra_fa(val_stats, val_image_count, args.ra_fa_conf, args.ra_fa_iou_index)

        row = {
            "epoch": epoch,
            "epochs_arg": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "ra_fa_conf": args.ra_fa_conf,
            "ra_fa_iou": 0.50 + 0.05 * args.ra_fa_iou_index,
            "Ra": ra,
            "Fa": fa,
            "TP": tp,
            "GT": gt,
            "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
            "box_loss": losses.get("train/box_loss", 0.0),
            "cls_loss": losses.get("train/cls_loss", 0.0),
            "dfl_loss": losses.get("train/dfl_loss", 0.0),
            "val_box_loss": metrics.get("val/box_loss", 0.0),
            "val_cls_loss": metrics.get("val/cls_loss", 0.0),
            "val_dfl_loss": metrics.get("val/dfl_loss", 0.0),
        }
        history.append(row)

        # 控制台输出一行高信号摘要，方便训练时直接观察每个 epoch 的变化。
        print(
            "[infrared] "
            f"epoch={epoch} "
            f"Ra={format_float(row['Ra'])} "
            f"Fa={format_float(row['Fa'])} "
            f"TP={row['TP']} "
            f"GT={row['GT']} "
            f"mAP50={format_float(row['mAP50'])} "
            f"mAP50-95={format_float(row['mAP50-95'])} "
            f"box_loss={format_float(row['box_loss'])} "
            f"cls_loss={format_float(row['cls_loss'])} "
            f"dfl_loss={format_float(row['dfl_loss'])}"
        )

    # 注册回调：不改 Ultralytics 源码，也能在训练/验证流程中插入自定义逻辑。
    model.add_callback("on_val_start", reset_val_stats)
    model.add_callback("on_val_batch_end", collect_val_stats)
    model.add_callback("on_fit_epoch_end", log_epoch)

    # 真正开始训练。这里的参数会进入 ultralytics/engine/model.py 的 train()，
    # 再创建 DetectionTrainer，最后进入 BaseTrainer._do_train() 训练主循环。
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(DEFAULT_PROJECT),
        name=args.name,
        exist_ok=True,
        val=True,
        plots=True,
        save=True,
        single_cls=True,
    )

    run_dir = Path(model.trainer.save_dir)
    metrics_csv = run_dir / "epoch_metrics_ra_fa.csv"
    if history:
        # 把每个 epoch 的 Ra/Fa、mAP 和 loss 保存成 CSV，便于画图或写论文表格。
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print(f"[infrared] run_dir={run_dir}")
    print(f"[infrared] epoch_metrics={metrics_csv}")
    print(f"[infrared] best_model={run_dir / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    main()
