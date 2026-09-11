#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ==================== 默认配置（按需修改，命令行同名参数会覆盖这里的值） ====================
# None 表示沿用框架的可选默认（不强制设置），如 TIME、FREEZE、CLASSES、DISTILL_MODEL。
# 各项取值与 ultralytics/cfg/default.yaml 保持一致；修改后会作为命令行覆盖项传入。

# 基础
TASK="detect"
MODEL="" # 预训练权重或架构：.pt 检查点（如自己训出的 best.pt）或 .yaml；留空 "" 使用 TASK 对应的内置预训练模型
DATA="coco8.yaml" # 需与 TASK 匹配，可改为自己的数据集 YAML 绝对路径
EPOCHS=100
BATCH=8
DEVICE="cpu"
PROJECT="fire-smoke" # 输出项目目录：留空 "" 放在框架默认 runs/<任务>/ 下；相对名追加在任务目录后；绝对路径直接作为项目目录
NAME="test" # 输出子目录名：留空 "" 使用默认 train；与已有目录重名时自动递增（如 train-2）

# 优化器与学习率
OPTIMIZER="auto"
LR0=0.01
LRF=0.01
MOMENTUM=0.937
WEIGHT_DECAY=0.0005
COS_LR=False
NBS=64

# 预热
WARMUP_EPOCHS=3.0
WARMUP_MOMENTUM=0.8
WARMUP_BIAS_LR=0.1

# 训练控制
PATIENCE=100
TIME=None
SEED=0
DETERMINISTIC=True
AMP=True
CACHE=False
FREEZE=None
RESUME=False
PRETRAINED=True

# 数据与性能
FRACTION=1.0
RECT=False
SINGLE_CLS=False
CLASSES=None
MULTI_SCALE=0.0
COMPILE=False
CHANNELS_LAST=None
CLS_REMAP=True

# 色彩与几何增强
HSV_H=0.015
HSV_S=0.7
HSV_V=0.4
DEGREES=0.0
TRANSLATE=0.1
SCALE=0.5
SHEAR=0.0
PERSPECTIVE=0.0
FLIPUD=0.0
FLIPLR=0.5
BGR=0.0

# 样本混合增强
MOSAIC=1.0
MIXUP=0.0
CUTMIX=0.0
COPY_PASTE=0.0
COPY_PASTE_MODE="flip"
CLOSE_MOSAIC=10

# 分割与分类
AUTO_AUGMENT="randaugment"
ERASING=0.4
DROPOUT=0.0
OVERLAP_MASK=True
MASK_RATIO=4

# 损失权重与蒸馏
BOX=7.5
CLS=0.5
CLS_PW=0.0
DFL=1.5
POSE=12.0
KOBJ=1.0
RLE=1.0
ANGLE=1.0
DLOG=1.0
DGRAD=0.5
DLAM=1.0
DISTILL_MODEL=None
DIS=6.0

# 保存与日志
VERBOSE=True
SAVE=True
SAVE_PERIOD=-1
PLOTS=True
VAL=True
EXIST_OK=False
# ==========================================================================================

exec python "$SCRIPT_DIR/train.py" \
    --task "$TASK" \
    --model "$MODEL" \
    --data "$DATA" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --project "$PROJECT" \
    --name "$NAME" \
    --optimizer "$OPTIMIZER" \
    --lr0 "$LR0" \
    --lrf "$LRF" \
    --momentum "$MOMENTUM" \
    --weight_decay "$WEIGHT_DECAY" \
    --cos_lr "$COS_LR" \
    --nbs "$NBS" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --warmup_momentum "$WARMUP_MOMENTUM" \
    --warmup_bias_lr "$WARMUP_BIAS_LR" \
    --patience "$PATIENCE" \
    --time "$TIME" \
    --seed "$SEED" \
    --deterministic "$DETERMINISTIC" \
    --amp "$AMP" \
    --cache "$CACHE" \
    --freeze "$FREEZE" \
    --resume "$RESUME" \
    --pretrained "$PRETRAINED" \
    --fraction "$FRACTION" \
    --rect "$RECT" \
    --single_cls "$SINGLE_CLS" \
    --classes "$CLASSES" \
    --multi_scale "$MULTI_SCALE" \
    --compile "$COMPILE" \
    --channels_last "$CHANNELS_LAST" \
    --cls_remap "$CLS_REMAP" \
    --hsv_h "$HSV_H" \
    --hsv_s "$HSV_S" \
    --hsv_v "$HSV_V" \
    --degrees "$DEGREES" \
    --translate "$TRANSLATE" \
    --scale "$SCALE" \
    --shear "$SHEAR" \
    --perspective "$PERSPECTIVE" \
    --flipud "$FLIPUD" \
    --fliplr "$FLIPLR" \
    --bgr "$BGR" \
    --mosaic "$MOSAIC" \
    --mixup "$MIXUP" \
    --cutmix "$CUTMIX" \
    --copy_paste "$COPY_PASTE" \
    --copy_paste_mode "$COPY_PASTE_MODE" \
    --close_mosaic "$CLOSE_MOSAIC" \
    --auto_augment "$AUTO_AUGMENT" \
    --erasing "$ERASING" \
    --dropout "$DROPOUT" \
    --overlap_mask "$OVERLAP_MASK" \
    --mask_ratio "$MASK_RATIO" \
    --box "$BOX" \
    --cls "$CLS" \
    --cls_pw "$CLS_PW" \
    --dfl "$DFL" \
    --pose "$POSE" \
    --kobj "$KOBJ" \
    --rle "$RLE" \
    --angle "$ANGLE" \
    --dlog "$DLOG" \
    --dgrad "$DGRAD" \
    --dlam "$DLAM" \
    --distill_model "$DISTILL_MODEL" \
    --dis "$DIS" \
    --verbose "$VERBOSE" \
    --save "$SAVE" \
    --save_period "$SAVE_PERIOD" \
    --plots "$PLOTS" \
    --val "$VAL" \
    --exist_ok "$EXIST_OK" \
    "$@"
