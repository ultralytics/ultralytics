#!/bin/bash
set -e  # Exit on any error

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${GREEN}🔧${NC} $1"
}
# Initialize conda for bash shell
source ~/miniconda3/etc/profile.d/conda.sh
# Activate sam2 environment
conda activate ultralytics






export PYTHONPATH="/root/ultra_louis_work/yoloe:$PYTHONPATH"


cd  /root/ultra_louis_work/yoloe

if [ ! -f "yoloe-11s-seg.pt" ]; then
    echo "File yoloe-11s-seg.pt does not exist. Downloading..."
    wget https://huggingface.co/jameslahm/yoloe/resolve/main/yoloe-8s-seg.pt
else
    echo "File yoloe-11s-seg.pt already exists. Skipping download."
fi



# python tools/convert_segm2det.py
# wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
export PYTHONPATH="/root/ultra_louis_work/yoloe/third_party/ml-mobileclip:$PYTHONPATH"

# python tools/generate_label_embedding.py
# python tools/generate_global_neg_cat.py



# python train_vp.py


val_pro=/root/ultra_louis_work/runs/detect/val5
DATA_DIR="/root/autodl-tmp/datasets/"

python tools/eval_fixed_ap.py $DATA_DIR/lvis/annotations/lvis_v1_minival.json $val_pro/predictions.json
# python val_vp.py
