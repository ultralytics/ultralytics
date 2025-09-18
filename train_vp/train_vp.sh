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






export PYTHONPATH="/root/ultra_louis_work/ultralytics:$PYTHONPATH"

if [ ! -f "yoloe-v8s-seg.pt" ]; then
    echo "File yoloe-v8s-seg.pt does not exist. Downloading..."
    wget https://huggingface.co/jameslahm/yoloe/resolve/main/yoloe-v8s-seg.pt
else
    echo "File yoloe-v8s-seg.pt already exists. Skipping download."
fi




DATA_DIR="../datasets"
log_step "Step 1: Generating LVIS visual prompt dataset..."

if [ ! -d "$DATA_DIR/lvis_train_vps" ]; then
    log_info "Generating LVIS visual prompt data..."
    python train_vp/generate_lvis_visual_prompt_data.py
    log_success "LVIS visual prompt dataset generated"
else
    log_info "LVIS visual prompt dataset already exists. Skipping generation."
fi




python train_vp/train_vp.py

# python val_vp.py
