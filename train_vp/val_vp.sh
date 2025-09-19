

# Initialize conda for bash shell
source ~/miniconda3/etc/profile.d/conda.sh
# Activate sam2 environment
conda activate ultralytics



for vp_weight in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
    echo "Running validation with vp_weight: $vp_weight"
    python train_vp/val_vp.py --vp_weight $vp_weight --model_weight /home/louis/ultra_louis_work/ultralytics/runs/detect/train2/weights/best.pt
done