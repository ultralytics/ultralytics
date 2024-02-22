# Must do to start 
module load Python/3.10.4-GCCcore-11.3.0__
module load Python/3.10.8-GCCcore-12.2.0__

pip install ultralytics__
pip install -e '.[dev]'__
pip install einops

# Connect with slurm job (new GPU)
ssh -L 8888:127.0.0.1:8888 -J <username>@idun-login1.hpc.ntnu.no <username>@idun-xx-xx

# Kill process
sbatch idun_config.slurm__
squeue -u <username>__
scancel -u <username>