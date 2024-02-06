# Must do to start 
module load Python/3.10.4-GCCcore-11.3.0

pip install ultralytics 
pip install -e '.[dev]'
pip install einops

# Connect with slurm job (new GPU)
ssh -L 8888:127.0.0.1:8888 -J vemundtl@idun-login1.hpc.ntnu.no vemundtl@idun-xx-xx

# Kill process
scancel -u vemundtl