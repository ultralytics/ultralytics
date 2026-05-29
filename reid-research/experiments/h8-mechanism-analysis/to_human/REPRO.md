# h8 — Reproducibility Manifest

## Code
- Repo: ultralytics, branch `reid-task-official-clean`
- Commit SHA: (fill in `git rev-parse HEAD` after final commit)

## Environment (westd analysis box)
- torch: (fill in)
- CUDA: (fill in)
- cudnn: (fill in)
- Ultralytics: (fill in `python -c 'import ultralytics; print(ultralytics.__version__)'`)

## Environment (seetacloud Stage-6 training)
- torch: (fill in)
- CUDA: (fill in)
- 4× (fill in GPU model)

## Datasets
- Market-1501-v15.09.15 (sha256 of `.zip`: fill in `sha256sum`)
- MSMT17 (sha256: fill in)

## Model checkpoints (SHA256)
- champion: (fill in)
- mgn-t3: (fill in)
- mgn-t4: (fill in)
- t5fix: (fill in)
- solider (Swin-Base teacher): (fill in)

## Seeds
- Stage 6 seeds: (fill in from EXPERIMENT.md)

## Per-stage runtime
- Stage 1 (extract.py): (fill in)
- Stage 2 (s2_failure_taxonomy.py): (fill in)
- Stage 3 (s3_solider_gap.py): (fill in)
- Stage 4 (s4_training_dynamics.py): (fill in)
- Stage 6 (s6_validate.py, 3 seeds parallel on 3 GPUs): (fill in)
