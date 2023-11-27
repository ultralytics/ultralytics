# FastDistill in FastReID

This project provides a strong distillation method for both embedding and classification training.
The feature distillation comes from [overhaul-distillation](https://github.com/clovaai/overhaul-distillation/tree/master/ImageNet).


## Datasets Prepration
- DukeMTMC-reID


## Train and Evaluation
```shell
# teacher model training
python3 projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/sbs_r101ibn.yml \
--num-gpus 4

# loss distillation
python3 projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r101ibn-sbs_r34.yaml \
--num-gpus 4 \
MODEL.META_ARCHITECTURE Distiller
KD.MODEL_CONFIG '("projects/FastDistill/logs/dukemtmc/r101_ibn/config.yaml",)' \
KD.MODEL_WEIGHTS '("projects/FastDistill/logs/dukemtmc/r101_ibn/model_best.pth",)'

# loss+overhaul distillation
python3 projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r101ibn-sbs_r34.yaml \
--num-gpus 4 \
MODEL.META_ARCHITECTURE DistillerOverhaul
KD.MODEL_CONFIG '("projects/FastDistill/logs/dukemtmc/r101_ibn/config.yaml",)' \
KD.MODEL_WEIGHTS '("projects/FastDistill/logs/dukemtmc/r101_ibn/model_best.pth",)'
```

## Experimental Results

### Settings

All the experiments are conducted with 4 V100 GPUs.


### DukeMTMC-reID

| Model | Rank@1 | mAP |
| --- | --- | --- |
| R101_ibn (teacher) | 90.66 | 81.14 |
| R34 (student) | 86.31 | 73.28 |
| JS Div | 88.60 | 77.80 |
| JS Div + Overhaul | 88.73 | 78.25 |

## Contact
This project is conducted by [Xingyu Liao](https://github.com/L1aoXingyu) and [Guan'an Wang](https://wangguanan.github.io/) (guan.wang0706@gmail).
