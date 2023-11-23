# NAIC20 Competition (ReID Track) 

This repository contains the 1-st place solution of ReID Competition of NAIC. We got the first place in the final stage. 

## Introduction

Detailed information about the NAIC competition can be found [here](https://naic.pcl.ac.cn/homepage/index.html).

## Useful Tricks

- [x] DataAugmentation (RandomErasing + ColorJitter + Augmix + RandomAffine + RandomHorizontallyFilp + Padding + RandomCrop)
- [x] LR Scheduler (Warmup + CosineAnnealing)
- [x] Optimizer (Adam)
- [x] FP16 mixed precision training
- [x] CircleSoftmax
- [x] Pairwise Cosface
- [x] GeM pooling
- [x] Remove Long Tail Data (pid with single image)
- [x] Channel Shuffle
- [x] Distmat Ensemble

1. Due to the competition's rule, pseudo label is not allowed in the preliminary and semi-finals, but can be used in finals.
2. We combine naic19, naic20r1 and naic20r2 datasets, but there are overlap and noise between these datasets. So we
use an automatic data clean strategy for data clean. The cleaned txt files are put here. Sorry that this part cannot ben open sourced.
3. Due to the characteristics of the encrypted dataset, we found **channel shuffle** very helpful.
It's an offline data augmentation method. Specifically, for each id, random choice an order of channel, 
such as `(2, 1, 0)`, then apply this order for all images of this id, and make it a new id.
With this method, you can enlarge the scale of identities. Theoretically, each id can be enlarged to 5 times.
Considering computational efficiency and marginal effect, we just enlarge each id once.
But this trick is no effect in normal dataset.
4. Due to the distribution of dataset, we found pairwise cosface can greatly boost model performance.
5. The performance of `resnest` is far better than `ibn`. 
We choose `resnest101`, `resnest200` with different resolution (192x256, 192x384) to ensemble. 

## Training & Submission in Command Line

Before starting, please see [GETTING_STARTED.md](https://github.com/JDAI-CV/fast-reid/blob/master/GETTING_STARTED.md) for the basic setup of FastReID.
All configs are made for 2-GPU training.

1. To train a model, first set up the corresponding datasets following [datasets/README.md](https://github.com/JDAI-CV/fast-reid/tree/master/datasets), then run:

```bash
python3 projects/NAIC20/train_net.py --config-file projects/NAIC20/configs/r34-ibn.yml --num-gpus 2 
```

2. After the model is trained, you can start to generate submission file. First, modify the content of `MODEL` in `submit.yml` to 
adapt your trained model, and set `MODEL.WEIGHTS` to the path of your trained model, then run:

```bash
python3 projects/NAIC20/train_net.py --config-file projects/NAIC20/configs/submit.yml --eval-only --commit --num-gpus 2
```

You can find `submit.json` and `distmat.npy` in `OUTPUT_DIR` of `submit.yml`.

## Ablation Study

To quickly verify the results, we use resnet34-ibn as backbone to conduct ablation study.
The datasets are `naic19`, `naic20r1` and `naic20r2`.

| Setting | Rank-1 | mAP |
| ------  | ------ | --- |
| Baseline | 70.11 | 63.29 |
| w/ tripletx10 | 73.79 | 67.01 |
| w/ cosface | 75.61 | 70.07 |
