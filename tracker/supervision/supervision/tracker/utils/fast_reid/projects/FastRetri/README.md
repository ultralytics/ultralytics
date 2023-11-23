# FastRetri in FastReID

This project provides a strong baseline for fine-grained image retrieval.

## Datasets Preparation

We use `CUB200`, `CARS-196`, `Standford Online Products` and `In-Shop` to evaluate the model's performance.
You can do data management following [dml_cross_entropy](https://github.com/jeromerony/dml_cross_entropy) instruction.

## Usage

Each dataset's config file can be found in `projects/FastRetri/config`, which you can use to reproduce the results of the repo.

For example, if you want to train with `CUB200`, you can run an experiment with `cub.yml`

```bash
python3 projects/FastRetri/train_net.py --config-file projects/FastRetri/config/cub.yml --num-gpus 4
```

## Experiment Results

We refer to [A unifying mutual information view of metric learning: cross-entropy vs. pairwise losses](arxiv.org/abs/2003.08983) as our baseline methods, and on top of it, we add some tricks, such as gem pooling.
More details can be found in the config file and code.

### CUB

| Method | Pretrained | Recall@1 | Recall@2 | Recall@4 | Recall@8 | Recall@16 | Recall@32 |
| :---: | :---: | :---: |:---: | :---: | :---: | :---: | :---: |
| [dml_cross_entropy](https://github.com/jeromerony/dml_cross_entropy) | ImageNet | 69.2 | 79.2 | 86.9 | 91.6 | 95.0 | 97.3 |
| Fastretri | ImageNet | 69.46 | 79.57 | 87.53 | 92.61 | 95.75 | 97.35 |

### Cars-196

| Method | Pretrained | Recall@1 | Recall@2 | Recall@4 | Recall@8 | Recall@16 | Recall@32 |
| :---: | :---: | :---: |:---: | :---: | :---: | :---: | :---: |
| [dml_cross_entropy](https://github.com/jeromerony/dml_cross_entropy) | ImageNet | 89.3 | 93.9 | 96.6 | 98.4 | 99.3 | 99.7 |
| Fastretri | ImageNet | 92.31 | 95.99 | 97.60 | 98.63 | 99.24 | 99.62 |

### Standford Online Products

| Method | Pretrained | Recall@1 | Recall@10 | Recall@100 | Recall@1000 |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [dml_cross_entropy](https://github.com/jeromerony/dml_cross_entropy) | ImageNet | 81.1 | 91.7 | 96.3 | 98.8 |
| Fastretri | ImageNet | 82.46 | 92.56 | 96.78 | 98.95 |

### In-Shop

| Method | Pretrained | Recall@1 | Recall@10 | Recall@20 | Recall@30 | Recall@40 | Recall@50 |
| :---: | :---: | :---: |:---: | :---: | :---: | :---: | :---: |
| [dml_cross_entropy](https://github.comjeromerony/dml_cross_entropy) | ImageNet | 90.6 | 98.0 | 98.6 | 98.9 | 99.1 | 99.2 |
| Fastretri | ImageNet | 91.97 | 98.29 | 98.85 | 99.11 | 99.24 | 99.35 |

