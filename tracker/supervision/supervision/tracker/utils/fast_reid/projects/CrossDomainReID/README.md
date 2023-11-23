# Cross-domain Person Re-Identification

## Introduction

[UDAStrongBaseline](https://github.com/zkcys001/UDAStrongBaseline) is a transitional code based pyTorch framework for both unsupervised learning (USL) 
and unsupervised domain adaptation (UDA) in the object re-ID tasks. It provides stronger 
baselines on these tasks. It needs the enviorment: Python >=3.6 and PyTorch >=1.1. We will transfer all the codes to the [fastreid](https://github.com/JDAI-CV/fast-reid) in the future (ongoing) from [UDAStrongBaseline](https://github.com/zkcys001/UDAStrongBaseline).


### Unsupervised domain adaptation (UDA) on Person re-ID

- `Direct Transfer` models are trained on the source-domain datasets 
([source_pretrain]()) and directly tested on the target-domain datasets.
- UDA methods (`MMT`, `SpCL`, etc.) starting from ImageNet means that they are trained end-to-end 
in only one stage without source-domain pre-training. `MLT` denotes to the implementation of our NeurIPS-2020. 
Please note that it is a pre-released repository for the anonymous review process, and the official 
repository will be released upon the paper published.

#### DukeMTMC-reID -> Market-1501

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | 
| Direct Transfer | ResNet50 | DukeMTMC | 32.2 | 64.9 | 78.7 | 83.4 | ~1h | 
| [UDA_TP](https://github.com/open-mmlab/OpenUnReID/) PR'2020| ResNet50 | DukeMTMC | 52.3 | 76.0 | 87.8 | 91.9 | ~2h | 
| [MMT](https://github.com/open-mmlab/OpenUnReID/) ICLR'2020| ResNet50 | DukeMTMC | 80.9 | 92.2 | 97.6 | 98.4 | ~6h |
| [SpCL](https://github.com/open-mmlab/OpenUnReID/) NIPS'2020 submission| ResNet50 | DukeMTMC | 78.2 | 90.5 | 96.6 | 97.8 | ~3h |
| [strong_baseline](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | DukeMTMC | 75.6 | 90.9 | 96.6 | 97.8 | ~3h | 
| [Our stronger_baseline](https://github.com/JDAI-CV/fast-reid) | ResNet50 | DukeMTMC | 78.0 | 91.0 | 96.4 | 97.7 | ~3h |
| [MLT] NeurIPS'2020 submission| ResNet50 | DukeMTMC | 81.5| 92.8| 96.8| 97.9 | ~ |

#### Market-1501 -> DukeMTMC-reID

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | 
| Direct Transfer | ResNet50 | Market | 34.1 | 51.3 | 65.3 | 71.7 | ~1h | 
| [UDA_TP](https://github.com/open-mmlab/OpenUnReID/) PR'2020| ResNet50 | Market | 45.7 | 65.5 | 78.0 | 81.7 | ~2h |
| [MMT](https://github.com/open-mmlab/OpenUnReID/) ICLR'2020| ResNet50 | Market | 67.7 | 80.3 | 89.9 | 92.9 | ~6h |
| [SpCL](https://github.com/open-mmlab/OpenUnReID/) NIPS'2020 submission | ResNet50 | Market | 70.4 | 83.8 | 91.2 | 93.4 | ~3h |
| [strong_baseline](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | Market | 60.4 | 75.9 | 86.2 | 89.8 | ~3h |
| [Our stronger_baseline](https://github.com/JDAI-CV/fast-reid) | ResNet50 | Market | 66.7 | 80.0 | 89.2 | 92.2  | ~3h |
| [MLT] NeurIPS'2020 submission| ResNet50 | Market | 71.2 |83.9| 91.5| 93.2| ~ |

### Market1501 -> MSMT17

| Method | Source | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| DirectTransfer(R50) | Market1501 | 29.8% | 10.3% | 9.3% |
| Our method | DukeMTMC | 56.6% | 26.5% | - |

### DukeMTMC -> MSMT17
| Method | Source | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| DirectTransfer(R50) | DukeMTMC | 34.8% | 12.5% | 0.3% |
| Our method | DukeMTMC | 59.5% | 27.7% | - |
