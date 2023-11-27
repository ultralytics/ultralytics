# FastReID Model Zoo and Baselines

## Introduction

This file documents collection of baselines trained with fastreid. All numbers were obtained with 1 NVIDIA V100 GPU.
The software in use were PyTorch 1.6, CUDA 10.1.

In addition to these official baseline models, you can find more models in [projects/](https://github.com/JDAI-CV/fast-reid/tree/master/projects).

### How to Read the Tables

- The "Name" column contains a link to the config file.
Running `tools/train_net.py` with this config file and 1 GPU will reproduce the model.

### Common Settings for all Person reid models

**BoT**:

[Bag of Tricks and A Strong Baseline for Deep Person Re-identification](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf). CVPRW2019, Oral.

**AGW**:

[ReID-Survey with a Powerful AGW Baseline](https://github.com/mangye16/ReID-Survey).

**MGN**:

[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

**SBS**:

stronger baseline on top of BoT:

Bag of Freebies(BoF):

1. Circle loss
2. Freeze backbone training
3. Cutout data augmentation & Auto Augmentation
4. Cosine annealing learning rate decay
5. Soft margin triplet loss

Bag of Specials(BoS):

1. Non-local block
2. GeM pooling

### Market1501 Baselines

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R50.yml) | ImageNet | 94.4% | 86.1% | 59.4% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth) |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R50-ibn.yml) | ImageNet | 94.9% | 87.6% | 64.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50-ibn.pth) |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_S50.yml) | ImageNet | 95.2% | 88.7% | 66.9% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_S50.pth) |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R101-ibn.yml) | ImageNet| 95.4% | 88.9% | 67.4% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R101-ibn.pth) |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: |:---: |
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R50.yml) | ImageNet | 95.3% | 88.2% | 66.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R50-ibn.yml) | ImageNet | 95.1% | 88.7% | 67.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_S50.yml) | ImageNet | 95.3% | 89.3% | 68.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R101-ibn.yml) | ImageNet | 95.5% | 89.5% | 69.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R101-ibn.pth) |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: |:---:|
| [SBS(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_R50.yml) | ImageNet | 95.4% | 88.2% | 64.8% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50.pth) |
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_R50-ibn.yml) | ImageNet | 95.7% | 89.3% | 67.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50-ibn.pth) |
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_S50.yml) | ImageNet | 95.8% | 89.4% | 67.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_S50.pth) |
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_R101-ibn.yml) | ImageNet | 96.3% | 90.3% | 70.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R101-ibn.pth) |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/mgn_R50-ibn.yml) | ImageNet | 95.8% | 89.8% | 67.7% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_mgn_R50-ibn.pth) |

### DukeMTMC Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R50.yml) | ImageNet | 87.2% | 77.0% | 42.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R50.pth) |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R50-ibn.yml) | ImageNet | 89.3% | 79.6% | 45.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R50-ibn.pth) |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_S50.yml) | ImageNet | 90.0% | 80.13% | 45.8% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_S50.pth) |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R101-ibn.yml) | ImageNet| 91.2% | 81.2% | 47.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R101-ibn.pth) |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50.yml) | ImageNet | 89.0% | 79.9% | 46.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50-ibn.yml) | ImageNet | 90.5% | 80.8% | 47.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_S50.yml) | ImageNet | 90.9% | 82.4% | 49.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R101-ibn.yml) | ImageNet | 91.7% | 82.3% | 50.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R101-ibn.pth) |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_R50.yml) | ImageNet | 90.3% | 80.3% | 46.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50.pth) |
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_R50-ibn.yml) | ImageNet | 90.8% | 81.2% | 47.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50-ibn.pth) |
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_S50.yml) | ImageNet | 91.0% | 81.4% | 47.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_S50.pth) |
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_R101-ibn.yml) | ImageNet | 91.9% | 83.6% | 51.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R101-ibn.pth) |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/mgn_R50-ibn.yml) | ImageNet | 91.1% | 82.0% | 46.8% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_mgn_R50-ibn.pth) |

### MSMT17 Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_R50.yml) | ImageNet | 74.1%  | 50.2% | 10.4% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R50.pth) |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_R50-ibn.yml) | ImageNet | 77.0% | 54.4% | 12.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R50-ibn.pth) |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_S50.yml) | ImageNet | 80.8% | 59.9% | 16.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_S50.pth) |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_R101-ibn.yml) | ImageNet| 81.0% | 59.4% | 15.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R101-ibn.pth) |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_R50.yml) | ImageNet | 78.3% | 55.6% | 12.9% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_R50-ibn.yml) | ImageNet | 81.2% | 59.7% | 15.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_S50.yml) | ImageNet | 82.6% | 62.6% | 17.7% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_R101-ibn.yml) | ImageNet | 82.0% | 61.4% | 17.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_R101-ibn.pth) |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R50.yml) | ImageNet | 81.8% | 58.4% | 13.9% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50.pth) |
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R50-ibn.yml) | ImageNet | 83.9% | 60.6% | 15.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50-ibn.pth) |
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_S50.yml) | ImageNet | 84.1% | 61.7% | 15.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_S50.pth) |
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R101-ibn.yml) | ImageNet | 84.8% | 62.8% | 16.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R101-ibn.pth) |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/mgn_R50-ibn.yml) | ImageNet | 85.1% | 65.4% | 18.4% | - |

### VeRi Baseline

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:| 
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VeRi/sbs_R50-ibn.yml) | ImageNet | 97.0%  | 81.9% | 46.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth) |

### VehicleID Baseline

**BoT**:  
Test protocol: 10-fold cross-validation; trained on 4 NVIDIA P40 GPU.

<table>
<thead>
  <tr>
    <th rowspan="3" align="center">Method</th>
    <th rowspan="3" align="center">Pretrained</th>
    <th colspan="6" align="center">Testset size</th>
    <th rowspan="3" align="center">download</th>
  </tr>
  <tr>
    <td colspan="2" align="center">Small</td>
    <td colspan="2" align="center">Medium</td>
    <td colspan="2" align="center">Large</td>
  </tr>
  <tr>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td nowrap align="center"><a href="https://github.com/JDAI-CV/fast-reid/blob/master/configs/VehicleID/bagtricks_R50-ibn.yml">BoT(R50-ibn)</a></td>
    <td align="center">ImageNet</td>
    <td align="center">86.6%</td>
    <td align="center">97.9%</td>
    <td align="center">82.9%</td>
    <td align="center">96.0%</td>
    <td align="center">80.6%</td>
    <td align="center">93.9%</td>
    <td align="center"><a href="https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth">model</a></td>
  </tr>
</tbody>
</table>

### VERI-Wild Baseline

**BoT**:  
Test protocol: Trained on 4 NVIDIA P40 GPU.

<table>
<thead>
  <tr>
    <th rowspan="3" align="center"> Method</th>
    <th rowspan="3" align="center">Pretrained</th>
    <th colspan="9" align="center">Testset size</th>
    <th rowspan="3" align="center">download</th>
  </tr>
  <tr>
    <td colspan="3" align="center">Small</td>
    <td colspan="3" align="center">Medium</td>
    <td colspan="3" align="center">Large</td>
  </tr>
  <tr>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
    <td align="center">mINP</td>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
    <td align="center">mINP</td>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
    <td align="center">mINP</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td nowrap align="center"><a href="https://github.com/JDAI-CV/fast-reid/blob/master/configs/VERIWild/bagtricks_R50-ibn.yml">BoT(R50-ibn)</a></td>
    <td align="center">ImageNet</td>
    <td align="center">96.4%</td>
    <td align="center">87.7%</td>
    <td align="center">69.2%</td>
    <td align="center">95.1%</td>
    <td align="center">83.5%</td>
    <td align="center">61.2%</td>
    <td align="center">92.5%</td>
    <td align="center">77.3%</td>
    <td align="center">49.8%</td>
    <td align="center"><a href="https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth">model</a></td>
  </tr>
</tbody>
</table>
