# Semi-Supervised Domain Generalizable Person Re-Identification (SSKD)

## Introduction

SSKD is implemented based on **FastReID v1.0.0**. You can refer to [sskd github link](https://github.com/xiaomingzhid/sskd) It provides a semi-supervised feature learning framework to learn domain-general representations. The framework is shown in 

<img src="images/framework.png" width="850" >

## Dataset

**FastHuman** is very challenging, as it contains more complex application scenarios and large-scale training, testing datasets. It has diverse images from different application scenarios including campus, airport, shopping mall, street, and railway station.
It contains 447,233 labeled images of 40,061 subjects captured by 82 cameras. The details of FastHuman, you can refer to [paper](https://arxiv.org/pdf/2108.05045.pdf).

| Source Domain |  \#subjects | \#images | \#cameras | collection place |
| ----- | :------: | :---------: | :----: | :------: |
| CUHK03|  1,090 | 14,096 | 2 | campus |  
| SAIVT | 152   | 7,150  | 8 | buildings |
| AirportALERT | 9,651 | 30,243 | 6 | airport |
|iLIDS|  300   | 4,515  | 2 | airport |
|PKU  |  114   | 1,824  | 2 | campus |
|PRAI |   1,580 | 39,481| 2 | aerial imagery |
|SenseReID | 1,718 | 3,338  | 2 | unknown |
|SYSU | 510  | 30,071 | 4 | campus |
|Thermalworld | 409   | 8,103  | 1 | unknown |
|3DPeS  | 193  | 1,012  | 1 | outdoor  |
|CAVIARa | 72  | 1,220  | 1 | shopping mall |
|VIPeR | 632   | 1,264  | 2 | unknown |
|Shinpuhkan| 24 | 4,501  | 8 | unknown |
|WildTrack | 313 | 33,979 | 7| outdoor |
|cuhk-sysu | 11,934| 34,574 | 1| street |
|LPW |  2,731 | 30,678 | 4 | street |
|GRID |  1,025 | 1,275 | 8 | underground |
|Total | 31,423| 246,049 | 57 | - |


|Unseen Domain|  \#subjects | \#images | \#cameras | collection place  |
| ----- | :------: | :---------: | :----: | :------: |
|Market1501 | 1,501  | 32,217 | 6 | campus |
|DukeMTMC | 1,812 | 36,441 | 8 | campus |
|MSMT17 | 4,101 | 126,441| 15| campus |
|PartialREID | 60 | 600| 6|campus |
|PartialiLIDS | 119  | 238 | 2 | airport |
|OccludedREID | 200  | 2,000| 5| campus |
|CrowdREID | 845  | 3,257 | 11 | railway station| 
|Total   | 8,638  | 201,184| 49 | - |

**YouTube-Human** is a unlabeled human dataset. You can download the Street-View video from YouTube website, and the use the human detection algorithm ([centerX](https://github.com/JDAI-CV/centerX)) to obtain the human images.

## Training & Evaluation

The whole training process is divided into two stages:

- Train a student model (r34-ibn) and a teacher model (r101_ibn), you can run:
```bash
python3 projects/Basic_Project/train_net.py --config-file projects/Basic_Project/configs/r34-ibn.yml --num-gpu 4
python3 projects/Basic_Project/train_net.py --config-file projects/Basic_Project/configs/r101-ibn.yml --num-gpu 4
```
- Train the student model based unlabeled dataset and sskd, you can run:
```bash
python3 projects/SSKD/train_net.py --config-file projects/SSKD/configs/sskd.yml --num-gpu 4
```
### Results
<img src="images/result1.png" width="550" >
<img src="images/result2.png" width="500" >
Other some experimental results you could find in our [arxiv paper](https://arxiv.org/pdf/2108.05045.pdf).

## Reference Project
- [fastreid](https://github.com/JDAI-CV/fast-reid)
- [centerX](https://github.com/JDAI-CV/centerX)

## Citation
If you use **fastreid** or **sskd** in your research, please give credit to the following papers:

```BibTeX
@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
```
```BibTeX
@article{he2021semi,
  title={Semi-Supervised Domain Generalizable Person Re-Identification},
  author={He, Lingxiao and Liu, Wu and Liang, Jian and Zheng, Kecheng and Liao, Xingyu and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2108.05045},
  year={2021}
}
```
