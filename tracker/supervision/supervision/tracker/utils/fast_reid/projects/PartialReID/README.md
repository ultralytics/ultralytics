# DSR in FastReID
**Deep Spatial Feature Reconstruction for Partial Person Re-identification**

Lingxiao He, Xingyu Liao

[[`CVPR2018`](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_Deep_Spatial_Feature_CVPR_2018_paper.pdf)] [[`BibTeX`](#CitingDSR)] 

**Foreground-aware Pyramid Reconstruction for Alignment-free Occluded Person Re-identification**

Lingxiao He, Xingyu Liao

[[`ICCV2019`](http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Foreground-Aware_Pyramid_Reconstruction_for_Alignment-Free_Occluded_Person_Re-Identification_ICCV_2019_paper.pdf)] [[`BibTeX`](#CitingFPR)]

## News！

[1] The old_version code can be check in [old_version](https://github.com/JDAI-CV/Partial-Person-ReID), you can obtain the same result published in paper, and the new version code is updating, please waiting!

## Installation

First install FastReID, and then put Partial Datasets in directory datasets. The whole framework of FastReID-DSR is
<div align="center">
<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FSherlockWorkspace%2F1nVTE3Sn5c.jpg?alt=media&token=e7e9fcfc-4fc1-49c8-bcf4-c007028fdd25" width="700px" />
</div>

and the detail you can refer to
## Datasets

The datasets can find in [Google Drive](https://drive.google.com/file/d/1p7Jvo-RJhU_B6hf9eAhIEFNhvrzM5cdh/view?usp=sharing)

PartialREID---gallery: 300 images of 60 ids, query: 300 images of 60 ids

PartialiLIDS---gallery: 119 images of 119 ids, query: 119 images of 119 ids

OccludedREID---gallery: 1,000 images of 200 ids, query: 1,000 images of 200 ids

## Training and Evaluation

To train a model, run:
```bash
python3 projects/PartialReID/train_net.py --config-file <config.yaml>
```

For example, to train the re-id network with IBN-ResNet-50 Backbone
one should execute:
```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python3 projects/PartialReID/train_net.py --config-file 'projects/PartialReID/configs/partial_market.yml'
```

## Results

| Method | PartialREID | OccludedREID | PartialiLIDS |
|:--:|:--:|:--:|:--:|
|   | Rank@1 (mAP)| Rank@1 (mAP)| Rank@1 (mAP)|
| DSR (CVPR’18)  |73.7(68.1) |72.8(62.8)|64.3(58.1)| 
| FPR (ICCV'19) | 81.0(76.6)|78.3(68.0)|68.1(61.8)| 
| FastReID-DSR | 82.7(76.8)|81.6(70.9)|73.1(79.8) | 

## <a name="CitingDSR"></a >Citing DSR and Citing FPR

If you use DSR or FPR, please use the following BibTeX entry.

```
@inproceedings{he2018deep,
  title={Deep spatial feature reconstruction for partial person re-identification: Alignment-free approach},
  author={He, Lingxiao and Liang, Jian and Li, Haiqing and Sun, Zhenan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
@inproceedings{he2019foreground,
  title={Foreground-aware Pyramid Reconstruction for Alignment-free Occluded Person Re-identification},
  author={He, Lingxiao and Wang, Yinggang and Liu, Wu and Zhao, He and Sun, Zhenan and Feng, Jiashi},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
