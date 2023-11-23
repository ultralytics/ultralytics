# FastFace in FastReID

This project provides a baseline for face recognition.

## Datasets Preparation

| Function | Dataset |
| --- | --- |
| Train | MS-Celeb-1M |
| Test-1 | LFW      |
| Test-2 | CPLFW |
| Test-3 | CALFW |
| Test-4 | VGG2_FP |
| Test-5 | AgeDB-30 |
| Test-6 | CFP_FF |
| Test-7 | CFP-FP |

We do data wrangling following [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) instruction.

## Dependencies

- bcolz
- mxnet (optional) if you want to read `.rec` directly

## Experiment Results

We refer to [insightface_pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) as our baseline methods, and on top of it, we use circle loss and cosine lr scheduler.

| Method | LFW(%) | CFP-FF(%) | CFP-FP(%)| AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| :---: | :---: | :---: |:---: | :---: | :---: | :---: | :---: |
| [insightface_pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) | 99.52 | 99.62 | 95.04 | 96.22 | 95.57 | 91.07 | 93.86 |
| ir50_se | 99.70 | 99.60 | 96.43 | 97.87 | 95.95 | 91.10 | 94.32 |
| ir100_se | 99.65 | 99.69 | 97.10 |  97.98 | 96.00 | 91.53 | 94.62 |
| ir50_se_0.1 |  |  |  |   |  |  |  |
| ir100_se_0.1 |  |  |  |  |  |  |  |
