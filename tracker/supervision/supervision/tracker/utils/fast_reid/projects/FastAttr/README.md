# FastAttr in FastReID

This project provides a strong baseline for pedestrian attribute recognition.

## Datasets Preparation

We use `PA100k` to evaluate the model's performance.
You can do download dataset from [HydraPlus-Net](https://github.com/xh-liu/HydraPlus-Net).

## Usage

The training config file can be found in `projects/FastAttr/config`, which you can use to reproduce the results of the repo.

For example

```bash
python3 projects/FastAttr/train_net.py --config-file projects/FastAttr/configs/pa100.yml --num-gpus 4
```

## Experiment Results

We refer to [A Strong Baseline of Pedestrian Attribute Recognition](https://github.com/valencebond/Strong_Baseline_of_Pedestrian_Attribute_Recognition/tree/master) as our baseline methods and conduct the experiment
with 4 GPUs.
More details can be found in the config file and code.

### PA100k

| Method | Pretrained | mA | Accu | Prec | Recall | F1 | 
| :---: | :---: | :---: |:---: | :---: | :---: | :---: |
| attribute baseline | ImageNet | 80.50 | 78.84 | 87.24 | 87.12 | 86.78 | 
| FastAttr | ImageNet | 77.57 | 78.03 | 88.39 | 84.98 | 86.65 | 
