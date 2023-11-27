# FastClas in FastReID

This project provides a baseline and example for image classification based on fastreid.

## Datasets Preparation

We refer to [pytorch tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) for dataset 
preparation. This is just an example for building a classification task based on fastreid. You can customize
your own datasets and model.

## Usage

If you want to train models with 4 gpus, you can run
```bash
python3 projects/FastClas/train_net.py --config-file projects/FastClas/config/base-clas.yml --num-gpus 4
```
