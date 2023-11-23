# Installation

## Requirements

- Linux or macOS with python ≥ 3.6
- PyTorch ≥ 1.6
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- [yacs](https://github.com/rbgirshick/yacs)
- Cython (optional to compile evaluation code)
- tensorboard (needed for visualization): `pip install tensorboard`
- gdown (for automatically downloading pre-train model)
- sklearn
- termcolor
- tabulate
- [faiss](https://github.com/facebookresearch/faiss) `pip install faiss-cpu`



# Set up with Conda
```shell script
conda create -n fastreid python=3.7
conda activate fastreid
conda install pytorch==1.6.0 torchvision tensorboard -c pytorch
pip install -r docs/requirements.txt
```

# Set up with Dockder
comming soon
