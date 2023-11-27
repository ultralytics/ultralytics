# Changelog

### v1.3

#### New Features
- Vision Transformer backbone, see config in `configs/Market1501/bagtricks_vit.yml`
- Self-Distillation with EMA update
- Gradient Clip

#### Improvements
- Faster dataloader with pre-fetch thread and cuda stream
- Optimize DDP training speed by removing `find_unused_parameters` in DDP


### v1.2 (06/04/2021)

#### New Features

- Multiple machine training support
- [RepVGG](https://github.com/DingXiaoH/RepVGG) backbone 
- [Partial FC](projects/FastFace)

#### Improvements

- Torch2trt pipeline 
- Decouple linear transforms and softmax
- config decorator

### v1.1 (29/01/2021)

#### New Features

- NAIC20(reid track) [1-st solution](projects/NAIC20) 
- Multi-teacher Knowledge Distillation
- TRT network definition APIs in [FastRT](projects/FastRT)

#### Bug Fixes

#### Improvements