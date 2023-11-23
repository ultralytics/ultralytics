<img src=".github/FastReID-Logo.png" width="300" >

[![Gitter](https://badges.gitter.im/fast-reid/community.svg)](https://gitter.im/fast-reid/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Gitter: [fast-reid/community](https://gitter.im/fast-reid/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link)

Wechat: 

<img src=".github/wechat_group.png" width="150" >


FastReID is a research platform that implements state-of-the-art re-identification algorithms. It is a ground-up rewrite of the previous version, [reid strong baseline](https://github.com/michuanhaohao/reid-strong-baseline).

## What's New

- [Sep 2021] [DG-ReID](https://github.com/xiaomingzhid/sskd) is updated, you can check the [paper](https://arxiv.org/pdf/2108.05045.pdf).
- [June 2021] [Contiguous parameters](https://github.com/PhilJd/contiguous_pytorch_params) is supported, now it can
  accelerate ~20%.
- [May 2021] Vision Transformer backbone supported, see `configs/Market1501/bagtricks_vit.yml`.
- [Apr 2021] Partial FC supported in [FastFace](projects/FastFace)!
- [Jan 2021] TRT network definition APIs in [FastRT](projects/FastRT) has been released! 
Thanks for [Darren](https://github.com/TCHeish)'s contribution.
- [Jan 2021] NAIC20(reid track) [1-st solution](projects/NAIC20) based on fastreid has been released！
- [Jan 2021] FastReID V1.0 has been released！🎉
  Support many tasks beyond reid, such image retrieval and face recognition. See [release notes](https://github.com/JDAI-CV/fast-reid/releases/tag/v1.0.0).
- [Oct 2020] Added the [Hyper-Parameter Optimization](projects/FastTune) based on fastreid. See `projects/FastTune`.
- [Sep 2020] Added the [person attribute recognition](projects/FastAttr) based on fastreid. See `projects/FastAttr`.
- [Sep 2020] Automatic Mixed Precision training is supported with `apex`. Set `cfg.SOLVER.FP16_ENABLED=True` to switch it on.
- [Aug 2020] [Model Distillation](projects/FastDistill) is supported, thanks for [guan'an wang](https://github.com/wangguanan)'s contribution.
- [Aug 2020] ONNX/TensorRT converter is supported.
- [Jul 2020] Distributed training with multiple GPUs, it trains much faster.
- Includes more features such as circle loss, abundant visualization methods and evaluation metrics, SoTA results on conventional, cross-domain, partial and vehicle re-id, testing on multi-datasets simultaneously, etc.
- Can be used as a library to support [different projects](projects) on top of it. We'll open source more research projects in this way.
- Remove [ignite](https://github.com/pytorch/ignite)(a high-level library) dependency and powered by [PyTorch](https://pytorch.org/).

We write a [fastreid intro](https://l1aoxingyu.github.io/blogpages/reid/fastreid/2020/05/29/fastreid.html) 
and [fastreid v1.0](https://l1aoxingyu.github.io/blogpages/reid/fastreid/2021/04/28/fastreid-v1.html) about this toolbox.

## Changelog

Please refer to [changelog.md](CHANGELOG.md) for details and release history.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

See [GETTING_STARTED.md](GETTING_STARTED.md).

Learn more at out [documentation](https://fast-reid.readthedocs.io/). And see [projects/](projects) for some projects that are build on top of fastreid.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Fastreid Model Zoo](MODEL_ZOO.md).

## Deployment

We provide some examples and scripts to convert fastreid model to Caffe, ONNX and TensorRT format in [Fastreid deploy](tools/deploy).

## License

Fastreid is released under the [Apache 2.0 license](LICENSE).

## Citing FastReID

If you use FastReID in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
```
