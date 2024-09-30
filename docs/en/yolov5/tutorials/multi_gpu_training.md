---
comments: true
description: Learn how to train YOLOv5 on multiple GPUs for optimal performance. Guide covers single and multiple machine setups.
keywords: YOLOv5, multiple GPUs, machine learning, deep learning, PyTorch, data parallel, distributed data parallel
---

📚 This guide explains how to properly use **multiple** GPUs to train a dataset with YOLOv5 🚀 on single or multiple machine(s).

## Before You Start

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

💡 ProTip! **Docker Image** is recommended for all Multi-GPU trainings. See [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

💡 ProTip! `torch.distributed.run` replaces `torch.distributed.launch` in **[PyTorch](https://www.ultralytics.com/glossary/pytorch)>=1.9**. See [docs](https://pytorch.org/docs/stable/distributed.html) for details.

## Training

Select a pretrained model to start training from. Here we select [YOLOv5s](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml), the smallest and fastest model available. See our README [table](https://github.com/ultralytics/yolov5#pretrained-checkpoints) for a full comparison of all models. We will train this model with Multi-GPU on the [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset.

<p align="center"><img width="700" alt="YOLOv5 Models" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-model-comparison.avif"></p>

### Single GPU

```bash
python train.py  --batch 64 --data coco.yaml --weights yolov5s.pt --device 0
```

### Multi-GPU [DataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel) Mode (⚠️ not recommended)

You can increase the `device` to use Multiple GPUs in DataParallel mode.

```bash
python train.py  --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
```

This method is slow and barely speeds up training compared to using just 1 GPU.

### Multi-GPU [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) Mode (✅ recommended)

You will have to pass `python -m torch.distributed.run --nproc_per_node`, followed by the usual arguments.

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
```

`--nproc_per_node` specifies how many GPUs you would like to use. In the example above, it is 2.
`--batch ` is the total batch-size. It will be divided evenly to each GPU. In the example above, it is 64/2=32 per GPU.

The code above will use GPUs `0... (N-1)`.

<details>
  <summary>Use specific GPUs (click to expand)</summary>

You can do so by simply passing `--device` followed by your specific GPUs. For example, in the code below, we will use GPUs `2,3`.

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights '' --device 2,3
```

</details>

<details>
  <summary>Use SyncBatchNorm (click to expand)</summary>

[SyncBatchNorm](https://pytorch.org/docs/master/generated/torch.nn.SyncBatchNorm.html) could increase [accuracy](https://www.ultralytics.com/glossary/accuracy) for multiple gpu training, however, it will slow down training by a significant factor. It is **only** available for Multiple GPU DistributedDataParallel training.

It is best used when the batch-size on **each** GPU is small (<= 8).

To use SyncBatchNorm, simple pass `--sync-bn` to the command like below,

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights '' --sync-bn
```

</details>

<details>
  <summary>Use Multiple machines (click to expand)</summary>

This is **only** available for Multiple GPU DistributedDataParallel training.

Before we continue, make sure the files on all machines are the same, dataset, codebase, etc. Afterward, make sure the machines can communicate to each other.

You will have to choose a master machine(the machine that the others will talk to). Note down its address(`master_addr`) and choose a port(`master_port`). I will use `master_addr = 192.168.1.1` and `master_port = 1234` for the example below.

To use it, you can do as the following,

```bash
# On master machine 0
python -m torch.distributed.run --nproc_per_node G --nnodes N --node_rank 0 --master_addr "192.168.1.1" --master_port 1234 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights ''
```

```bash
# On machine R
python -m torch.distributed.run --nproc_per_node G --nnodes N --node_rank R --master_addr "192.168.1.1" --master_port 1234 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights ''
```

where `G` is number of GPU per machine, `N` is the number of machines, and `R` is the machine number from `0...(N-1)`. Let's say I have two machines with two GPUs each, it would be `G = 2` , `N = 2`, and `R = 1` for the above.

Training will not start until <b>all </b> `N` machines are connected. Output will only be shown on master machine!

</details>

### Notes

- Windows support is untested, Linux is recommended.
- `--batch ` must be a multiple of the number of GPUs.
- GPU 0 will take slightly more memory than the other GPUs as it maintains EMA and is responsible for checkpointing etc.
- If you get `RuntimeError: Address already in use`, it could be because you are running multiple trainings at a time. To fix this, simply use a different port number by adding `--master_port` like below,

```bash
python -m torch.distributed.run --master_port 1234 --nproc_per_node 2 ...
```

## Results

DDP profiling results on an [AWS EC2 P4d instance](../environments/aws_quickstart_tutorial.md) with 8x A100 SXM4-40GB for YOLOv5l for 1 COCO [epoch](https://www.ultralytics.com/glossary/epoch).

<details>
  <summary>Profiling code</summary>

```bash
# prepare
t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
cd .. && rm -rf app && git clone https://github.com/ultralytics/yolov5 -b master app && cd app
cp data/coco.yaml data/coco_profile.yaml

# profile
python train.py --batch-size 16 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0
python -m torch.distributed.run --nproc_per_node 2 train.py --batch-size 32 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0,1
python -m torch.distributed.run --nproc_per_node 4 train.py --batch-size 64 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0,1,2,3
python -m torch.distributed.run --nproc_per_node 8 train.py --batch-size 128 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0,1,2,3,4,5,6,7
```

</details>

| GPUs<br>A100 | batch-size | CUDA_mem<br><sup>device0 (G) | COCO<br><sup>train | COCO<br><sup>val |
| ------------ | ---------- | ---------------------------- | ------------------ | ---------------- |
| 1x           | 16         | 26GB                         | 20:39              | 0:55             |
| 2x           | 32         | 26GB                         | 11:43              | 0:57             |
| 4x           | 64         | 26GB                         | 5:57               | 0:55             |
| 8x           | 128        | 26GB                         | 3:09               | 0:57             |

## FAQ

If an error occurs, please read the checklist below first! (It could save your time)

<details>
  <summary>Checklist (click to expand) </summary>

<ul>
    <li>Have you properly read this post?  </li>
    <li>Have you tried to re-clone the codebase? The code changes <b>daily</b>.</li>
    <li>Have you tried to search for your error? Someone may have already encountered it in this repo or in another and have the solution. </li>
    <li>Have you installed all the requirements listed on top (including the correct Python and Pytorch versions)? </li>
    <li>Have you tried in other environments listed in the "Environments" section below? </li>
    <li>Have you tried with another dataset like coco128 or coco2017? It will make it easier to find the root cause. </li>
</ul>

If you went through all the above, feel free to raise an Issue by giving as much detail as possible following the template.

</details>

## Supported Environments

Ultralytics provides a range of ready-to-use environments, each pre-installed with essential dependencies such as [CUDA](https://developer.nvidia.com/cuda-zone), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), to kickstart your projects.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**: [GCP Quickstart Guide](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**: [AWS Quickstart Guide](../environments/aws_quickstart_tutorial.md)
- **Azure**: [AzureML Quickstart Guide](../environments/azureml_quickstart_tutorial.md)
- **Docker**: [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

This badge indicates that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are successfully passing. These CI tests rigorously check the functionality and performance of YOLOv5 across various key aspects: [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py), and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py). They ensure consistent and reliable operation on macOS, Windows, and Ubuntu, with tests conducted every 24 hours and upon each new commit.

## Credits

We would like to thank @MagicFrogSJTU, who did all the heavy lifting, and @glenn-jocher for guiding us along the way.
