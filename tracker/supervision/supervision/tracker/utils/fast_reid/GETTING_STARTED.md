# Getting Started with Fastreid

## Prepare pretrained model

If you use backbones supported by fastreid, you do not need to do anything. It will automatically download the pre-train models.
But if your network is not connected, you can download pre-train models manually and put it in `~/.cache/torch/checkpoints`.

If you want to use other pre-train models, such as MoCo pre-train, you can download by yourself and set the pre-train model path in `configs/Base-bagtricks.yml`.

## Compile with cython to accelerate evalution

```bash
cd fastreid/evaluation/rank_cylib; make all
```

## Training & Evaluation in Command Line

We provide a script in "tools/train_net.py", that is made to train all the configs provided in fastreid.
You may want to use it as a reference to write your own training script.

To train a model with "train_net.py", first setup up the corresponding datasets following [datasets/README.md](https://github.com/JDAI-CV/fast-reid/tree/master/datasets), then run:

```bash
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml MODEL.DEVICE "cuda:0"
```

The configs are made for 1-GPU training.

If you want to train model with 4 GPUs, you can run:

```bash
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --num-gpus 4
```

If you want to train model with multiple machines, you can run:

```
# machine 1
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

python3 tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml \
--num-gpus 4 --num-machines 2 --machine-rank 0 --dist-url tcp://ip:port 

# machine 2
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

python3 tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml \
--num-gpus 4 --num-machines 2 --machine-rank 1 --dist-url tcp://ip:port 
```

Make sure the dataset path and code are the same in different machines, and machines can communicate with each other. 

To evaluate a model's performance, use

```bash
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --eval-only \
MODEL.WEIGHTS /path/to/checkpoint_file MODEL.DEVICE "cuda:0"
```

For more options, see `python3 tools/train_net.py -h`.
