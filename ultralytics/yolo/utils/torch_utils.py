import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ultralytics.yolo.utils import check_version

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def DDP_model(model):
    # Model DDP creation with checks
    assert not check_version(torch.__version__, '1.12.0', pinned=True), \
        'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. ' \
        'Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if check_version(torch.__version__, '1.11.0'):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    # s = f'YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} '
    s = f'YOLOv5 🚀 torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    print(s)
    return torch.device(arg)
