# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/collect_env.py
import importlib
import os
import re
import subprocess
import sys
from collections import defaultdict

import PIL
import numpy as np
import torch
import torchvision
from tabulate import tabulate

__all__ = ["collect_env_info"]


def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module():
    var_name = "FASTREID_ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, "bin", "cuobjdump")
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output(
                "'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True
            )
            output = output.decode("utf-8").strip().split("\n")
            sm = []
            for line in output:
                line = re.findall(r"\.sm_[0-9]*\.", line)[0]
                sm.append(line.strip("."))
            sm = sorted(set(sm))
            return ", ".join(sm)
        else:
            return so_file + "; cannot find cuobjdump"
    except Exception:
        # unhandled failure
        return so_file


def collect_env_info():
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    torch_version = torch.__version__

    # NOTE: the use of CUDA_HOME and ROCM_HOME requires the CUDA/ROCM build deps, though in
    # theory detectron2 should be made runnable with only the corresponding runtimes
    from torch.utils.cpp_extension import CUDA_HOME

    has_rocm = False
    if tuple(map(int, torch_version.split(".")[:2])) >= (1, 5):
        from torch.utils.cpp_extension import ROCM_HOME

        if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None):
            has_rocm = True
    has_cuda = has_gpu and (not has_rocm)

    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import fastreid  # noqa

        data.append(
            ("fastreid", fastreid.__version__ + " @" + os.path.dirname(fastreid.__file__))
        )
    except ImportError:
        data.append(("fastreid", "failed to import"))

    data.append(get_env_module())
    data.append(("PyTorch", torch_version + " @" + os.path.dirname(torch.__file__)))
    data.append(("PyTorch debug build", torch.version.debug))

    data.append(("GPU available", has_gpu))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

        if has_rocm:
            data.append(("ROCM_HOME", str(ROCM_HOME)))
        else:
            data.append(("CUDA_HOME", str(CUDA_HOME)))

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cuda_arch_list:
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    data.append(("Pillow", PIL.__version__))

    try:
        data.append(
            (
                "torchvision",
                str(torchvision.__version__) + " @" + os.path.dirname(torchvision.__file__),
            )
        )
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec("torchvision._C").origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(("torchvision arch flags", msg))
            except ImportError:
                data.append(("torchvision._C", "failed to find"))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    try:
        import fvcore

        data.append(("fvcore", fvcore.__version__))
    except ImportError:
        pass

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


if __name__ == "__main__":
    try:
        import detectron2  # noqa
    except ImportError:
        print(collect_env_info())
    else:
        from supervision.tracker.utils.fast_reid.fastreid.utils.collect_env import collect_env_info

        print(collect_env_info())
