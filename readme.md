# 基本配置

- 操作系统：ubuntu25.04
- 操作内核：6.14.0-1006-intel
- GPU：蓝戟或铭瑄B60Pro，实际上与硬件无关，因为torch并没有和GPU绑定但是驱动安装比较复杂，所以只要您能装上驱动，基本上可以正常进行操作
- 驱动及安装教程：https://github.com/intel/llm-scaler/blob/main/vllm/README.md/#1-getting-started-and-usagexit
- 驱动版本：multi-arc-bmg-offline-installer-25.38.4.1

# 如何安装环境

- 定制化安装torch支持：https://pytorch-extension.intel.com/installation
- 安装和我一样的torch版本支持

```bash
cd ultralytics
pip install -e .
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
# 下面两个可以不装，因为目前不支持多卡训练
# pip install intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# pip install oneccl_bind_pt==2.8.0+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

- 验证您是否安装成功

```bash
(B60) root@b60:~/ultralytics# python
Python 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.version.xpu)
20250101
>>> print(torch.xpu.is_available())
True
>>> print(torch.xpu.get_device_name(0))
Intel(R) Graphics [0xe211]
```

# 代码修改处

- ultralytics/ultralytics/utils/torch_utils.py/ time_sync()
- 目的：时间同步

```python
def time_sync():
    """Return PyTorch-accurate time."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.synchronize()
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
```

- 测试用例：无

---

- ultralytics/ultralytics/utils/torch_utils.py/ get_gpu_info()
- 目的：支持解析xpu信息

```python
@functools.lru_cache
def get_gpu_info(index):
    """Return a string with system GPU information, i.e. 'Tesla T4, 15102MiB'."""
    properties = torch.cuda.get_device_properties(index)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        properties = torch.xpu.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"
```

- 测试用例：与训练代码相同
- 测试结果：修改此处后会支持在训练的时候输出显卡信息

```bash
Ultralytics 8.3.231 🚀 Python-3.10.19 torch-2.8.0+xpu XPU:0 (Intel(R) Graphics [0xe211]
```

---

- ultralytics/ultralytics/utils/torch_utils.py/ select_device()
- 选择GPU的时候支持选择xpu，目前你可以填写多块GPU，但只会默认使用0卡

```python
def select_device(device="", newline=False, verbose=True):
    """Select the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object. Options are 'None', 'cpu', or
            'cuda', or '0' or '0,1,2,3'. Auto-selects the first available GPU, or CPU if no GPU is available.
        newline (bool, optional): If True, adds a newline at the end of the log string.
        verbose (bool, optional): If True, logs the device information.

    Returns:
        (torch.device): Selected device.

    Examples:
        >>> select_device("cuda:0")
        device(type='cuda', index=0)

        >>> select_device("cpu")
        device(type='cpu')

    Notes:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """
    if isinstance(device, torch.device) or str(device).startswith(("tpu", "intel")):
        return device

    s = f"Ultralytics {__version__} 🚀 Python-{PYTHON_VERSION} torch-{TORCH_VERSION} "
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'

    # Auto-select GPUs
    if "-1" in device:
        from ultralytics.utils.autodevice import GPUInfo

        # Replace each -1 with a selected GPU or remove it
        parts = device.split(",")
        selected = GPUInfo().select_idle_gpu(count=parts.count("-1"), min_memory_fraction=0.2)
        for i in range(len(parts)):
            if parts[i] == "-1":
                parts[i] = str(selected.pop(0)) if selected else ""
        device = ",".join(p for p in parts if p)

    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)

    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force torch.cuda.is_available() = False
    # ⭐新增：无 CUDA 但有 XPU 时，让 device="0" 走 XPU
    elif device and not torch.cuda.is_available() and hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            idx = int(device)
            if idx >= torch.xpu.device_count():
                raise ValueError(f"XPU index {idx} out of range")
            p = torch.xpu.get_device_properties(idx)
            mem = p.total_memory / (1 << 10)
            s += f"XPU:{idx} ({p.name}, {mem:.0f}MiB)\n"
            if verbose:
                LOGGER.info(s)
            return torch.device(f"xpu:{idx}")
        except:
            pass

    # ⭐修改：只有 CUDA 可用时才走 CUDA 分支
    elif device and torch.cuda.is_available():
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device

        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            LOGGER.info(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}'."
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # i.e. "0,1" -> ["0", "1"]
        space = " " * len(s)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(i)})\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # Prefer MPS if available
        s += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # Default auto-detect XPU
        props = torch.xpu.get_device_properties(0)
        mem = props.total_memory / (1 << 10)
        s += f"XPU:0 ({props.name}, {mem:.0f}MiB)\n"
        arg = "xpu"
    else:  # revert to CPU
        s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"
    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)  # reset OMP_NUM_THREADS for cpu training
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)
```

- 测试用例：训练时修改device参数

```python
from ultralytics import YOLO

model = YOLO("/root/ultralytics/ultralytics/cfg/models/v8/yolov8n.yaml")
model.train(
    data="coco128.yaml",
    epochs=50,
    imgsz=256,
    # device="xpu：1"
    device="xpu",
)
```

- 测试结果：支持训练

---

- ultralytics/ultralytics/utils/checks.py/ check_amp()
- 目的禁止AMP检查

```python
def check_amp(model):
    """Check the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLO model.

    If the checks fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP
    results, so AMP will be disabled during training.

    Args:
        model (torch.nn.Module): A YOLO model instance.

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLO11 model, else False.

    Examples:
        >>> from ultralytics import YOLO
        >>> from ultralytics.utils.checks import check_amp
        >>> model = YOLO("yolo11n.pt").model.cuda()
        >>> check_amp(model)
    """
    from ultralytics.utils.torch_utils import autocast

    device = next(model.parameters()).device  # get model device
    prefix = colorstr("AMP: ")
    if hasattr(torch, "xpu") and torch.xpu.is_available() and device.type == "xpu":
        LOGGER.warning(f"{prefix}Intel XPU detected. AMP is disabled (not supported on XPU).")
        return False

    if device.type in {"cpu", "mps"}:
        return False  # AMP only used on CUDA devices
    else:
        # GPUs that have issues with AMP
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )

        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}checks failed ❌. AMP training on {gpu} GPU may cause "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False

    def amp_allclose(m, im):
        """All close FP32 vs AMP results."""
        batch = [im] * 8
        imgsz = max(256, int(model.stride.max() * 4))  # max stride P5-32 and P6-64
        a = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # FP32 inference
        with autocast(enabled=True):
            b = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # AMP inference
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

    im = ASSETS / "bus.jpg"  # image to check
    LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks...")
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
    try:
        from ultralytics import YOLO

        assert amp_allclose(YOLO("yolo11n.pt"), im)
        LOGGER.info(f"{prefix}checks passed ✅")
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped. Offline and unable to download YOLO11n for AMP checks. {warning_msg}")
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}checks skipped. "
            f"Unable to load YOLO11n for AMP checks due to possible Ultralytics package modifications. {warning_msg}"
        )
    except AssertionError:
        LOGGER.error(
            f"{prefix}checks failed. Anomalies were detected with AMP on your system that may lead to "
            f"NaN losses or zero-mAP results, so AMP will be disabled during training."
        )
        return False
    return True
```

- 测试用例：正常训练
- 测试结果：显示如下内容不支持AMP

```bash
WARNING ⚠️ AMP: Intel XPU detected. AMP is disabled (not supported on XPU).
```

---

- ultralytics/engine/trainer.py \_clear_memory()
- 支持清除显存，但是我不得不说，由于我们的传参会改变数据类型，所以很可能传递到这里的参数是0或者1，那就永远不会用到这个
- 所以如果后面需要支持多卡训练，我们得改变一下参数传递的过程

```python
    def _clear_memory(self, threshold: float | None = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        elif self.device.type == "xpu":
            torch.xpu.empty_cache()
        else:
            torch.cuda.empty_cache()
```

> [!WARNING]
> 这是一个警告内容，请注意这里的说明。

- 测试用例：将显卡塞上一个模型再训练
- 测试结果：intelbug，当显存被占满时，会驱逐不活跃的显存至内存上，故无法测试，此问题我已经在一个月前反馈intel相关团队
- 此问题与底层驱动有关，暂时无法验证，待定

---

- ultralytics/engine/trainer.py \_get_memory()
- 目的：修改得到显存的方式

```python
    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "xpu":
            try:
                idx = self.device.index if isinstance(self.device, torch.device) else int(self.device)
                memory = torch.xpu.memory_allocated(idx)
                total = torch.xpu.get_device_properties(self.device).total_memory
                return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)
            except Exception:
                return 0.0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)
```

- 针对这段代码，我将提供最小验证证明显存计算的方式没有问题，他是以字节的方式展示数据

```bash
(B60) root@b60:~/ultralytics# python
Python 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
Ctrl click to launch VS Code Native REPL
>>> import torch
>>> torch.xpu.get_device_properties(0).total_memory
24385683456
>>> x = torch.randn((1024, 1024, 256), device="xpu")
>>> torch.xpu.memory_allocated(0)
1073741824
```

> [!WARNING]
> 这是一个警告内容，请注意这里的说明。
> 但是在实际运行时，当我使用yolov8x运行，并且图像输入为640时，他的显存占用非常小，但我觉得我的代码没有问题

```bash
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/50      1.18G      3.651       5.77        4.3        162        640: 100% ━━━━━━━━━━━━ 8/8 3.4s/it 27.1s
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━━━━━━─── 3/4 4.0s/it 6.4s<4.0s
```

---

# 合并测试用例

- 测试用例

```bash
/root/anaconda3/envs/B60/bin/python -m pytest -s -q test.py
```

- 命名为test.py，并创建文件

```python
import pytest
import torch

from ultralytics import YOLO

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU not available",
)


def test_yolo_xpu_forward():
    model = YOLO("/root/ultralytics/yolov8n.pt")  # 填入本地的模型
    model.to("xpu")
    x = torch.rand(1, 3, 64, 64, device="xpu")
    y = model.model(x)
    assert y is not None
    print("\n[XPU Test] YOLO XPU forward passed successfully ✔")
```

- 测试用例结果

```bash
(B60) root@b60:~/ultralytics# /root/anaconda3/envs/B60/bin/python -m pytest -s -q test.py
[W1128 16:05:14.607604493 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())

[XPU Test] YOLO XPU forward passed successfully ✔
.
=============================================================== slowest 30 durations ================================================================
1.27s call     test.py::test_yolo_xpu_forward

(2 durations < 0.005s hidden.  Use -vv to show these durations.)
1 passed in 10.53s
```

# 从源码安装的测试

```bash
git clone https://github.com/hzdzkjdxyjs/ultralytics.git ultralytics_b60
cd ultralytics_b60
conda activate B60 # 这是老环境，但我认为不需要新环境去验证
```

在当前目录下创建自动测试文件

```python
import pytest
import torch

from ultralytics import YOLO

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU not available",
)


def test_yolo_xpu_forward():
    model = YOLO("yolov8n.pt")
    model.to("xpu")
    x = torch.rand(1, 3, 64, 64, device="xpu")
    y = model.model(x)
    assert y is not None
    print("\n[XPU Test] YOLO XPU forward passed successfully ✔")
```

- 执行自动化脚本

```bash
(B60) root@b60:~/ultralytics_b60# /root/anaconda3/envs/B60/bin/python -m pytest -s -q test.py

[XPU Test] YOLO XPU forward passed successfully ✔
.
=============================================================== slowest 30 durations ================================================================
1.18s call     test.py::test_yolo_xpu_forward

(2 durations < 0.005s hidden.  Use -vv to show these durations.)
1 passed in 22.78s
```

- xpu支持测试
- 在当前目录下创建训练文件，修改device参数 xpu ；xpu：0；xpu：1；xpu：0，1

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="coco128.yaml", epochs=50, imgsz=256, device="xpu:0")
```

- 我必须强调一下这个的结果，无论你输入什么参数最后只会运行0卡，这是因为目前我没有看到有比较好的方式能支持多卡训练，所以我想等待之后再对这个代码进行修改，这不是bug！！！

- 长压测试，修改epochs为50轮
- 非常遗憾的来说，如果我从yaml文件从头开始训练，他的效果并不好
- 但我认为社区的生态不能仅仅局限于NV一张卡，所以我们可以先进行框架适配
- 之后再进行算子适配，这时候就需要督促Intel的团队了
- 如果仅仅预训练的权重来训练的话，效果会好一点

---

# 本次更改只支持单卡训练，为什么不支持多卡？

- 并非是torch及其依赖不支持多卡训练，因为我能够在llamafactory进行多卡训练
- 而是缺乏相应的参数和框架支持，
- 例如ultralytics/ultralytics/engine/trainer.py中的BaseTrainer

```python
self.device = select_device(self.args.device)
self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
```

- 我发现这两行代码会决定您的设备数量，但是主要是os.getenv("CUDA_VISIBLE_DEVICES")决定的
- ultralytics/ultralytics/engine/trainer.py中的BaseTrainer会进一步决定是否开启DDP

```python
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0
        self.ddp = world_size > 1 and "LOCAL_RANK" not in os.environ
```

- Intel也有类似的环境变量ZE_AFFINITY_MASK，但是如果我选择这个作为设备指定数字时
- 当真正传入ddp训练的设备参数就是0，1，这些数字了，此时再次进行设备选择时则会认为是cuda类设备
- 我也用了很多方法去修改，但是我发现因为yolo实在太依赖cuda了，以至于里面和多线程有关的变量都与cuda绑定死了例如RANK，world_size
- 所以我们很难去进行修改，我希望后期能够有一个集中的入口，比如将变量绑定到某个值上，而不是依赖于CUDA_VISIBLE_DEVICES
- 我后来放弃修改的还有一个原因是，如果我要改变整个框架的话，那就和贡献者的初衷违背了，尽量是最小改动
- 但是我提供了一些按理，希望对您框架调整是有启发的
  关于xpu如何启动多线程

```bash
torchrun --nproc_per_node=2 /root/LLaMA-Factory/test_dist.py
```

- test_dist.py

```python
# 必须先导入 IPEX，再导入 oneCCL
import os

import torch
import torch.distributed as dist


def main():
    # 通过 torchrun 传入的 LOCAL_RANK 识别当前进程使用的设备
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"[Before init] rank={rank}, local_rank={local_rank}")
    # ---- XPU 设备设置 ----
    torch.xpu.set_device(local_rank)
    print(f"XPU device set to {local_rank}: {torch.xpu.get_device_name(local_rank)}")
    # ---- CCL init ----
    dist.init_process_group(backend="ccl", rank=rank, world_size=world_size)
    print(f"[After init] Backend={dist.get_backend()}, Rank={dist.get_rank()}, WorldSize={dist.get_world_size()}")
    dist.barrier()
    print(f"[Rank {rank}] barrier passed.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```
