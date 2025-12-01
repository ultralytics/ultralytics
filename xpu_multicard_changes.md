# XPU 多卡支持改动说明（基于 main 对比）

## 关键改动与代码片段

### 1) `ultralytics/utils/torch_utils.py::select_device`
**目的：** 兼容 `device=xpu:0,1` 的多卡写法，同时仍能对越界/非法输入给出明确提示。

**当前实现：**
```python
elif device.startswith("xpu"):  # Intel XPU
    index_str = device.split(":", 1)[1] if ":" in device else "0"
    index_list = [int(i) for i in index_str.split(",") if i]
    if not index_list:
        index_list = [0]
    if len(index_list) > 1:
        # Limit visible XPUs to the requested set
        os.environ["ZE_AFFINITY_MASK"] = ",".join(str(i) for i in index_list)
    index = index_list[0]
    if verbose:
        info = get_xpu_info(index)
        s += f"XPU:{index} ({info})\n"
        LOGGER.info(s if newline else s.rstrip())
    return torch.device("xpu", index)
```
> 相比 main：main 直接 `int(index_str)`，遇到 `0,1` 会抛 `ValueError`。现在解析逗号列表并设置 `ZE_AFFINITY_MASK`，超出索引仍由 torch 抛错。

### 2) `ultralytics/engine/trainer.py`
- **保留原始设备字符串**用于 world_size 计算，避免多 XPU 被截断；CUDA 流程保持原样。
- **DDP 初始化 XPU 分支**：加载 IPEX/oneCCL，使用 `LOCAL_RANK` 绑卡并 `backend="ccl"`。

```python
original_device_arg = self.args.device
self.device = select_device(self.args.device)
if isinstance(original_device_arg, str) and original_device_arg.lower().startswith("xpu") and "," in original_device_arg:
    self.args.device = original_device_arg.replace(" ", "")
```

```python
def _setup_ddp(self):
    local_rank = int(os.getenv("LOCAL_RANK", RANK))
    if self.device.type == "xpu":
        import intel_extension_for_pytorch  # noqa: F401
        import oneccl_bindings_for_pytorch  # noqa: F401
        torch.xpu.set_device(local_rank)
        self.device = torch.device("xpu", local_rank)
        dist.init_process_group(
            backend="ccl",
            timeout=timedelta(seconds=10800),
            rank=RANK,
            world_size=self.world_size,
        )
    else:
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        ...
```

- **AMP 兼容 XPU**：优先尝试 `torch.amp.GradScaler("xpu")`，失败则关闭 AMP，避免缺口退出。
```python
if self.device.type == "xpu":
    try:
        self.scaler = torch.amp.GradScaler("xpu", enabled=self.amp)
    except Exception:
        self.amp = False
        self.scaler = None
        LOGGER.warning("AMP scaler for XPU not available, disabling AMP.")
```

### 3) `ultralytics/engine/validator.py`
- **设备选择按用户硬件类型**：只要传入 `xpu*` 就强制用 XPU；`cuda*` 或空字符串走 CUDA；其他回退 CPU。CUDA 路径不变。
```python
if RANK == -1:
    eval_device = select_device(self.args.device)
else:
    device_arg = str(self.args.device or "").lower()
    if device_arg.startswith("xpu"):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Requested XPU device but torch.xpu is not available.")
        eval_device = torch.device("xpu", RANK)
    elif device_arg.startswith("cuda") or device_arg == "":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda is not available.")
        eval_device = torch.device("cuda", RANK)
    else:
        eval_device = torch.device("cpu")
```

- **验证阶段 loss 归约**：XPU 不支持 `ReduceOp.AVG`，改为 `SUM` 再在 rank0 手动平均。
```python
loss = self.loss.clone().detach()
if trainer.world_size > 1:
    dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
    if RANK == 0:
        loss /= trainer.world_size
```

## 原因与场景说明
- 原始 main 仅支持单 XPU，传入 `device=xpu:0,1` 会在 `int('0,1')` 时报错；本改动显式解析逗号分隔索引，兼容多卡训练。
- DDP 需要提前注册 CCL backend 并按 `LOCAL_RANK` 绑卡，否则多 XPU 训练无法初始化。
- XPU 平台缺少 `ReduceOp.AVG` 和部分 AMP Scaler 接口，改为 SUM 再均值、或在缺口时关闭 AMP，以保证训练流程可继续。
- CUDA 路径保持不变；只有用户明确请求 XPU 时才走 XPU 分支，避免误判设备。
