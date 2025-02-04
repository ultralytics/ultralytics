# RGB-IR-Ultralytics

## 支持多模态输入的YOLO

该项目扩展了[Ultralytics]([ultralytics/ultralytics: Ultralytics YOLO11 🚀](https://github.com/ultralytics/ultralytics))8.3.70版本，为物体检测任务添加了对 RGB+IR（可见光 + 红外）双模输入的支持。此外，它还提供了[LLVIP](https://github.com/bupt-ai-cz/LLVIP)数据集的调整和改编版本，名为[LLVIP-For-Ultralytics](https://github.com/Tdzdele/LLVIP-For-Ultralytics)，用于训练和测试。如果用户在研究或应用中使用，我们鼓励他们引用原始作品。

**注意：**双模态输入训练需要更多的内存。如果训练自动终止并显示 “Killed” 提示符，则可能表示内存溢出。请在训练期间监控内存使用情况，并根据设备的规格调整超参数。
