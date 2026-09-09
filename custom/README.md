# Custom Ultralytics workspace

这个目录保存本仓库的项目级扩展和维护约定。官方 `ultralytics/` 目录尽量保持接近
`upstream/main`，这样上游更新时冲突只会集中在真正需要改动的文件。

## 目录约定

```text
custom/
├── configs/
│   ├── datasets/
│   │   └── private.example.yaml
│   └── models/
│       └── README.md
└── scripts/
    └── sync_upstream.sh
```

- `configs/datasets/` 保存数据集配置。私有配置从 `private.example.yaml` 复制为
  `private.yaml`，后者已被 Git 忽略。
- `configs/models/` 保存项目自己的模型 YAML。只有在确实需要新增模块时，才修改
  `ultralytics/nn/` 或 `ultralytics/nn/tasks.py`。
- `scripts/sync_upstream.sh` 用于从官方仓库获取更新并在当前分支执行 rebase。
- 权重、数据集、`runs/` 和实验输出保留在仓库外或被 `.gitignore` 忽略。

## 第一次配置

在仓库根目录执行：

```bash
git remote add upstream https://github.com/ultralytics/ultralytics.git
cp custom/configs/datasets/private.example.yaml custom/configs/datasets/private.yaml
```

如果 `upstream` 已存在，只需确认地址：

```bash
git remote get-url upstream
```

编辑 `private.yaml` 的 `path` 和 `names`，不要把真实数据路径、图片、标签或权重提交到
Git。训练时直接使用：

```bash
yolo detect train \
  model=yolo26n.pt \
  data=custom/configs/datasets/private.yaml \
  epochs=100 imgsz=640
```

## 分支和提交

`main` 只接收已经验证的变更。每项工作创建独立分支：

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/custom-block
```

建议把自定义模块、模型 YAML、数据配置模板和文档拆成独立提交。不要在同一个提交中
混入大范围格式化或无关的上游修改。

## 新增自定义网络模块

如果只调整已有模块的排列，优先新增模型 YAML，不改 Python 代码。如果必须新增模块，
按下面顺序接入：

1. 在 `ultralytics/nn/modules/` 新增模块文件。
2. 在 `ultralytics/nn/modules/__init__.py` 导出模块。
3. 在 `ultralytics/nn/tasks.py` 导入模块，使模型 YAML 可以解析模块名。
4. 模块需要自动处理输入输出通道时，再加入 `parse_model()` 的 `base_modules`；
   支持 YAML 重复次数时才加入 `repeat_modules`。
5. 在 `custom/configs/models/` 增加模型 YAML，并用公开数据集做最小训练验证。

尽量不要修改 `default.yaml`、公共数据集 YAML 或现有任务的训练流程。需要改变训练
行为时，优先通过 `model.train(..., trainer=...)` 传入派生的 Trainer。

## 上游同步后的验证

每次同步后至少运行：

```bash
yolo checks
yolo detect train \
  model=yolo26n.pt data=coco8.yaml \
  epochs=1 imgsz=320 batch=2 workers=0 device=cpu \
  project=runs/smoke name=upstream-sync exist_ok=True
```

如果仓库包含自定义模块，还应加载自己的模型 YAML，执行一次前向推理，并验证导出：

```bash
yolo export model=custom/configs/models/your_model.yaml format=torchscript
```

## 版本记录

每次同步在 `custom/UPSTREAM.md` 更新官方 commit、日期和验证结果。训练实验同时记录
自己的 Git commit、Python、PyTorch、CUDA 和数据集版本；这样旧权重可以准确复现。
