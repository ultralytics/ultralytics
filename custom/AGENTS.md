# Project-specific agent instructions

本仓库是 Ultralytics 的项目 fork，用于在官方 YOLO 实现基础上维护自己的网络结构、公开数据集实验和私有数据集训练。根目录的 `AGENTS.md` 仍是通用规则和上游项目规则的最高约束；本文件只补充本 fork 的目标、分支和数据约定。

## 目标和边界

- 优先复用官方 `ultralytics/` 的训练、验证、预测、导出和数据处理能力。
- 只在需求明确时修改网络结构；先尝试新增模型 YAML，只有现有模块无法表达时才新增 Python 模块。
- 私有数据、权重、实验输出和机器本地路径不进入 Git。
- 上游代码保持尽量少的本地修改，方便将 `upstream/main` rebase 到本分支。

## 远程和分支

- `upstream` 必须指向官方仓库：`https://github.com/ultralytics/ultralytics.git`。
- `origin` 应指向用户自己的 fork；如果它仍指向官方仓库，禁止直接 push，先让用户配置自己的 URL。
- `main` 是稳定分支，禁止直接修改、直接 push 或 force push。
- 新工作必须从最新 `main` 创建 `feature/<topic>` 分支，并在隔离 worktree 中完成。
- 同步上游时创建 `sync/upstream-YYYY-MM-DD` 分支，在该分支执行 `git fetch upstream --prune --tags` 和 `git rebase upstream/main`，验证通过后再合并到 `main`。
- 不要 reset、revert 或覆盖别人尚未审查的提交；push 前先 `git pull --rebase`。

开始工作前检查：

```bash
git status --short --branch
git worktree list
git remote -v
```

## 自定义网络

1. 只调整已有层时，新增 `custom/configs/models/*.yaml`。
2. 新增层时，将实现放入 `ultralytics/nn/modules/`，在模块 `__init__.py` 导出，并在 `ultralytics/nn/tasks.py` 注册；只有需要通道缩放或 YAML 重复次数时才修改 `parse_model()` 的对应集合。
3. 训练逻辑变化优先通过派生 Trainer 传给 `model.train(..., trainer=...)`，不要复制整个官方训练器。
4. 每个模型 YAML 都要用公开数据集完成一次最小建模或训练验证，并记录对应 Git commit。

## 数据和权重

- 从 `custom/configs/datasets/private.example.yaml` 复制 `private.yaml`，配置私有数据集。
- `private.yaml`、数据目录、`runs/`、`*.pt` 和导出文件已被忽略；不要提交它们。
- 保持类别编号和类别名称稳定。实验记录数据集版本、代码 commit、Python、PyTorch 和 CUDA 版本。

## 必要验证

代码修改后至少运行：

```bash
yolo checks
python3 -m compileall -q ultralytics
```

模型或训练流程变化后，再运行一次 CPU 小规模 COCO8 训练：

```bash
yolo detect train model=yolo26n.pt data=coco8.yaml \
    epochs=1 imgsz=320 batch=2 workers=0 device=cpu \
    project=runs/smoke name=agent-check exist_ok=True
```

提交前检查 `git diff --check` 和完整 diff。不要为可逆的小改动新增与实现重复的测试。

## 维护文档

- 总体维护流程：`custom/README.md`
- 上游同步记录：`custom/UPSTREAM.md`
- 上游同步脚本：`custom/scripts/sync_upstream.sh`
