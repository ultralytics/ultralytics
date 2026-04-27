[根目录](../CLAUDE.md) > **ultralytics**

# ultralytics 模块

## 模块职责

`ultralytics/` 为上游框架源码与文档镜像，承担底层训练/推理能力与官方示例。

## 入口与启动

- 工程元数据：`ultralytics/pyproject.toml`
- 主包目录：`ultralytics/ultralytics/**`
- 官方测试：`ultralytics/tests/**`
- 示例：`ultralytics/examples/**`
- 文档：`ultralytics/docs/**`

## 对外接口

- `project.scripts` 暴露 `yolo` / `ultralytics` 命令入口；
- 配置、模型、数据集模板分布在 `ultralytics/ultralytics/cfg/**`；
- 本仓业务脚本通过 `from ultralytics import YOLO` 调用核心 API。

## 关键依赖与配置

- 依赖在 `pyproject.toml` 声明（numpy/torch/opencv 等）；
- 可选依赖包括 export/solutions/logging/typing；
- 许可：AGPL-3.0。

## 数据模型

该模块主要是算法框架代码，无业务数据库模型。

## 测试与质量

- 自带官方 tests 目录；
- 本次仅进行了模块级扫描，未深入到每个 engine/model 子包实现细节。

## 常见问题 (FAQ)

1. 本地修改后行为不一致：需确认导入的是仓内包还是环境 pip 包。  
2. 依赖冲突：优先以 `pyproject.toml` 版本约束为准。  
3. 升级成本高：建议业务脚本层与上游改动分支隔离。

## 相关文件清单

- `ultralytics/pyproject.toml`
- `ultralytics/README.md`
- `ultralytics/ultralytics/__init__.py`
- `ultralytics/tests/test_python.py`

## 变更记录 (Changelog)

- 2026-04-15 13:54:17：新建模块文档，记录上游镜像定位与扫描边界。