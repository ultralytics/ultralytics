# Project model YAML files

把项目自己的模型 YAML 放在这里，并使用明确的版本名，例如：

```text
yolo26n_custom_v1.yaml
yolo26n_custom_attention_v2.yaml
```

推荐的演进顺序：

1. 只使用现有模块时，复制对应官方 YAML 后仅修改层连接或类别数。
2. 新增模块时，把 Python 模块放在 `ultralytics/nn/modules/`，再在 YAML 中引用它。
3. 需要训练逻辑变化时，新增派生 Trainer，而不是复制整个官方训练器。

不要把 `*.pt` 权重放入这个目录；权重由实验制品库管理，模型 YAML 和对应 Git commit
一起保存。
