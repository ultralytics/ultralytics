# 电动自行车整车抠图

该工具使用官方 `yolo11l-seg.pt` 实例分割模型，只推理 COCO `class_id=3`（`motorcycle`）。检出多个实例时选择置信度最高的一个，并使用同一结果索引的掩码生成透明背景 PNG。

## 功能

- 单张图片：上传图片后查看整车框、实例掩码和透明抠图，可直接下载 PNG。
- 批量目录：通过浏览器目录选择器递归读取图片，点击“一键批量抠图”后逐张处理并显示进度，可随时停止后续图片。
- 批量结果：未检出整车的图片不中断任务；最终 ZIP 保留原目录相对结构，并附带覆盖全部登记图片的 `manifest.json` 成功/失败清单。
- 产物留存：单张预览与透明 PNG、批量 ZIP 与清单保存到 `runs/`；上传原图和批量松散 PNG 会及时清理，不修改用户原目录。
- 异常回收：批量归档响应中断会自动重试；服务启动时及运行期间每分钟检查一次，超过 1 小时没有继续请求的未完成批次会释放内存并删除临时文件，已完成 ZIP 与清单保留。
- 输入限制：支持 JPG、JPEG、PNG、BMP、WebP；单张文件最大 20 MB、最大 2000 万像素，单个批次最多 2000 张。

## 模型权重

权重固定位置：

```text
validate/ebike_yolo_seg_cutout/models/yolo11l-seg.pt
```

当前已下载权重的校验信息：

```text
大小: 56,096,965 bytes
SHA-256: cabe90049795dfc9a370b7934d6dec7f6b9e44a20e573b0ff81b7e205512c872
```

权重受仓库 `*.pt` 忽略规则管理。新环境使用临时文件下载，校验大小、SHA-256 和 ZIP 完整性后再原子替换：

```bash
(
  set -e
  model_target="validate/ebike_yolo_seg_cutout/models/yolo11l-seg.pt"
  model_download="${model_target}.download"
  trap 'rm -f "${model_download}"' EXIT
  curl -L --fail --retry 3 \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l-seg.pt \
    -o "${model_download}"
  test "$(wc -c < "${model_download}" | tr -d ' ')" = "56096965"
  echo "cabe90049795dfc9a370b7934d6dec7f6b9e44a20e573b0ff81b7e205512c872  ${model_download}" | shasum -a 256 -c -
  unzip -t "${model_download}"
  mv "${model_download}" "${model_target}"
  trap - EXIT
)
```

## 启动

在仓库根目录执行：

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python validate/ebike_yolo_seg_cutout/app.py
```

浏览器访问：

```text
http://127.0.0.1:8768/
```

模型路径、目标类别、置信度、输入尺寸和端口均在 `app.py` 的 `build_service()` / `main()` 中直接配置，无需命令行参数。
页面连接服务后会实际加载模型，并校验其任务为实例分割且包含 3 号类别；权重损坏或类别不符时不会创建批量任务。替换权重后需重启服务。

## 校验

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest -q tests/test_ebike_yolo_seg_cutout.py
uvx ruff check validate/ebike_yolo_seg_cutout/app.py tests/test_ebike_yolo_seg_cutout.py
```

COCO `motorcycle` 是通用类别，不等同于专门训练的电动自行车类别。遮挡严重、尺寸过小、特殊角度或合格证排版图仍可能漏检；推理异常、浏览器读取失败、请求中断和停止后未处理的图片也会作为失败项写入批量清单。
