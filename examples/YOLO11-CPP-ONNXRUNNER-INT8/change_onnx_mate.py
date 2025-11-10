import onnx

# 加载ONNX模型
model_path = "best_int8.onnx"
model = onnx.load(model_path)

# 判断时候有METADATA数据
"""
删除走下面的流程
if len(model.metadata_props) > 0:
    a = 0
    for item in model.metadata_props:

        if item.key == "names":
            continue
        elif item.key == "imgsz":
            continue
        else:
            item.Clear()
"""
want_add_names = "{0: '关闭按钮', 1: '退出对局按钮', 2: '方向摇杆', 3: '攻击按钮', 4: '游戏时间和比分区', 5: '帧率区域', 6: '时延区域', 7: '设置按钮', 8: '失败结束标识', 9: '点击继续按钮', 10: '确定按钮', 11: '胜利结束标识', 12: '退出对局按钮（不可用）', 13: '对战开始按钮', 14: '英雄选择确认按钮（不可用）', 15: '被选中英雄', 16: '英雄备选区域', 17: '游戏主界面选区开始按钮', 18: '游戏资源下载提示标签', 19: '标准模式按钮', 20: '难度1', 21: '英雄选择确认按钮（可用）', 22: '阵亡提示', 23: '人机对战选择按钮', 24: '取消按钮', 25: '荣耀战令标识', 26: '断线重连标识', 27: '信息提示框', 28: '5v5模式选择按钮', 29: '3v3模式选择按钮', 30: '游戏结束后返回大厅按钮', 31: '微信登录按钮', 32: 'qq登录按钮', 33: '标准模式'}"
image_size = "[640, 640]"
meta = model.metadata_props.add()
meta.key, meta.value = "names", want_add_names
meta = model.metadata_props.add()
meta.key, meta.value = "imgsz", image_size
meta = model.metadata_props.add()
meta.key, meta.value = "author", "Ultralytics"
meta = model.metadata_props.add()
meta.key, meta.value = "license", "AGPL-3.0 License (https://ultralytics.com/license)"
meta = model.metadata_props.add()
meta.key, meta.value = (
    "description",
    "Ultralytics YOLOv12n model trained on /mnt/d/ultralytics/ultralytics/datasets/YOLODataset/dataset.yaml",
)
meta = model.metadata_props.add()
meta.key, meta.value = "date", "2025-09-23T11:02:06.864770"
meta = model.metadata_props.add()
meta.key, meta.value = "version", "8.3.203"
meta = model.metadata_props.add()
meta.key, meta.value = "docs", "https://docs.ultralytics.com"
meta = model.metadata_props.add()
meta.key, meta.value = "stride", "32"
meta = model.metadata_props.add()
meta.key, meta.value = "task", "detect"
meta = model.metadata_props.add()
meta.key, meta.value = "batch", "1"
meta = model.metadata_props.add()
meta.key, meta.value = "args", "{'batch': 1, 'fraction': 0.2, 'half': False, 'int8': True, 'nms': False}"
meta = model.metadata_props.add()
meta.key, meta.value = "channels", "3"

# 保存修改后的模型
onnx.save(model, "best_int8_add.onnx")
