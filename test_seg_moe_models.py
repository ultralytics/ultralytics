"""
测试 yolov8-seg-p23456-moe 和 yolov8-seg-p23456-neck-moe 模型训练
使用 0814lake 数据集，在 GPU 2,3 上训练.
"""

import os

import torch

from ultralytics import YOLO


def test_model(model_yaml, model_name, data_yaml, device, batch_size=64, epochs=3):
    """测试单个模型的训练.

    Args:
        model_yaml: 模型配置文件路径
        model_name: 模型名称（用于日志）
        data_yaml: 数据集配置文件路径
        device: 使用的GPU设备
        batch_size: 批次大小
        epochs: 训练轮数
    """
    print(f"\n{'=' * 80}")
    print(f"开始测试: {model_name}")
    print(f"模型配置: {model_yaml}")
    print(f"数据集: {data_yaml}")
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"{'=' * 80}\n")

    try:
        # 创建模型
        model = YOLO(model_yaml)

        # 开始训练
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            device=device,
            project="runs/segment",
            name=f"{model_name}_test",
            exist_ok=True,
            verbose=True,
            save=True,
            save_period=1,  # 每个epoch都保存
        )

        print(f"\n✓ {model_name} 训练成功完成！")
        print(f"模型保存位置: runs/segment/{model_name}_test")

        # 验证保存的模型
        weights_path = f"runs/segment/{model_name}_test/weights/last.pt"
        if os.path.exists(weights_path):
            print(f"✓ 模型权重文件已保存: {weights_path}")

            # 尝试加载保存的模型
            YOLO(weights_path)
            print("✓ 已成功加载保存的模型")
        else:
            print(f"✗ 警告: 未找到模型权重文件: {weights_path}")

        return True

    except Exception as e:
        print(f"\n✗ {model_name} 训练失败！")
        print(f"错误信息: {e!s}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数."""
    # 配置参数
    data_yaml = "/slow_disk2/ccl/datasets/0814lake/data.yaml"
    device = "2,3"  # 使用2号和3号GPU
    batch_size = 64
    epochs = 3  # 测试用, 只训练3个epoch

    # 检查数据集文件是否存在
    if not os.path.exists(data_yaml):
        print(f"错误: 数据集文件不存在: {data_yaml}")
        return

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法使用GPU训练")
        return

    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"当前CUDA版本: {torch.version.cuda}")

    # 测试两个模型
    models_to_test = [
        {"yaml": "ultralytics/cfg/models/v8/yolov8-seg-p23456-moe.yaml", "name": "yolov8-seg-p23456-moe"},
        {"yaml": "ultralytics/cfg/models/v8/yolov8-seg-p23456-neck-moe.yaml", "name": "yolov8-seg-p23456-neck-moe"},
    ]

    results = {}

    for model_info in models_to_test:
        model_yaml = model_info["yaml"]
        model_name = model_info["name"]

        if not os.path.exists(model_yaml):
            print(f"错误: 模型配置文件不存在: {model_yaml}")
            results[model_name] = False
            continue

        success = test_model(
            model_yaml=model_yaml,
            model_name=model_name,
            data_yaml=data_yaml,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
        )

        results[model_name] = success

    # 打印总结
    print(f"\n{'=' * 80}")
    print("测试总结")
    print(f"{'=' * 80}")
    for model_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{model_name}: {status}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
