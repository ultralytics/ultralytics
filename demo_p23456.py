#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-P23456 演示
结合 p2 和 p6 生成 P2-P6 五层输出的 YOLO 模型
"""

from ultralytics import YOLO

def demo():
    # 加载自定义模型配置
    # 支持指定规模: n, s, m, l, x (例如 yolov8n-p23456.yaml)
    model = YOLO('ultralytics/cfg/models/v8/yolov8-p23456.yaml')
    
    print("=" * 60)
    print("YOLOv8-P23456 模型信息 (5层输出: P2-P6)")
    print("=" * 60)
    
    # 打印详细信息
    model.info(verbose=True)
    
    print("\n✅ 模型成功创建!")
    print("📌 输出层级: P2(1/4), P3(1/8), P4(1/16), P5(1/32), P6(1/64)")
    print("📌 使用方法:")
    print("   1. 训练: model.train(data='coco.yaml', epochs=100)")
    print("   2. 推理: model.predict('image.jpg')")
    print("   3. 导出: model.export(format='onnx')")
    
    return model

if __name__ == '__main__':
    model = demo()
